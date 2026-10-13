#!/usr/bin/env python3
"""CSI300 cash-notional full spreads and aggregate-first calibration (Zhao protocol).

See projects/liquidity/csi300/README.md for input contracts and runnable examples.
The default audit never replays the archive. Explicit build dates are required.
"""
from __future__ import annotations

import fcntl
import argparse
from collections import Counter, OrderedDict
from dataclasses import asdict
import hashlib
import json
import math
import time
import os
from pathlib import Path
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'projects/liquidity'))
sys.path.insert(0, str(ROOT))
import replay
import shanghai
import historical_books
import feed_quality
from scripts.liquidity import csi300_incremental as incremental
# Re-export for existing callers; selection logic lives with weight generation.
from scripts.liquidity.csi300_backprop_weights import weight_schedule, members_on, weights_on

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
PHASES = {'am': (34200000, 41400000), 'pm': (46800000, 53820000)}
# Intersection of continuous sessions; deliberately exclude 14:57 onward on BOTH exchanges.
WEIGHT_COLUMNS = ['month', 'code', 'name', 'weight_pct', 'industry', 'source']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def read_weights(path):
    """Read monthly backcast CSV or legacy workbook, preserving provenance."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        a = pd.read_csv(path, dtype={'date':str, 'code':str}, encoding='utf-8-sig')
        required = {'date','code','weight_pct','in_anchor'}
        if not required <= set(a):
            raise ValueError(f'Monthly weights CSV requires {sorted(required)}')
        if not a.date.str.fullmatch(r'\d{8}').all():
            raise ValueError('Invalid YYYYMMDD in monthly weights')
        pd.to_datetime(a.date,format='%Y%m%d',errors='raise')
        a['snapshot_date'] = a.date
        a['month'] = a.date.str[:6]
        if (a.groupby('month').date.nunique()!=1).any():
            raise ValueError('Expected one snapshot date per month')
        a['in_anchor'] = pd.to_numeric(a.in_anchor,errors='raise')
        if not a.in_anchor.isin([0,1]).all():raise ValueError('in_anchor must be 0 or 1')
        a['name'] = ''; a['industry'] = ''
        a['source'] = np.where(a.in_anchor.eq(1),'backcast: anchor-price-ratio','backcast: non-anchor-median')
    else:
        records = []
        with zipfile.ZipFile(path) as z:
            strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                strings = [''.join(x.text or '' for x in si.findall('.//m:t', NS))
                           for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
            rel = {r.get('Id'): r.get('Target') for r in ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
            for sheet in ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets', NS):
                target = rel[sheet.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')]
                target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
                for row in ET.fromstring(z.read(target)).findall('m:sheetData/m:row', NS):
                    cells = {}
                    for cell in row:
                        value = cell.find('m:v', NS)
                        value = value.text if value is not None else ''
                        if cell.get('t') == 's' and value:
                            value = strings[int(value)]
                        elif cell.get('t') == 'inlineStr':
                            value = ''.join(x.text or '' for x in cell.findall('.//m:t', NS))
                        if cell.find('m:f', NS) is not None:
                            raise ValueError('Historical weights must contain stored observations, not formulas')
                        cells[re.sub(r'\d', '', cell.get('r'))] = value
                    if cells.get('A') in ('年月', '', None):
                        continue
                    values = [cells.get(c, '') for c in 'ABCDEF']
                    records.append(values)
        a = pd.DataFrame(records, columns=WEIGHT_COLUMNS)
    if not a.month.str.fullmatch(r'\d{6}').all():
        raise ValueError('Invalid YYYYMM in weights')
    pd.to_datetime(a.month, format='%Y%m', errors='raise')
    a.code = a.code.str.replace(r'\.0$', '', regex=True).str.zfill(6)
    if not a.code.str.fullmatch(r'\d{6}').all() or a.duplicated(['month', 'code']).any():
        raise ValueError('Invalid or duplicate constituent identifiers')
    a.weight_pct = pd.to_numeric(a.weight_pct, errors='raise')
    if not (np.isfinite(a.weight_pct) & (a.weight_pct > 0)).all():
        raise ValueError('Nonpositive/nonfinite weights')
    a['exchange'] = np.where(a.code.str.startswith('6'), 'SH', 'SZ')
    if not a.code.str.startswith(('0', '3', '6')).all():
        raise ValueError('Unsupported exchange code')
    a['estimated'] = a.source.str.contains('反推|估算|back.?cast', case=False, regex=True)
    totals = a.groupby('month').weight_pct.transform('sum')
    a['weight'] = a.weight_pct / totals
    return a


def weight_audit(weights, out):
    rows = []
    for month, g in weights.groupby('month', sort=True):
        digest = hashlib.sha256(g.sort_values('code')[['code', 'weight_pct']].to_csv(index=False).encode()).hexdigest()
        rows.append(dict(month=month, count=len(g), weight_sum_pct=g.weight_pct.sum(),
                         sh_count=int((g.exchange == 'SH').sum()), sz_count=int((g.exchange == 'SZ').sum()),
                         estimated_rows=int(g.estimated.sum()), snapshot_hash=digest,
                         sources=' | '.join(sorted(g.source.unique())),
                         usable_structure=len(g) == 300 and abs(g.weight_pct.sum() - 100) <= .5))
    a = pd.DataFrame(rows)
    a.to_csv(out / 'weights_audit.csv', index=False)
    weights.to_parquet(out / 'weights_normalized.parquet', index=False)
    write_json(out / 'weights_summary.json', dict(months=len(a), rows=len(weights),
               first_month=a.month.min(), last_month=a.month.max(),
               estimated_months=int((a.estimated_rows > 0).sum()),
               invalid_months=a.loc[~a.usable_structure, 'month'].tolist(),
               duplicate_snapshot_groups=[g.month.tolist() for _, g in a.groupby('snapshot_hash') if len(g) > 1]))
    return a


MEASURE_VERSION = 'zhao-cash-full-spread-v1'


def missing_measure(state):
    return dict(sell_bps=np.nan, buy_bps=np.nan, raw_bps=np.nan, capped_bps=np.nan,
                midpoint=np.nan, best_bid=np.nan, best_ask=np.nan, quoted_bps=np.nan,
                sell_shares=np.nan, buy_shares=np.nan, sell_cash=np.nan, buy_cash=np.nan,
                sell_average_price=np.nan, buy_average_price=np.nan,
                sell_levels=0, buy_levels=0, sell_capacity=np.nan, buy_capacity=np.nan,
                insuff_sell=False, insuff_buy=False, cap_applied=False, state=state)


def displayed_costs(bids, asks, notional, cap, depth_complete):
    """Cash sweep at common per-stock/per-side RMB q; no lot rounding.

    The optional cap applies once to the full spread. Unknown deeper depth is
    not a genuine shortage. A complete one-sided book has no finite midpoint.
    """
    if not math.isfinite(notional) or notional <= 0 or not math.isfinite(cap) or cap <= 0:
        raise ValueError('Notional and cap must be finite and positive')
    r = missing_measure('ok')
    if any(not math.isfinite(p) or p <= 0 or not math.isfinite(q) or q <= 0
           for levels in (bids, asks) for p,q in levels.items()):
        r['state']='invalid_levels'; return r
    for side, levels, reverse in [('sell',bids,True),('buy',asks,False)]:
        remaining=float(notional); cash=shares=0.; count=0
        capacity=sum(p*q for p,q in levels.items())
        for p in sorted(levels, reverse=reverse):
            take_cash=min(remaining, p*levels[p]); shares+=take_cash/p
            cash+=take_cash; remaining-=take_cash; count+=1
            if remaining <= 0: break
        r.update({side+'_shares':shares, side+'_cash':cash, side+'_levels':count,
                  side+'_capacity':capacity, side+'_average_price':cash/shares if shares else np.nan,
                  'insuff_'+side:bool(remaining>0 and depth_complete)})
    if bids:r['best_bid']=max(bids)
    if asks:r['best_ask']=min(asks)
    if not bids or not asks:
        r['state']='nontrading' if depth_complete else 'unknown_depth'
        if depth_complete:r.update(capped_bps=float(cap),cap_applied=True)
        return r
    if r['best_bid'] >= r['best_ask']:
        r['state']='crossed';r['insuff_sell']=r['insuff_buy']=False;return r
    mid=(r['best_bid']+r['best_ask'])/2
    r.update(midpoint=mid,quoted_bps=1e4*(r['best_ask']-r['best_bid'])/mid)
    short=any(r[side+'_cash'] < notional and not math.isclose(r[side+'_cash'],notional,rel_tol=1e-12)
              for side in ['sell','buy'])
    if short:
        r['state']='shortage' if depth_complete else 'unknown_depth'
        if depth_complete:r.update(capped_bps=float(cap),cap_applied=True)
        return r
    sell=1e4*(1-r['sell_average_price']/mid)
    buy=1e4*(r['buy_average_price']/mid-1)
    raw=sell+buy
    r.update(sell_bps=sell,buy_bps=buy,raw_bps=raw,capped_bps=min(raw,cap),cap_applied=raw>cap)
    return r


def endpoints(grid_ms, session):
    start, end = PHASES[session]
    return np.arange(start + grid_ms, end + 1, grid_ms, dtype=np.int64)


def sample_snapshots(path, times, quantities, cap, session):
    """Certified state intervals explicitly bound every permitted carry-forward."""
    a = pd.read_parquet(path)
    needed = {'session', 't_ms', 'valid_until_ms', 'book_valid', 'depth_complete',
              'bid_prices', 'bid_sizes', 'ask_prices', 'ask_sizes', 'validation_source'}
    if not needed <= set(a):
        raise ValueError(f'Snapshot input missing {sorted(needed-set(a))}')
    a = a[a.session == session].sort_values('t_ms').reset_index(drop=True)
    if a.t_ms.duplicated().any() or (a.valid_until_ms < a.t_ms).any():
        raise ValueError('Invalid certified state intervals')
    indices = np.searchsorted(a.t_ms.to_numpy(), times, side='right') - 1
    results = []; cache = {}
    for t, ix in zip(times, indices):
        if ix < 0 or t > a.valid_until_ms.iloc[ix]:
            results.append([missing_measure('uncertified_interval')] * len(quantities)); continue
        if ix not in cache:
            r = a.iloc[ix]
            if not isinstance(r.book_valid, (bool, np.bool_)) or not isinstance(r.depth_complete, (bool, np.bool_)):
                raise ValueError('Snapshot validity columns must be booleans')
            if not r.book_valid or not isinstance(r.validation_source, str) or not r.validation_source.strip():
                cache[ix] = [missing_measure('invalid_snapshot')] * len(quantities)
            else:
                for side in ['bid', 'ask']:
                    if len(r[side+'_prices']) != len(r[side+'_sizes']) or len(set(r[side+'_prices'])) != len(r[side+'_prices']):
                        raise ValueError('Mismatched or duplicate snapshot price levels')
                bids = dict(zip(r.bid_prices, r.bid_sizes)); asks = dict(zip(r.ask_prices, r.ask_sizes))
                cache[ix] = [displayed_costs(bids, asks, q, cap, r.depth_complete) for q in quantities]
        results.append(cache[ix])
    return results


def stock_job(job):
    """Workers persist panels; Ray returns small metadata, not stock tensors."""
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    out = Path(job['out']); code = job['code']; day = job['day']
    folder = out / 'constituents' / day; folder.mkdir(parents=True, exist_ok=True)
    target = folder / f'{code}.parquet'; audit_path = folder / f'{code}.json'
    snapshot = Path(job['snapshots']) / day / f'{code}.parquet' if job['snapshots'] else None
    sources = [snapshot] if snapshot and snapshot.exists() else [
        Path(job['root'])/feed/f'year={day[:4]}'/f'{day}.parquet' for feed in ['orders','trades','cancels']]
    fingerprint = {str(p): [p.stat().st_size,p.stat().st_mtime_ns] if p.exists() else None for p in sources}
    cache_key = incremental.stock_key(job,fingerprint,snapshot and snapshot.exists()) if 'stock_logic_version' in job else None
    if audit_path.exists():
        old = json.loads(audit_path.read_text())
        reuse = incremental.reusable(old,job,cache_key,fingerprint,Path(old.get('panel_path',target))) if cache_key else (
            target.exists() and old.get('run_hash')==job['run_hash'] and old.get('complete') and old.get('input_file_stats')==fingerprint)
        if reuse:
            return dict(old, _cache_reused=True) if cache_key else old
    borrowed = incremental.borrow(job,cache_key,fingerprint) if cache_key and job.get('reuse_from') else None
    if borrowed is not None:
        write_json(audit_path,borrowed)
        return dict(borrowed,_cache_reused=True)
    audit = dict(code=code, day=day, run_hash=job['run_hash'], source='none', certified=False,
                 input_file_stats=fingerprint,stock_cache_key=cache_key,
                 replay_version=job.get('replay_version'),stock_logic_version=job.get('stock_logic_version'))
    quality = job.get('feed_quality')
    if quality:
        audit['feed_quality'] = quality
    if job['exchange'] == 'SH' and not job.get('shanghai_residual') and not (snapshot and snapshot.exists()):
        audit.update(status='unsupported_SH_reconstruction', complete=True)
        write_json(audit_path, audit); return audit
    if not (snapshot and snapshot.exists()) and not job['diagnostic']:
        audit.update(status='missing_certified_snapshot', complete=True)
        write_json(audit_path, audit); return audit
    quantities = job['notionals']
    audit.update(target_notionals=quantities, measure_version=MEASURE_VERSION,
                 notional_rule='executed_cash', rounding_rule='fractional_terminal_fill')
    rp = None
    if snapshot and snapshot.exists():
        audit.update(source=str(snapshot), source_hash=sha(snapshot), certified=True)
    else:
        frames = replay.read_day(Path(job['root']), day, code, labels=[f'{code}.SZ', f'{code}.SH'])
        if job['exchange']=='SH':
            if quality:
                audit['source_ledger'] = shanghai.audit(frames)
                try:
                    frames, audit['source_sanitization'] = feed_quality.sanitize_shanghai(frames)
                except ValueError as exc:
                    audit.update(status='ambiguous_source_events',error=str(exc),complete=True)
                    write_json(audit_path,audit); return audit
                if all(len(v)==0 for v in frames.values()):
                    evidence=quality.get('empty_evidence',{})
                    audit.update(status='suspended' if evidence.get('status')=='suspended' else 'missing_feed',
                                 ledger_clean=False,complete=True,empty_evidence=evidence)
                    write_json(audit_path,audit); return audit
            if day.startswith('2020'):
                audit['original_ledger'] = shanghai.audit(frames)
                try:
                    frames = historical_books.shanghai_clock_frames(frames)
                except ValueError as exc:
                    audit.update(status='unresolved_historical_ordering', error=str(exc), complete=True)
                    write_json(audit_path, audit); return audit
                audit['historical_adapter'] = historical_books.VERSION
            ledger = shanghai.audit(frames)
            audit['ledger'] = ledger; audit['ledger_clean'] = ledger['clean']
            audit['trade_linkage'] = None; audit['cancel_linkage'] = None
            audit['source'] = shanghai.VERSION
        else:
            ledger = replay.audit_day(Path(job['root']), day, code, frames=frames)
            audit['ledger'] = asdict(ledger)
            audit['ledger_clean'] = bool(ledger.clean and not ledger.fill_refs_wrong_side and not ledger.cancel_refs_wrong_side and ledger.n_orders)
            audit['trade_linkage'] = 1-ledger.unknown_fill_refs/max(2*ledger.n_trades, 1)
            audit['cancel_linkage'] = 1-ledger.unknown_cancel_refs/max(ledger.n_cancels, 1)
            audit['source'] = 'SZ_replay_provisional'
        audit['reconstructed_fraction'] = float((frames['orders']['来源'] != 'orig').mean()) if len(frames['orders']) else None
        # Entire-day checks are retrospective construction QA, not forecast-origin features.
        if not audit['ledger_clean']:
            audit.update(status='failed_ledger', complete=True); write_json(audit_path, audit); return audit
        if set(frames['orders']['交易所'].dropna().unique()) != {job['exchange']}:
            raise ValueError('Unexpected exchange in raw replay')
        replay_class = shanghai.Replayer if job['exchange']=='SH' else replay.Replayer
        if job['exchange']=='SZ' and historical_books.cage_applies(day,code):
            replay_class = historical_books.CageReplayer
            audit['historical_adapter'] = historical_books.VERSION
        rp = replay_class(day, code, frames)
        if np.any(np.diff(rp.t) < 0):
            raise ValueError('Nonmonotone event clock')
        audit['off_100ms_fraction'] = float(np.mean(rp.t % 100 != 0))
    panels = []; counts = Counter()
    for session in PHASES:
        times = endpoints(job['grid_ms'], session)
        if rp is None:
            values = sample_snapshots(snapshot, times, quantities, job['cap'], session)
        else:
            values = []; previous_pos = -1; cached = None
            for t in times:
                rp.advance_to(t / 1000)
                quarantine = feed_quality.missing_state(quality,t)
                if quarantine:
                    values.append([missing_measure(quarantine)] * len(quantities))
                    continue
                if previous_pos != rp.pos:
                    previous_pos = rp.pos
                    bad = rp.n_unknown_fill or rp.n_unknown_cancel or getattr(rp,'n_negative_debits',0)
                    cached = [missing_measure('ledger_uncertain')] * len(quantities) if bad else [
                        displayed_costs(rp.bid_lv, rp.ask_lv, q, job['cap'], False) for q in quantities]
                values.append(cached)
        for j, spec in enumerate(job['size_specs']):
            data = [v[j] for v in values]
            p = pd.DataFrame(data)
            p = p.assign(date=day,code=code,exchange=job['exchange'],session=session,t_ms=times,
                         size_spec=spec,notional_rmb=quantities[j],weight=job['weight'],
                         measure_version=MEASURE_VERSION,notional_rule='executed_cash',
                         rounding_rule='fractional_terminal_fill',grid_ms=job['grid_ms'],cap_bps=job['cap'])
            p['certified'] = audit['certified']; p['valid'] = np.isfinite(p.raw_bps)
            counts.update(p.state); panels.append(p)
    panel = pd.concat(panels, ignore_index=True)
    if rp is not None:
        audit['runtime_unknown_fill'] = rp.n_unknown_fill
        audit['runtime_unknown_cancel'] = rp.n_unknown_cancel
        audit['runtime_negative_debits'] = getattr(rp,'n_negative_debits',0)
    ordered = panel.pivot(index=['session','t_ms'],columns='size_spec',values='raw_bps').reindex(columns=job['size_specs']).to_numpy()
    difference = np.diff(ordered,axis=1)
    temp = target.with_suffix('.tmp.parquet'); panel.to_parquet(temp, index=False); temp.replace(target)
    audit.update(dormant_added=getattr(rp,'n_dormant_added',0), dormant_activated=getattr(rp,'n_activated',0))
    audit.update(status='constructed', state_counts=dict(counts), rows=len(panel), valid_rows=int(panel.valid.sum()),
                 size_monotonicity_pairs=int(np.isfinite(difference).sum()),
                 size_monotonicity_violations=int((difference < -1e-8).sum()),
                 complete=True, panel_path=str(target), book_reconstruction_version=job['book_hash'])
    audit['panel_file_stats'] = incremental.file_stats(target)
    write_json(audit_path, audit)
    return audit


def run_jobs(jobs, backend, workers):
    if backend == 'serial':
        return [stock_job(j) for j in tqdm(jobs, desc='Stock books')]
    import ray
    owned = not ray.is_initialized()
    if owned:
        ray.init(num_cpus=workers, include_dashboard=False, log_to_driver=False,
                 runtime_env={'env_vars':{'PYTHONPATH':str(ROOT/'projects/liquidity')+':'+str(ROOT)}})
    remote = ray.remote(num_cpus=1)(stock_job)
    iterator = iter(jobs); pending = {}; results = []
    try:
        for job in list_next(iterator, min(workers, len(jobs))):
            pending[remote.remote(job)] = job['code']
        with tqdm(total=len(jobs), desc='Stock books (Ray)') as progress:
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                ref = ready[0]; pending.pop(ref); results.append(ray.get(ref)); progress.update(1)
                job = next(iterator, None)
                if job is not None:
                    pending[remote.remote(job)] = job['code']
    finally:
        if owned:
            ray.shutdown()
    return results


def list_next(iterator, count):
    for _ in range(count):
        item = next(iterator, None)
        if item is None:
            return
        yield item


def fit_normalization(out, start, end, bin_ms, min_days, radius, floor):
    """Calibrate the log of the full verified aggregate, never stock composites."""
    from collections import defaultdict
    if min_days < 1 or radius < 0 or min(bin_ms,floor) <= 0:
        raise ValueError('Invalid normalization settings')
    pools=defaultdict(list);days=defaultdict(set)
    for path in tqdm(sorted((out/'index').glob('*.parquet')),desc='Calibrate aggregate'):
        if not start <= path.stem <= end:continue
        a=pd.read_parquet(path)
        if not (a.measure_version==MEASURE_VERSION).all():raise ValueError('Wrong index version')
        a=a[np.isfinite(a.liq_raw_vw) & (a.liq_raw_vw>0)].copy()
        a['bin']=((a.t_ms-a.session.map({k:v[0] for k,v in PHASES.items()})-1)//bin_ms).astype(int)
        a['log_cost']=np.log(a.liq_raw_vw)
        for key,g in a.groupby(['session','size_spec','bin']):
            pools[key].append(g.log_cost.to_numpy());days[key].add(path.stem)
    rows=[]
    for session,spec in sorted({k[:2] for k in pools}):
        nbins=int(np.ceil((PHASES[session][1]-PHASES[session][0])/bin_ms))
        for b in range(nbins):
            keys=[(session,spec,j) for j in range(max(0,b-radius),min(nbins,b+radius+1))]
            dates=set.union(set(),*(days[k] for k in keys))
            if len(dates)<min_days:continue
            values=np.concatenate([a for k in keys for a in pools[k]])
            center=float(np.median(values));mad=1.4826*float(np.median(abs(values-center)))
            rows.append(dict(session=session,size_spec=spec,bin=b,location=center,
                             scale=max(mad,floor),scale_floored=mad<floor,
                             calibration_days=len(dates),calibration_end=end))
    if not rows:raise ValueError('Insufficient full-coverage aggregate calibration days')
    pd.DataFrame(rows).to_parquet(out/'normalization.parquet',index=False)
    write_json(out/'normalization.json',dict(start=start,end=end,bin_ms=bin_ms,min_days=min_days,
        smoothing_radius=radius,scale_floor=floor,target='liq_raw_vw',c0_bps=1.,
        method='aggregate-first-log-median-MAD',measure_version=MEASURE_VERSION,
        construction_hash=sha(out/'run.json')))


class PanelReader:
    """Bounded per-aggregation cache; all other pandas operations unchanged."""
    columns = ['session', 'size_spec', 't_ms', 'weight', 'valid', 'raw_bps',
               'certified', 'sell_bps', 'buy_bps', 'capped_bps', 'quoted_bps', 'state', 'exchange']

    def __init__(self, pandas, limit_bytes=1024**3):
        self.pandas = pandas
        self.limit = limit_bytes
        self.cache = OrderedDict()
        self.bytes = 0
        self.reads = 0

    def __getattr__(self, name):
        return getattr(self.pandas, name)

    def read_parquet(self, path, **kwargs):
        filters = kwargs.get('filters')
        if set(kwargs) != {'filters'} or not filters or [f[:2] for f in filters] != [('session', '='), ('size_spec', '=')]:
            return self.pandas.read_parquet(path, **kwargs)
        key = str(path)
        if key not in self.cache:
            frame = self.pandas.read_parquet(path, columns=self.columns)
            self.reads += 1
            for col in ['session', 'size_spec', 'exchange', 'state']:
                frame[col] = frame[col].astype('category')
            size = int(frame.memory_usage(deep=True).sum())
            while self.cache and self.bytes + size > self.limit:
                _, (_, old_size) = self.cache.popitem(last=False)
                self.bytes -= old_size
            if size <= self.limit:
                self.cache[key] = (frame, size)
                self.bytes += size
        else:
            frame, _ = self.cache[key]
            self.cache.move_to_end(key)
        return frame[(frame.session == filters[0][2]) & (frame.size_spec == filters[1][2])].reset_index(drop=True)



def aggregate_day(out, day, weights, audits, args, specs):
    normal=meta=None
    if args.normalization:
        normal=pd.read_parquet(args.normalization)
        meta=json.loads(args.normalization.with_suffix('.json').read_text())
        if meta['end']>=day:raise ValueError('Normalization must end strictly before the output day')
        if meta.get('construction_hash')!=sha(out/'run.json') or meta.get('measure_version')!=MEASURE_VERSION:
            raise ValueError('Normalization belongs to a different construction')
    reader=PanelReader(pd);started=time.perf_counter();rows=[]
    total_by_exchange=weights.groupby('exchange').weight.sum().to_dict()
    weight_by_code=weights.set_index('code').weight.to_dict()
    for session in PHASES:
        times=endpoints(args.grid_ms,session);n=len(times)
        for spec in specs:
            keys=['coverage','certified','count','raw','sell','buy','ew','sh','sz','sh_cov','sz_cov',
                  'capped','capped_cov','capped_certified','capped_count','sh_cap_cov','sz_cap_cov',
                  'shortage','nontrading','unknown_depth','represented','technical','quote','quote_cov','quote_count',
                  'squared_weights']
            sums={key:np.zeros(n) for key in keys}
            top=np.zeros(n);top_code=np.full(n,'',dtype='U6')
            for audit in audits:
                if audit['status']!='constructed':continue
                a=reader.read_parquet(audit['panel_path'],filters=[('session','=',session),('size_spec','=',spec)])
                if len(a)!=n or not np.array_equal(a.t_ms,times):raise ValueError('Constituent grids do not align')
                w=float(weight_by_code[audit['code']]);ex=a.exchange.iloc[0].lower();valid=np.isfinite(a.raw_bps.to_numpy())
                raw=a.raw_bps.to_numpy();cert=a.certified.to_numpy(dtype=bool)
                cap_valid=np.isfinite(a.capped_bps.to_numpy());state=a.state.astype(str).to_numpy()
                sums['represented']+=w;sums['coverage']+=w*valid;sums['count']+=valid
                sums['certified']+=w*(valid & cert);sums[ex+'_cov']+=w*valid
                sums['squared_weights']+=w*w*valid
                for key,col in [('raw','raw_bps'),('sell','sell_bps'),('buy','buy_bps')]:
                    sums[key]+=np.where(valid,w*a[col].to_numpy(),0)
                contribution=np.where(valid,w*raw,0);sums[ex]+=contribution
                sums['ew']+=np.where(valid,raw,0)
                take=contribution>top;top[take]=contribution[take];top_code[take]=audit['code']
                sums['capped']+=np.where(cap_valid,w*a.capped_bps.to_numpy(),0)
                sums['capped_cov']+=w*cap_valid;sums['capped_count']+=cap_valid
                sums['capped_certified']+=w*(cap_valid & cert);sums[ex+'_cap_cov']+=w*cap_valid
                sums['shortage']+=w*(state=='shortage');sums['nontrading']+=w*(state=='nontrading')
                sums['unknown_depth']+=w*(state=='unknown_depth')
                sums['technical']+=w*~np.isin(state,['ok','shortage','nontrading','unknown_depth'])
                quote_valid=np.isfinite(a.quoted_bps.to_numpy()) & cert
                sums['quote']+=np.where(quote_valid,w*a.quoted_bps.to_numpy(),0)
                sums['quote_cov']+=w*quote_valid;sums['quote_count']+=quote_valid
            cov=sums['coverage'];capcov=sums['capped_cov']
            full=(sums['count']==len(weights)) & (sums['certified']>=1-1e-10)
            fullcap=(sums['capped_count']==len(weights)) & (sums['capped_certified']>=1-1e-10)
            available=(cov>=args.min_coverage-1e-12)&(sums['count']>=args.min_valid)
            cap_available=(capcov>=args.min_coverage-1e-12)&(sums['capped_count']>=args.min_valid)
            for ex in ['sh','sz']:
                total=total_by_exchange.get(ex.upper(),0)
                if total:
                    available &= sums[ex+'_cov']/total>=args.min_exchange_coverage-1e-12
                    cap_available &= sums[ex+'_cap_cov']/total>=args.min_exchange_coverage-1e-12
            divide=lambda x,d:np.divide(x,d,out=np.full(n,np.nan),where=d>0)
            a=pd.DataFrame(dict(timestamp=pd.to_datetime(day)+pd.to_timedelta(times,unit='ms'),date=day,
                session=session,t_ms=times,size_spec=spec,notional_rmb=float(spec.split('_',1)[1]),
                grid_ms=args.grid_ms,measure_version=MEASURE_VERSION,notional_rule='executed_cash',rounding_rule='fractional_terminal_fill',
                n_constituents=len(weights),n_valid=sums['count'].astype(int),weight_coverage=cov,finite_weight_coverage=cov,
                certified_weight_coverage=sums['certified'],capped_weight_coverage=capcov,
                shortage_weight=sums['shortage'],nontrading_weight=sums['nontrading'],unknown_depth_weight=sums['unknown_depth'],
                technical_invalid_weight=sums['technical']+np.maximum(0,1-sums['represented']),
                invalid_weight=np.maximum(0,1-cov),insuff_depth_weight=sums['shortage'],
                liq_raw_vw=np.where(full,sums['raw'],np.nan),liq_raw_ew=np.where(full,sums['ew']/len(weights),np.nan),
                liq_buy=np.where(full,sums['buy'],np.nan),liq_sell=np.where(full,sums['sell'],np.nan),
                liq_available_vw_bps=np.where(available,divide(sums['raw'],cov),np.nan),
                liq_available_ew_bps=np.where(available,divide(sums['ew'],sums['count']),np.nan),
                liq_available_buy_bps=np.where(available,divide(sums['buy'],cov),np.nan),
                liq_available_sell_bps=np.where(available,divide(sums['sell'],cov),np.nan),
                liq_capped_vw_bps=np.where(fullcap,sums['capped'],np.nan),
                liq_capped_available_vw_bps=np.where(cap_available,divide(sums['capped'],capcov),np.nan),
                liq_quoted_vw_bps=np.where(sums['quote_count']==len(weights),sums['quote'],np.nan),
                quote_weight_coverage=sums['quote_cov'],sh_weight_coverage=sums['sh_cov'],sz_weight_coverage=sums['sz_cov'],
                sh_contribution=np.where(full,sums['sh'],np.nan),sz_contribution=np.where(full,sums['sz'],np.nan),
                sh_available_contribution=np.where(available,divide(sums['sh'],cov),np.nan),
                sz_available_contribution=np.where(available,divide(sums['sz'],cov),np.nan),
                weight_hhi=divide(sums['squared_weights'],cov*cov),top_contributor=top_code,top_contribution=np.where(full,top,np.nan),
                membership_version=weights.membership_version.iloc[0],weight_version=weights.membership_version.iloc[0],
                weight_effective_time=weights.weight_effective_time.iloc[0],known_at=weights.known_at.iloc[0],
                weight_dates_assumed=bool(weights.weight_dates_assumed.iloc[0]) if 'weight_dates_assumed' in weights else True,
                book_reconstruction_version=args.book_hash,cap_bps=args.cap_bps,
                construction_status='diagnostic' if args.diagnostic else 'validated_input_construction'))
            for col in ['weight_imputed','weight_imputation_method','weight_lookahead_days','snapshot_date']:
                if col in weights:a[col]=weights[col].iloc[0]
            a['liq_raw_vw_bps']=a.liq_raw_vw;a['liq_raw_ew_bps']=a.liq_raw_ew
            a['liq_buy_bps']=a.liq_buy;a['liq_sell_bps']=a.liq_sell
            a['seasonal_location']=np.nan;a['seasonal_scale']=np.nan;a['liq_stress']=np.nan
            if normal is not None:
                g=normal[(normal.session==session)&(normal.size_spec==spec)].set_index('bin')
                aligned=g.reindex((times-PHASES[session][0]-1)//meta['bin_ms'])
                loc=aligned.location.to_numpy();scale=aligned.scale.to_numpy()
                positive=a.liq_raw_vw.to_numpy()>0
                logvalue=np.full(n,np.nan);np.log(a.liq_raw_vw.to_numpy(),out=logvalue,where=positive)
                a['seasonal_location']=loc;a['seasonal_scale']=scale;a['liq_stress']=(logvalue-loc)/scale
            a['is_release_eligible']=full & (not args.diagnostic) & (not weights.estimated.any())
            a.timestamp=a.timestamp.dt.tz_localize('Asia/Shanghai');rows.append(a)
    index=pd.concat(rows,ignore_index=True);(out/'index').mkdir(exist_ok=True)
    target=out/'index'/f'{day}.parquet';temporary=target.with_suffix('.tmp.parquet')
    index.to_parquet(temporary,index=False);temporary.replace(target)
    summary=index.groupby(['session','size_spec']).agg(rows=('t_ms','size'),mean_coverage=('weight_coverage','mean'),
        min_coverage=('weight_coverage','min'),valid_raw=('liq_raw_vw','count'),valid_available=('liq_available_vw_bps','count'),
        valid_stress=('liq_stress','count'),mean_raw_bps=('liq_raw_vw','mean'),release_rows=('is_release_eligible','sum')).reset_index()
    summary['date']=day
    print(f'Aggregation: {time.perf_counter()-started:.2f}s, {reader.reads} panel reads',flush=True)
    return summary


def notionals_on(day, weights, args, calendar):
    return args.notionals, [f'notional_{v:g}' for v in args.notionals]


def compatible_config(out, config):
    current = json.loads(json.dumps(config))
    manifest = out/'run.json'
    if not manifest.exists():
        return current
    old = json.loads(manifest.read_text())
    return incremental.compatible_manifest(old,current)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['audit','build','calibrate','aggregate'], default='audit')
    p.add_argument('--weights', type=Path, default=ROOT/'data/csi300/csi300_monthly_weights.csv')
    p.add_argument('--membership', type=Path, default=ROOT/'data/csi300/csi300_membership.csv')
    p.add_argument('--out', type=Path, default=ROOT/'artifacts/liquidity/csi300_zhao')
    p.add_argument('--root', type=Path, default=Path('/data/data/level2data'))
    p.add_argument('--start'); p.add_argument('--end')
    p.add_argument('--weight-timing', choices=['month-end','month-start'])
    p.add_argument('--weight-schedule', type=Path)
    p.add_argument('--weight-gap-policy', choices=['none','next-matching'], default='none')
    p.add_argument('--reuse-from', type=Path, help='Read-only source of compatible stock cost panels')
    p.add_argument('--max-weight-age-days', type=int, default=40)
    p.add_argument('--allow-estimated-weights', action='store_true')
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--shanghai-residual', action='store_true', help='Enable separately audited SH residual-add replay in diagnostic builds')
    p.add_argument('--codes', nargs='+', help='Diagnostic subset; retain full-universe weights and coverage')
    p.add_argument('--snapshots', type=Path, help='Certified snapshot root: YYYYMMDD/CODE.parquet')
    p.add_argument('--grid-ms', type=int, default=3000)
    p.add_argument('--cap-bps', type=float, default=200,help='Full-spread cap for separate proxy only; raw spreads remain uncapped')
    p.add_argument('--min-coverage', type=float, default=.95)
    p.add_argument('--min-valid', type=int, default=250)
    p.add_argument('--notionals', type=float, nargs='+', default=[1e6,5e6,1e7])
    p.add_argument('--min-exchange-coverage', type=float, default=.95)
    p.add_argument('--backend', choices=['serial','ray'], default='serial')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--normalization', type=Path)
    p.add_argument('--seasonal-bin-ms', type=int, default=60000)
    p.add_argument('--seasonal-min-days', type=int, default=20)
    p.add_argument('--seasonal-radius', type=int, default=1)
    p.add_argument('--scale-floor', type=float, default=.01)
    args = p.parse_args(); out = args.out; out.mkdir(parents=True,exist_ok=True)
    if args.stage == 'audit':
        weights = read_weights(args.weights); weight_audit(weights,out)
        print(json.dumps(json.loads((out/'weights_summary.json').read_text()),indent=2)); return
    if not args.start or not args.end:
        p.error('--start and --end are required (YYYYMMDD)')
    for d in [args.start,args.end]:
        if not re.fullmatch(r'\d{8}',d): p.error('Dates must be YYYYMMDD')
        pd.to_datetime(d,format='%Y%m%d')
    if args.start > args.end: p.error('Start follows end')
    if args.stage == 'calibrate':
        fit_normalization(out,args.start,args.end,args.seasonal_bin_ms,args.seasonal_min_days,
                          args.seasonal_radius,args.scale_floor); return
    args.weights_hash = sha(args.weights); args.book_hash = sha(Path(replay.__file__))
    if args.shanghai_residual:
        args.book_hash = hashlib.sha256((args.book_hash+sha(Path(shanghai.__file__))).encode()).hexdigest()
    args.book_hash = hashlib.sha256((args.book_hash+sha(Path(historical_books.__file__))).encode()).hexdigest()
    weights = read_weights(args.weights)
    if sha(args.weights) != args.weights_hash:
        raise ValueError('Weight file changed during reading; retry with a stable input')
    if not args.weight_timing and not args.weight_schedule:
        p.error('Specify --weight-timing or an effective/available-date --weight-schedule; month labels alone are ambiguous')
    if not args.diagnostic and (args.allow_estimated_weights or args.codes):
        p.error('Estimated weights and constituent subsets require --diagnostic')
    if args.shanghai_residual and not args.diagnostic:
        p.error('SH residual replay currently requires --diagnostic until independent depth validation')
    if not args.diagnostic and not args.weight_schedule:
        p.error('Primary construction requires explicit weight effective/available dates')
    if args.weight_gap_policy!='none' and (not args.diagnostic or not args.membership):
        p.error('Weight approximation requires --diagnostic and --membership')
    if args.reuse_from and args.reuse_from.resolve()==out.resolve():
        p.error('--reuse-from must differ from --out')
    if args.max_weight_age_days < 0: p.error('Weight age must be nonnegative')
    if not 0 < args.min_exchange_coverage <= 1 or not 0 < args.min_coverage <= 1 or not 1 <= args.min_valid <= 300 or args.grid_ms < 1 or args.cap_bps <= 0 or args.workers < 1:
        p.error('Invalid coverage, grid, cap or worker setting')
    if any((b-a) % args.grid_ms for a,b in PHASES.values()): p.error('Grid must divide each session duration')
    if any(x <= 0 or not np.isfinite(x) for x in args.notionals): p.error('Size scales must be finite and positive')
    scales = args.notionals
    if len(scales) < 3 or list(scales) != sorted(set(scales)):
        p.error('Provide at least three distinct size scales in increasing order')
    schedule = weight_schedule(weights,args.weight_timing,args.weight_schedule)
    calendar = sorted(x.stem for x in (args.root/'trades').glob('year=*/*.parquet'))
    days = [d for d in calendar if args.start <= d <= args.end]
    if not days: raise ValueError('No archive trading dates in requested interval')
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()
              if k not in ['stage','start','end','normalization','backend','workers','reuse_from','seasonal_min_days','seasonal_radius','seasonal_bin_ms','scale_floor']}
    if config.get('weight_gap_policy')=='none':config.pop('weight_gap_policy')
    config.update(incremental_cache_schema=incremental.SCHEMA,script_hash=sha(__file__),spec_hash=sha(ROOT/'docs/Liquidity/CSI300_HFLiquidityIndex_Construction.tex'),
                  membership_hash=sha(args.membership) if args.membership else None,
                  weight_schedule_hash=sha(args.weight_schedule) if args.weight_schedule else None,
                  measure_version=MEASURE_VERSION,notional_rule='executed_cash',rounding_rule='fractional_terminal_fill',
                  phases=PHASES,timezone='Asia/Shanghai',default_parameters='Design candidates, not empirically calibrated')
    with (out/'.initialization.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config = compatible_config(out, config)
        weight_audit(weights,out)
        schedule.to_csv(out/'weight_schedule.csv',index=False)
        if not (out/'run.json').exists():
            write_json(out/'run.json',config)
        run_hash=sha(out/'run.json')
        execution = out/'execution_versions'; execution.mkdir(exist_ok=True)
        write_json(execution/f'{sha(__file__)}.json', dict(
            implementation_hash=sha(__file__), construction_hash=run_hash,
            feed_quality_hash=sha(Path(feed_quality.__file__)),
            construction_script_hash=config['script_hash'],
            optimizations=['scalar_math','one_arrow_thread_per_worker','bounded_panel_cache']))
    stock_logic_version = incremental.source_hash(displayed_costs,missing_measure,sample_snapshots,endpoints)
    base_replay_hash = incremental.replay_cache_hash(sha(Path(replay.__file__)))
    legacy_base_hash = hashlib.sha256((base_replay_hash+sha(Path(shanghai.__file__))).encode()).hexdigest()
    legacy_cache = config.get('script_hash')==incremental.LEGACY_SCRIPT and config.get('book_hash')==legacy_base_hash==incremental.LEGACY_BOOK and stock_logic_version==incremental.LEGACY_STOCK_LOGIC
    aggregation_version = incremental.source_hash(aggregate_day,PanelReader)
    summaries=[]
    for day in tqdm(days,desc='Index days'):
        w = weights_on(weights,schedule,day,args)
        if args.codes and not set(args.codes) <= set(w.code):
            raise ValueError(f'Requested codes are not constituents on {day}')
        notionals,specs=notionals_on(day,w,args,calendar)
        jobs=[]
        for r in w.itertuples():
            if args.codes and r.code not in args.codes: continue
            jobs.append(dict(out=str(out),root=str(args.root),code=r.code,exchange=r.exchange,day=day,
                weight=r.weight,notionals=notionals,size_specs=specs,
                cap=args.cap_bps,grid_ms=args.grid_ms,
                diagnostic=args.diagnostic,snapshots=str(args.snapshots) if args.snapshots else None,
                shanghai_residual=args.shanghai_residual,
                run_hash=run_hash,book_hash=args.book_hash))
            job=jobs[-1]
            has_snapshot=bool(args.snapshots and (args.snapshots/day/f'{r.code}.parquet').exists())
            quality=feed_quality.policy(day,r.code,r.exchange,args.root,has_snapshot)
            job['feed_quality']=quality
            affected=incremental.affected(day,r.code,r.exchange,has_snapshot)
            versions=[base_replay_hash] if not has_snapshot else ['certified-snapshot-v1']
            if r.exchange=='SH' and not has_snapshot:versions.append(sha(Path(shanghai.__file__)))
            if affected:versions.append(sha(Path(historical_books.__file__)))
            if quality:versions.append(incremental.digest(dict(policy=quality,implementation=sha(Path(feed_quality.__file__)))))
            job.update(stock_logic_version=stock_logic_version,replay_version=incremental.digest(versions),
                       allow_legacy_cache=legacy_cache,historically_affected=affected,
                       legacy_base_hash=legacy_base_hash,reuse_from=str(args.reuse_from.resolve()) if args.reuse_from else None)
        if args.stage=='build':
            audits=run_jobs(jobs,args.backend,args.workers)
        else:
            audits=[json.loads((out/'constituents'/day/f'{j["code"]}.json').read_text()) for j in jobs]
            if any(a.get('run_hash')!=run_hash for a in audits): raise ValueError('Constituent construction hash mismatch')
        reused = sum(bool(a.get('_cache_reused')) for a in audits)
        for a in audits:a.pop('_cache_reused',None)
        # Ray stock jobs can finish out of order; align provenance by code.
        by_code={a['code']:a for a in audits};audits=[by_code[j['code']] for j in jobs]
        normalization_hash=sha(args.normalization) if args.normalization else None
        day_key=incremental.daily_key(jobs,audits,w,aggregation_version,normalization_hash)
        provenance=out/'daily_versions';provenance.mkdir(exist_ok=True)
        manifest=provenance/f'{day}.json'
        index_path=out/'index'/f'{day}.parquet'
        coverage_path=out/'coverage'/f'{day}.csv'
        previous=json.loads(manifest.read_text()) if manifest.exists() else None
        all_reused = reused==len(jobs)
        legacy_day = legacy_cache and aggregation_version==incremental.LEGACY_AGGREGATION and previous is None and all_reused and not args.normalization
        saved_membership=out/'audits'/f'{day}_membership.parquet'
        if legacy_day:
            legacy_day = saved_membership.exists() and incremental.same_membership(w,pd.read_parquet(saved_membership))
        matches = previous and previous.get('day_key')==day_key and previous.get('index_stats')==incremental.file_stats(index_path) and previous.get('coverage_stats')==incremental.file_stats(coverage_path)
        if args.stage=='build' and all_reused and index_path.exists() and coverage_path.exists() and (matches or legacy_day):
            print(f'{day}: reused {reused}/{len(jobs)} stock results; daily index unchanged',flush=True)
            if not previous:
                write_json(manifest,dict(day_key=day_key,index_stats=incremental.file_stats(index_path),coverage_stats=incremental.file_stats(coverage_path),legacy_adopted=True))
            summaries.append(pd.read_csv(coverage_path))
            continue
        print(f'{day}: reused {reused}/{len(jobs)} stock results; rebuilding daily index',flush=True)
        (out/'audits').mkdir(exist_ok=True);write_json(out/'audits'/f'{day}.json',audits)
        summaries.append(aggregate_day(out,day,w,audits,args,specs))
        w.to_parquet(out/'audits'/f'{day}_membership.parquet',index=False)
        coverage = out/'coverage'; coverage.mkdir(exist_ok=True)
        temp = coverage/f'{day}.tmp.csv'
        summaries[-1].to_csv(temp,index=False)
        temp.replace(coverage/f'{day}.csv')
        write_json(manifest,dict(day_key=day_key,index_stats=incremental.file_stats(index_path),coverage_stats=incremental.file_stats(coverage_path),implementation_hash=sha(__file__),stock_reused=reused,stock_rebuilt=len(jobs)-reused))
    print(pd.concat(summaries,ignore_index=True).to_string(index=False))


if __name__ == '__main__':
    main()
