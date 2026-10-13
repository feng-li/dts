"""Per-stock/day cache provenance and narrow migration of the audited v1 run."""
import hashlib
import inspect
import json
from pathlib import Path

SCHEMA = 'stock-day-v1'
LEGACY_STOCK_LOGIC = 'c04eb6c0a8e299cf2f80ec53e325dd7674e7d9f87614329ae4bedd6099478205'
LEGACY_AGGREGATION = 'eff16d2939599eb8c4128e99cec96895e27994688ba1e4ea04bbbf4860f4586d'
# Audited production builder preceding historical-clock/cage adapters.
LEGACY_SCRIPT = '346beb4287dfe2f99c9e41524cf0408449e895110f919f3e169fd2583ca1781b'
LEGACY_BOOK = 'a2ed7dfde9ca90b98e280af26feb57b34518a2e367004d586901e682e84a9b05'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()


def source_hash(*objects):
    return digest([inspect.getsource(x) for x in objects])


def file_stats(path):
    p=Path(path)
    return [p.stat().st_size,p.stat().st_mtime_ns] if p.exists() else None


def affected(day,code,exchange,has_snapshot=False):
    return not has_snapshot and ((exchange=='SH' and day.startswith('2020')) or
           (exchange=='SZ' and code.startswith('30') and '20200824'<=day<'20230410'))


def compatible_manifest(old,current):
    if old==current:return old
    supported=(old.get('incremental_cache_schema')==SCHEMA or
               (old.get('script_hash')==LEGACY_SCRIPT and old.get('book_hash')==LEGACY_BOOK))
    ignore={'script_hash','book_hash','incremental_cache_schema'}
    if not supported or {k:v for k,v in old.items() if k not in ignore}!={k:v for k,v in current.items() if k not in ignore}:
        raise ValueError('Construction configuration changed; use a new output directory')
    return old  # Immutable dataset identity; execution/day manifests track revisions.


def stock_key(job,fingerprint,snapshot):
    fields=['day','code','exchange','weight','notionals','size_specs','cap','grid_ms',
            'diagnostic','shanghai_residual','stock_logic_version','replay_version']
    return digest(dict(schema=SCHEMA,settings={k:job[k] for k in fields},
                       inputs=fingerprint,snapshot=bool(snapshot)))


def reusable(old,job,key,fingerprint,target):
    if not old.get('complete') or old.get('input_file_stats')!=fingerprint:return False
    if old.get('status')=='constructed' and not target.exists():return False
    if old.get('stock_cache_key') is not None:
        return old['stock_cache_key']==key and (
            old.get('status')!='constructed' or old.get('panel_file_stats')==file_stats(target))
    return bool(job.get('allow_legacy_cache') and not job.get('historically_affected')
                and old.get('run_hash')==job['run_hash'])


def daily_key(jobs,audits,weights,aggregation_version,normalization):
    return digest(dict(schema=SCHEMA,weights=weights.to_json(orient='split'),
        aggregation=aggregation_version,normalization=normalization,
        stocks=[dict(code=j['code'],key=a.get('stock_cache_key'),
                     audit=file_stats(Path(j['out'])/'constituents'/j['day']/f"{j['code']}.json"),
                     panel=file_stats(a.get('panel_path',Path(j['out'])/'constituents'/j['day']/f"{j['code']}.parquet")))
                for j,a in zip(jobs,audits)]))


def same_membership(current,saved):
    # Parquet intentionally drops the workbook's row index.
    return current.reset_index(drop=True).equals(saved.reset_index(drop=True))


def borrow(job,key,fingerprint):
    """Reuse weight-independent costs by reference; never mutate source panels."""
    import pandas as pd
    root=Path(job['reuse_from'])
    audit_path=root/'constituents'/job['day']/f"{job['code']}.json"
    manifest=root/'run.json'
    if not audit_path.exists() or not manifest.exists():return None
    conf=json.loads(manifest.read_text());old=json.loads(audit_path.read_text())
    if conf.get('measure_version')!='zhao-cash-full-spread-v1':return None
    fields={'root':'root','grid_ms':'grid_ms','cap_bps':'cap','notionals':'notionals',
            'diagnostic':'diagnostic','shanghai_residual':'shanghai_residual','snapshots':'snapshots'}
    if any(conf.get(c)!=job.get(j) for c,j in fields.items()):return None
    if conf.get('phases')!={'am':[34200000,41400000],'pm':[46800000,53820000]}:return None
    source_run=hashlib.sha256(manifest.read_bytes()).hexdigest()
    if old.get('run_hash')!=source_run:return None
    panel=Path(old.get('panel_path',root/'constituents'/job['day']/f"{job['code']}.parquet"))
    if old.get('stock_cache_key'):
        members=root/'audits'/f"{job['day']}_membership.parquet"
        if not members.exists():return None
        w=pd.read_parquet(members,columns=['code','weight']).set_index('code').weight
        if job['code'] not in w.index:return None
        source_job=dict(job,weight=float(w.loc[job['code']]))
        source_key=stock_key(source_job,fingerprint,bool(job.get('snapshots')))
        if not reusable(old,source_job,source_key,fingerprint,panel):return None
    else:
        trusted=(conf.get('script_hash')==LEGACY_SCRIPT and conf.get('book_hash')==LEGACY_BOOK
                 and job.get('legacy_base_hash')==LEGACY_BOOK
                 and job.get('stock_logic_version')==LEGACY_STOCK_LOGIC)
        source_job=dict(job,allow_legacy_cache=trusted,run_hash=source_run)
        if not reusable(old,source_job,key,fingerprint,panel):return None
    result=dict(old,run_hash=job['run_hash'],stock_cache_key=key,
                stock_logic_version=job['stock_logic_version'],replay_version=job['replay_version'],
                borrowed_from=str(audit_path.resolve()),borrowed_run_hash=source_run)
    if old.get('status')=='constructed':
        result.update(panel_path=str(panel.resolve()),panel_file_stats=file_stats(panel),
                      panel_weight_is_source_metadata=True)
    return result


def replay_cache_hash(actual_hash):
    """Narrow compatibility for null-reference rejection; clean books are identical.

    The full source hash remains in execution provenance. Only this reviewed exact
    revision reuses caches from before the audit stopped crashing on null refs.
    Such failed calls had no completed cache; they are retried normally.
    """
    return {"2381bc5e1fe7ee91892b6995b70f2ecb0945a0c40cb14c57572e9653e2ea2517":
            "fb8481468647e4b32b41025cc9c1c0973e1dad6ef5e0946455ff248dcbdebb95"}.get(actual_hash,actual_hash)
