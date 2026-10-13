"""Back-propagate CSI300 historical monthly weights from one official snapshot.

Method
------
w_i(t) ~ w_i(anchor) * P_i(t) / P_i(anchor), normalized within members(t)
(raw unadjusted closes; constant-share assumption). Members absent from the
anchor (index leavers) carry a flat median weight. Suspended stocks use the
last close on or before t.

Accuracy (validated against official HistoryWeights, 43 overlapping months):
per-stock MAE 6-16bp, median 3-11bp, cross-sectional corr 0.90-0.94, Top10 hit
7.5/10; portfolio monthly return diff mean ~0 (see data/csi300/
csi300_backprop_vs_official.csv). Known blind spots: share-structure events
(增发/回购/解禁) and 分级靠档 tier jumps.

Usage
-----
  python scripts/liquidity/csi300_download_prices_bs.py          # prices first
  python scripts/liquidity/csi300_backprop_weights.py [--anchor data/csi300/csi300_weights_20260901.csv]

Shared selection API
--------------------
  weight_schedule(), members_on(), weights_on() are also imported by the
  liquidity builder. They select historical snapshots and, when explicitly
  enabled, backfill gaps using the next matching snapshot with provenance.
  Importing this module does not regenerate weight files.

Outputs
-------
  data/csi300/csi300_monthly_weights.csv     (date, code, weight_pct, in_anchor)
  data/csi300/csi300_weights_validation.csv  monthly model-vs-index return diff (bp)
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data" / "csi300"


def weight_schedule(weights, timing, schedule_path=None):
    months = pd.DataFrame({'month': sorted(weights.month.unique())})
    if 'snapshot_date' in weights and timing=='month-start' and not schedule_path:
        raise ValueError('Dated month-end CSV snapshots cannot be used at month start')
    if schedule_path:
        schedule = pd.read_csv(schedule_path, dtype=str)
        required = {'month', 'effective_date', 'available_date'}
        if not required <= set(schedule):
            raise ValueError(f'Weight schedule needs {required}')
        if schedule.month.duplicated().any():
            raise ValueError('Duplicate month in weight schedule')
        months = months.merge(schedule, on='month', how='inner', validate='one_to_one')
        for col in ['effective_date', 'available_date']:
            months[col] = pd.to_datetime(months[col])
        months['usable_from'] = months[['effective_date', 'available_date']].max(axis=1)
    else:
        start = pd.to_datetime(months.month, format='%Y%m')
        months['usable_from'] = start + pd.offsets.MonthBegin(1) if timing == 'month-end' else start
    if 'snapshot_date' in weights:
        dates=weights.groupby('month').snapshot_date.first()
        earliest=pd.to_datetime(months.month.map(dates),format='%Y%m%d')+pd.Timedelta(days=1)
        if (months.usable_from < earliest).any():
            raise ValueError('Weight schedule precedes availability of dated snapshot')
    if months.usable_from.duplicated().any():
        raise ValueError('Ambiguous simultaneous weight snapshots')
    return months.sort_values('usable_from')


def members_on(path, day):
    a = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
    start = pd.to_datetime(a.valid_from, format='mixed')
    end = pd.to_datetime(a.valid_until, format='mixed', errors='coerce')
    a = a[(start <= pd.Timestamp(day)) & (end.isna() | (pd.Timestamp(day) < end))]
    if len(a) != 300 or a.code.duplicated().any():
        raise ValueError(f'Membership file does not define 300 distinct stocks on {day}')
    return set(a.code.str.zfill(6))


def weights_on(weights, schedule, day, args):
    available = schedule[schedule.usable_from <= pd.Timestamp(day)]
    impute = getattr(args,'weight_gap_policy','none')=='next-matching'
    if available.empty and not impute:
        raise ValueError(f'No prior available weight snapshot on {day}')
    official = members_on(args.membership, day) if args.membership else None
    rejected = []
    for row in available.sort_values('usable_from', ascending=False).itertuples(index=False):
        g = weights[weights.month == row.month].copy()
        reason = None
        if len(g) != 300 or g.code.duplicated().any() or not np.isfinite(g.weight_pct).all() or (g.weight_pct <= 0).any() or abs(g.weight_pct.sum() - 100) > .5:
            reason = 'invalid snapshot'
        elif (pd.Timestamp(day) - row.usable_from).days > args.max_weight_age_days:
            reason = 'stale weights'
        elif g.estimated.any() and not args.allow_estimated_weights:
            reason = 'back-calculated weights forbidden'
        elif official is not None and official != set(g.code):
            reason = 'membership mismatch'
        if reason is None:
            break
        rejected.append(f'{row.month}: {reason}')
    else:
        matched=False
        if impute:
            if not getattr(args,'diagnostic',False) or official is None:
                raise ValueError('Future weight approximation requires diagnostic mode and historical membership')
            for row in schedule[schedule.usable_from>pd.Timestamp(day)].sort_values('usable_from').itertuples(index=False):
                g=weights[weights.month==row.month].copy()
                if (len(g)==300 and not g.code.duplicated().any() and set(g.code)==official
                    and np.isfinite(g.weight_pct).all() and (g.weight_pct>0).all()
                    and abs(g.weight_pct.sum()-100)<=.5 and (args.allow_estimated_weights or not g.estimated.any())):
                    matched=True;break
        if not matched:
            raise ValueError(f'No admissible matching weight snapshot on {day}; ' + '; '.join(rejected))
    future = row.usable_from>pd.Timestamp(day)
    if impute:
        g['weight_imputed'] = future
        g['weight_imputation_method'] = 'next-matching-snapshot-backfill' if future else 'none'
        g['weight_lookahead_days'] = max(0,(row.usable_from-pd.Timestamp(day)).days)
    g['weight_age_days'] = (pd.Timestamp(day) - row.usable_from).days
    g['weight_selection_rule'] = 'latest-admissible-matching-v1'
    g['weight_usable_from'] = str(row.usable_from)
    g['membership_version'] = f'{args.weights_hash[:12]}:{row.month}:{row.usable_from.date()}'
    g['weight_effective_time'] = str(getattr(row, 'effective_date', row.usable_from))
    g['known_at'] = str(getattr(row, 'available_date', row.usable_from))
    g['weight_dates_assumed'] = not hasattr(row, 'available_date')
    return g


def ymd(s):
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return "99999999"
    return s.replace("-", "")


def load_series(prices, codes):
    series = {}
    for c in codes:
        p = prices / f"{c}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype={"date": str})
        df["date"] = df["date"].str.replace("-", "", regex=False)
        series[c] = dict(zip(df["date"], df["close"].astype(float)))
    return series


def main(anchor_path, prices):
    mem = pd.read_csv(D / "csi300_membership.csv", dtype={"code": str}).fillna("")
    w0 = pd.read_csv(anchor_path, dtype={"code": str})
    anchor_date = str(w0["date"].iloc[0])
    anchor = dict(zip(w0["code"], w0["weight_pct"]))
    median_w = float(w0["weight_pct"].median())

    cal = pd.read_csv(D / "raw" / "csi300_trading_days.csv", dtype=str)
    trade_days = sorted(cal["tradeDate"])
    assert set(cal["consNumber"].astype(float)) == {300.0}

    codes = sorted(set(mem["code"]))
    series = load_series(prices, codes)
    print(f"prices loaded for {len(series)}/{len(codes)} codes")

    def price_le(c, k):
        s = series.get(c)
        if not s:
            return None
        best = None
        for d, v in s.items():
            if d <= k:
                best = v
            else:
                break
        return best

    anchor_p = {c: price_le(c, ymd(anchor_date)) for c in anchor}
    miss = [c for c, v in anchor_p.items() if v is None]
    if miss:
        print(f"!! no price at anchor for {len(miss)} codes (treated as leavers)")

    def members_at(k):
        out = set()
        for r in mem.itertuples():
            vf, vu = ymd(r.valid_from), ymd(r.valid_until)
            if vf <= k and (vu == "99999999" or k < vu):
                out.add(r.code)
        return out

    # proper month-end trading days
    ends = []
    for y in range(2018, 2027):
        for m in range(1, 13):
            lim = (pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)).strftime("%Y%m%d")
            cand = [t for t in trade_days if f"{y}{m:02d}01" <= t <= lim]
            if cand:
                ends.append(cand[-1])
    ends = sorted(set(ends))

    rows = []
    for k in ends:
        num = {}
        for c in members_at(k):
            p_t = price_le(c, k)
            if p_t is None:
                continue
            if c in anchor and anchor_p.get(c):
                num[c] = anchor[c] * p_t / anchor_p[c]
            else:  # leaver: flat median weight, no drift
                num[c] = median_w
        tot = sum(num.values())
        assert tot > 0, k
        for c, v in num.items():
            rows.append({"date": k, "code": c, "weight_pct": round(100 * v / tot, 6),
                         "in_anchor": int(c in anchor)})
        assert abs(sum(r["weight_pct"] for r in rows if r["date"] == k) - 100) < 1e-3

    out = pd.DataFrame(rows)
    out.to_csv(D / "csi300_monthly_weights.csv", index=False, encoding="utf-8-sig")
    print(f"weights: {len(out)} rows, {len(ends)} month-ends -> {D/'csi300_monthly_weights.csv'}")

    # ---- validation: model basket return vs official index return
    import baostock as bs
    assert bs.login().error_code == "0"
    rs = bs.query_history_k_data_plus("sh.000300", "date,close", start_date="2017-12-01",
                                      end_date="2026-09-29", frequency="d")
    idx = {}
    while rs.next():
        d_, c_ = rs.get_row_data()
        idx[d_.replace("-", "")] = float(c_)
    bs.logout()

    diffs = []
    for a, b in zip(ends, ends[1:]):
        if a not in idx or b not in idx:
            continue
        r_idx = idx[b] / idx[a] - 1
        wa = {r["code"]: r["weight_pct"] / 100 for r in rows if r["date"] == a}
        r_model, ok = 0.0, True
        for c, w in wa.items():
            pa, pb = price_le(c, a), price_le(c, b)
            if pa is None or pb is None or pa == 0:
                ok = False
                break
            r_model += w * (pb / pa - 1)
        if ok:
            diffs.append({"from": a, "to": b, "diff_bp": (r_model - r_idx) * 1e4,
                          "idx_ret_pct": r_idx * 100})
    s = pd.DataFrame(diffs)
    s.to_csv(D / "csi300_weights_validation.csv", index=False, encoding="utf-8-sig")
    print(f"validation months: {len(s)} | diff mean={s['diff_bp'].mean():.1f}bp "
          f"median={s['diff_bp'].median():.1f}bp p90abs={s['diff_bp'].abs().quantile(0.9):.1f}bp")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor", type=Path, default=D / "csi300_weights_20260901.csv")
    ap.add_argument("--prices", type=Path, default=ROOT / "data" / "csi300" / "raw" / "prices")
    a = ap.parse_args()
    main(a.anchor, a.prices)
