#!/usr/bin/env python3
"""Cache RMB 1m market spreads on their original grid for Spectral DC fitting.

No imputation, seasonal adjustment or boundary concatenation is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def read_day(path):
    a = pd.read_parquet(path, columns=['size_spec', 't_ms', 'session', 'liq_available_vw_bps'])
    a = a[a.size_spec == 'notional_1e+06'].sort_values('t_ms')
    expected = np.r_[np.arange(34201000, 41400001, 1000), np.arange(46801000, 53820001, 1000)]
    if not np.array_equal(a.t_ms.to_numpy(), expected):
        raise ValueError(f'Unexpected second grid: {path}')
    y = a.liq_available_vw_bps.to_numpy()
    if np.isinf(y).any() or (y < 0).any():
        raise ValueError(f'Invalid spread: {path}')
    runs = []
    for session, lo, hi in [('am', 0, 7200), ('pm', 7200, 14220)]:
        good = np.isfinite(y[lo:hi])
        edges = np.diff(np.r_[False, good, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            v = y[lo + start:lo + end]
            runs.append(dict(date=path.stem, session=session, start=lo+int(start),
                             stop=lo+int(end), n=len(v), mean_bps=float(v.mean()),
                             sd_bps=float(v.std())))
    return y, runs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('artifacts/liquidity/csi300_monthly_imputed_1000ms'))
    p.add_argument('--out', type=Path, default=Path('artifacts/liquidity/csi300_dc_1m'))
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted((args.root/'index').glob('*.parquet'))
    if not files:
        raise ValueError('No daily index files')
    fingerprints = {str(f.resolve()): [f.stat().st_size, f.stat().st_mtime_ns] for f in files}
    cache = np.lib.format.open_memmap(args.out/'spread.partial.npy', mode='w+', dtype='float64', shape=(len(files), 14220))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, (y, runs) in enumerate(tqdm(executor.map(read_day, files), total=len(files), desc='Prepare DC input')):
            cache[i] = y
            rows.extend(dict(day_index=i, **r) for r in runs)
    cache.flush()
    del cache
    (args.out/'spread.partial.npy').replace(args.out/'spread.npy')
    runs = pd.DataFrame(rows)
    runs.to_csv(args.out/'contiguous_runs.csv', index=False)
    counts = {str(n): int(runs.loc[runs.n >= n, 'n'].sum()) for n in [1, 3, 30, 60, 300, 600, 1800, 3600]}
    manifest = dict(target='liq_available_vw_bps', size_spec='notional_1e+06',
                    dates=[f.stem for f in files], shape=[len(files), 14220],
                    valid_observations=int(runs.n.sum()), contiguous_runs=len(runs),
                    observations_in_runs_at_least_n_seconds=counts,
                    median_run_seconds=float(runs.n.median()), max_run_seconds=int(runs.n.max()),
                    source_fingerprints=fingerprints,
                    note='Original AM/PM grid; missing observations retained. No model fitted or transform applied.')
    (args.out/'input_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['dates','source_fingerprints']}, indent=2))


if __name__ == '__main__':
    main()
