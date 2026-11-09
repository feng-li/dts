#!/usr/bin/env python3
"""Assess saved CSI 300 draws against their common full Whittle posterior.

Implements the graph Stein discrepancy of Gorham and Mackey (2015), not
the later kernel Stein discrepancy. Requires the frozen numerical source
and inputs saved by benchmark_csi300_computation.py. No MCMC is rerun.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/csi300_stein_numba')

import numpy as np
from tqdm import tqdm

from _stein_discrepancy import graph_stein_discrepancy


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--date', default='20211013')
    parser.add_argument('--draws', type=int, default=1500,
                        help='First retained draws; default uses every saved draw')
    parser.add_argument('--workers', type=int, default=20,
                        help='Threads evaluating exact full-frequency scores')
    parser.add_argument('--lp-workers', type=int, default=4)
    parser.add_argument('--lp-time-limit', type=float, default=300.)
    parser.add_argument('--neighbours', type=int, default=12)
    args = parser.parse_args()
    if min(args.draws, args.workers, args.lp_workers, args.neighbours) < 1:
        parser.error('Draw counts, workers and neighbours must be positive')
    benchmark, out = args.benchmark.expanduser().resolve(), args.out.expanduser().resolve()
    source = benchmark/'source'
    inputs = benchmark/'inputs'/args.date
    case = benchmark/'cases'/args.date
    paths = {name: case/f'{name}_r2'/'posterior_draws.npy'
             for name in ['full', 'dc_serial', 'dc_parallel']}
    evidence = dict(q=inputs/'q.npy', periodogram=inputs/'periodogram.npy',
                    full_map=case/'full_r2/shards/group000.json', **paths)
    evidence.update(target=source/'scripts/liquidity/probe_csi300_fractional_changes.py',
                    coordinates=source/'scripts/liquidity/_spectral_coordinates.py',
                    diagnostic=Path(__file__).resolve(),
                    graph=Path(__file__).with_name('_stein_discrepancy.py').resolve())
    config = dict(benchmark=str(benchmark), date=args.date, draws=args.draws,
                  workers=args.workers, lp_workers=args.lp_workers,
                  lp_time_limit=args.lp_time_limit, neighbours=args.neighbours,
                  graph_stretch=2., target='full-data Whittle posterior including native prior',
                  coordinates=['atanh(theta)', 'log(lambda)', 'log(sigma2)', 'd'],
                  standardisation='common full-target MAP and inverse-Hessian Cholesky',
                  files={name: dict(path=str(path), sha256=digest(path))
                         for name, path in evidence.items()})
    out.mkdir(parents=True, exist_ok=True)
    config_path = out/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Output configuration changed; choose a new output directory')
    write_json(config_path, config)
    started = time.perf_counter()
    all_draws = {name: np.load(path) for name, path in paths.items()}
    if not np.array_equal(all_draws['dc_serial'], all_draws['dc_parallel']):
        raise ValueError('Serial and parallel consensus draws differ; evaluate them separately')
    if any(args.draws > len(draws) for draws in all_draws.values()):
        raise ValueError('Requested more draws than saved')

    # Bound JAX's initial pool, then restore affinity for the nogil score loop.
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, allowed[:1])
    sys.path.insert(0, str(source))
    from scripts.liquidity import probe_csi300_fractional_changes as probe
    from scripts.liquidity._spectral_coordinates import coordinate_map
    os.sched_setaffinity(0, allowed[:args.workers])
    if Path(probe.__file__).resolve() != evidence['target']:
        raise ValueError('Numerical source import did not resolve to the frozen benchmark')
    q = np.load(inputs/'q.npy', mmap_mode='r')
    pg = np.load(inputs/'periodogram.npy', mmap_mode='r')
    shard = json.loads((case/'full_r2/shards/group000.json').read_text())
    mode = coordinate_map(np.array(shard['native_map']), q, pg, 1, lambda_power=0.)
    center = np.array(mode['theta'])
    chol = np.linalg.cholesky(np.array(mode['covariance']))
    write_json(out/'common_coordinates.json', dict(mode=mode, cholesky=chol.tolist(),
                                                   positive_frequencies=len(q)))

    def score(native):
        _, negative_gradient = probe.target(np.ascontiguousarray(native), q, pg, 1)
        result = -negative_gradient*np.array([1-native[0]**2, 1., 1., 1.])
        result[0] -= 2*native[0]  # Exact atanh Jacobian contribution.
        return result@chol

    def log_density(point):
        u = center+chol@point
        native = np.array([np.tanh(u[0]), *u[1:]])
        log_jacobian = 2*(np.log(2.)-abs(u[0])-np.log1p(np.exp(-2*abs(u[0]))))
        return -probe.target(native, q, pg, 1)[0]+log_jacobian+np.log(np.diag(chol)).sum()

    check_point = np.array([.6, -.7, .8, -.9])
    u = center+chol@check_point
    analytic = score(np.array([np.tanh(u[0]), *u[1:]]))
    h = .001
    numeric = np.array([(log_density(check_point-2*h*unit)-8*log_density(check_point-h*unit)
                         +8*log_density(check_point+h*unit)-log_density(check_point+2*h*unit))/(12*h)
                        for unit in np.eye(4)])
    error = float(abs(analytic-numeric).max())
    write_json(out/'score_check.json', dict(analytic=analytic.tolist(), numeric=numeric.tolist(),
                                           maximum_absolute_error=error, step=h))
    if error > .001 or not np.isfinite(error):
        raise ValueError(f'Full-target transformed score failed finite differences: {error}')

    # Rejection repeats carry their empirical mass. Do not remove them and
    # incorrectly assign equal weights to the distinct support points.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name in ['full', 'dc_parallel']:
            native, counts = np.unique(all_draws[name][:args.draws], axis=0, return_counts=True)
            weights = counts/args.draws
            u = native.copy(); u[:, 0] = np.arctanh(native[:, 0])
            points = np.linalg.solve(chol, (u-center).T).T
            np.save(out/f'{name}_native.npy', native)
            np.save(out/f'{name}_points.npy', points)
            np.save(out/f'{name}_weights.npy', weights)
            cache = out/f'{name}_scores.npy'
            if cache.exists():
                scores = np.load(cache, mmap_mode='r+')
                if scores.shape != native.shape:
                    raise ValueError('Score cache shape mismatch')
            else:
                scores = np.lib.format.open_memmap(cache, mode='w+', dtype=float, shape=native.shape)
                scores[:] = np.nan; scores.flush()
            missing = np.flatnonzero(~np.isfinite(scores).all(axis=1))
            futures = {pool.submit(score, native[index]): index for index in missing}
            for number, future in enumerate(tqdm(as_completed(futures), total=len(missing),
                                                desc=f'{name}: full-target scores')):
                scores[futures[future]] = future.result()
                if number % 32 == 0:
                    scores.flush()
            scores.flush()
    score_seconds = time.perf_counter()-started

    results = {}
    # Spawn avoids forking numerical-runtime pools. Coordinate LPs are independent.
    with ProcessPoolExecutor(max_workers=args.lp_workers,
            mp_context=multiprocessing.get_context('spawn')) as executor:
        for name in ['full', 'dc_parallel']:
            result_path = out/f'{name}_discrepancy.json'
            if result_path.exists():
                results[name] = json.loads(result_path.read_text())
                continue
            begin = time.perf_counter()
            print(f'{name}: graph and four coordinate LPs', flush=True)
            result = graph_stein_discrepancy(np.load(out/f'{name}_points.npy'),
                np.load(out/f'{name}_scores.npy'), np.load(out/f'{name}_weights.npy'),
                neighbours=args.neighbours, executor=executor, time_limit=args.lp_time_limit)
            result.update(retained_draws=args.draws, diagnostic_seconds=time.perf_counter()-begin)
            write_json(result_path, result)
            results[name] = result
            print(f'{name}: Stein discrepancy = {result["discrepancy"]:.6g}', flush=True)
    results['dc_serial'] = results['dc_parallel']
    write_json(out/'summary.json', dict(results=results, source_hashes=config['files'],
        serial_parallel_draws_bitwise_equal=True, score_stage_seconds=score_seconds,
        diagnostic_total_seconds=time.perf_counter()-started,
        note='Lower discrepancy indicates better empirical approximation to the full Whittle target; '
             'this is not a percentage error or a test of the Whittle approximation to the time-domain posterior. '
             'Diagnostic computation is excluded from inference timing.'))


if __name__ == '__main__':
    main()
