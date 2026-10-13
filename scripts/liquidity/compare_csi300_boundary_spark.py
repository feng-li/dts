#!/usr/bin/env python3
"""Re-fit saved CSI 300 training histories with repository Spark Spectral DC.

Reuse the application response/cutoff exactly; run dts.dfft_blocks on the full
response, then dts.mcmc.sampler independently on systematic frequency groups.
One chain per shard is the production default; independent diagnostic chains
are requested explicitly. All output is checkpointed separately from the
existing Ray experiment.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('SPARK_LOCAL_HOSTNAME', 'localhost')
os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dts-csi300-spark-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/dts-csi300-spark-numba')
import numpy as np
import pandas as pd
from tqdm import tqdm
from dts.dfft_blocks import spark_fft_contiguous_blocks

RUN_ARGUMENTS = ['reference', 'out', 'series', 'groups', 'chains', 'iterations',
                 'burn', 'seed', 'spark_slots', 'fft_partitions', 'dates', 'sampler_backend']
AFFINITY_PREVIOUS_SHA256 = '19d64364e4d597b50c1818e721c9ec1f8bbf84aa907699ee13b9c652ea269f35'
SINGLE_CHAIN_PREVIOUS_SHA256 = 'eb3c9f0391d6246f505f71cbb468e870c2940db15540efa64da2c88b9e562978'
CHECKPOINT_PREVIOUS_SHA256 = '4a3592fd9baa850bb8e8a0a2fde3ddf635cc612746c63993fbb667d8749f61cb'


def compatible_worker_affinity_update(previous, current):
    """Allow only this recorded scheduling fix with unchanged fit settings."""
    left, right = dict(previous), dict(current)
    old_sources, new_sources = left.pop('source_sha256'), right.pop('source_sha256')
    left.pop('source_history', None)
    right.pop('source_history', None)
    key = 'scripts/liquidity/compare_csi300_boundary_spark.py'
    return (left == right and old_sources.get(key) == AFFINITY_PREVIOUS_SHA256
            and {k: v for k, v in old_sources.items() if k != key}
            == {k: v for k, v in new_sources.items() if k != key})


def compatible_single_chain_update(previous, current):
    """Retain existing diagnostic runs only when their settings are unchanged."""
    left, right = dict(previous), dict(current)
    old_sources, new_sources = left.pop('source_sha256'), right.pop('source_sha256')
    left.pop('source_history', None)
    right.pop('source_history', None)
    key = 'scripts/liquidity/compare_csi300_boundary_spark.py'
    return (left == right and previous.get('chains', 0) >= 2
            and old_sources.get(key) == SINGLE_CHAIN_PREVIOUS_SHA256
            and {k: v for k, v in old_sources.items() if k != key}
            == {k: v for k, v in new_sources.items() if k != key})


def compatible_checkpoint_update(previous, current):
    """Allow this checkpoint repair with identical numerical code and settings."""
    left, right = dict(previous), dict(current)
    old_sources, new_sources = left.pop('source_sha256'), right.pop('source_sha256')
    left.pop('source_history', None)
    right.pop('source_history', None)
    key = 'scripts/liquidity/compare_csi300_boundary_spark.py'
    return (left == right and old_sources.get(key) == CHECKPOINT_PREVIOUS_SHA256
            and {k: v for k, v in old_sources.items() if k != key}
            == {k: v for k, v in new_sources.items() if k != key})


def sampling_quality(draws):
    """Check ESS for production; compare independent chains when available.

    The existing estimator splits each chain to estimate bulk/tail ESS. Its
    one-chain split Rhat is not reported as an independent-chain diagnostic.
    """
    from scripts.liquidity.compare_csi300_boundary_random_days import chain_quality
    draws = np.asarray(draws)
    if draws.ndim != 3 or draws.shape[0] < 1 or draws.shape[1] < 4:
        raise ValueError('At least one chain with four retained draws is required')
    result = chain_quality(draws)
    result['independent_chains_checked'] = bool(draws.shape[0] >= 2)
    if draws.shape[0] == 1:
        result['rhat_max'] = None
        result['passed'] = bool(result['ess_bulk_min'] >= 400 and result['ess_tail_min'] >= 400)
    return result


def release_worker_threads(allowed):
    """Let the scheduler place all initialized worker threads on usable CPUs."""
    if hasattr(os, 'sched_setaffinity'):
        for thread in Path('/proc/self/task').iterdir():
            try:
                os.sched_setaffinity(int(thread.name), set(allowed))
            except ProcessLookupError:
                pass


def write_json(path, value):
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def save_npz(path, **values):
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def save_npy(path, value):
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('wb') as handle:
        np.save(handle, value)
    temporary.replace(path)


def fft_partitions_for(n, maximum):
    """Preserve n by choosing a power-of-two decomposition that divides it."""
    if n < 2 or n % 2 or maximum < 1 or maximum & (maximum-1):
        raise ValueError('Even n and a positive power-of-two maximum are required')
    p = maximum
    while n % p:
        p //= 2
    return p


def select_dates(available, references, count, seed):
    """Include full-MCMC checks, then sample remaining dates without outcomes."""
    if not 1 <= count <= len(available):
        raise ValueError('Series count exceeds the saved evaluation')
    fixed = sorted(set(references))
    if len(fixed) > count:
        fixed = fixed[:count]
    pool = sorted(set(available)-set(fixed))
    extra = np.random.default_rng(seed).choice(pool, count-len(fixed), replace=False)
    return sorted(fixed+extra.tolist())


def prepare_response(source, saved_folder, folder):
    from scripts.liquidity.compare_csi300_boundary_random_days import standardized_training_changes
    saved = json.loads((saved_folder/'fit.json').read_text())
    path = folder/'response.npy'
    if path.exists():
        # Older versions wrote directly to the destination. Discard only a
        # truncated legacy file; new writes become visible after completion.
        try:
            response = np.load(path, mmap_mode='r')
        except (OSError, ValueError, EOFError):
            path.replace(path.with_name(path.name+'.interrupted'))
        else:
            if response.shape != (saved['n'],) or not np.isfinite(response).all():
                raise ValueError('Invalid saved response')
            return saved
    if not path.exists():
        raw = np.load(source/'spread.npy', mmap_mode='r')
        response = standardized_training_changes(raw[:saved['train_stop_day']],
                    np.load(saved_folder/'clock_mean.npy'), np.load(saved_folder/'clock_scale.npy'))
        if saved['earliest_change_removed']:
            response = response[saved['earliest_change_removed']:]
        response /= saved['scale']
        if len(response) != saved['n'] or len(response) < 10_000_000:
            raise ValueError('Reconstructed training response/count differs from saved application')
        save_npy(path, response)
    response = np.load(path, mmap_mode='r')
    if response.shape != (saved['n'],) or not np.isfinite(response).all():
        raise ValueError('Invalid saved response')
    return saved


def load_input_block(block, path, n, p):
    response = np.load(path, mmap_mode='r')
    width = n//p
    return block, np.ascontiguousarray(response[block*width:(block+1)*width])


def frequency_chunks(record, n, p, groups):
    """Pass array chunks directly from block-cyclic FFT to systematic groups."""
    _, (offset, transformed) = record
    block_size = n//p
    for radix in range(p):
        start = radix*block_size+offset
        lo, hi = max(1-start, 0), min(n//2-start, transformed.shape[1])
        if hi <= lo:
            continue
        indices = np.arange(start+lo, start+hi, dtype=np.int64)
        values = np.abs(transformed[radix, lo:hi])**2/(2*np.pi*n)
        for group in range(groups):
            shift = (group-(int(indices[0])-1)) % groups
            if shift < len(indices):
                yield group, (indices[shift::groups], values[shift::groups])


def cache_frequency_group(item, folder, saved_folder, n, groups):
    group, chunks = item
    indices, values = zip(*list(chunks))
    indices, pg = np.concatenate(indices), np.concatenate(values)
    order = np.argsort(indices)
    indices, pg = indices[order], pg[order]
    path = Path(folder)/'frequencies'/f'shard{group:03d}.npz'
    omega = 2*np.pi*indices/n
    error = validate_frequency_group(group, indices, pg, omega, 4*np.sin(omega/2)**2,
                                     saved_folder, n, groups)
    save_npz(path, indices=indices, periodogram=pg, omega=omega,
             q=4*np.sin(omega/2)**2)
    write_json(path.with_suffix('.json'), error)
    return error


def validate_frequency_group(group, indices, pg, omega, q, saved_folder, n, groups):
    """Validate a new or interrupted FFT checkpoint against the original FFT."""
    np.testing.assert_array_equal(indices, np.arange(group+1, n//2, groups, dtype=np.int64))
    original = np.load(Path(saved_folder)/'periodogram.npy', mmap_mode='r')[indices-1]
    np.testing.assert_allclose(pg, original, rtol=1e-8, atol=1e-12)
    np.testing.assert_array_equal(omega, 2*np.pi*indices/n)
    np.testing.assert_array_equal(q, 4*np.sin(omega/2)**2)
    error = dict(group=int(group), ordinates=len(indices),
                 max_absolute_error=float(np.max(np.abs(pg-original))),
                 max_relative_error=float(np.max(np.abs(pg-original)/np.maximum(np.abs(original), 1e-12))),
                 checked_against_saved_full_fft=True)
    return error


def recover_frequency_checks(folder, saved_folder, n, groups):
    """Reuse complete FFT groups and recover sidecars without repeating FFTs."""
    paths = [folder/'frequencies'/f'shard{g:03d}.npz' for g in range(groups)]
    if not all(path.exists() for path in paths):
        return None
    checks = []
    for group, path in enumerate(paths):
        sidecar = path.with_suffix('.json')
        if sidecar.exists():
            checks.append(json.loads(sidecar.read_text()))
        else:
            with np.load(path) as saved:
                check = validate_frequency_group(group, saved['indices'], saved['periodogram'],
                    saved['omega'], saved['q'], saved_folder, n, groups)
            write_json(sidecar, check)
            checks.append(check)
    return checks


def saved_shard_quality(path):
    """A saved sample file contains the metadata needed to finish its commit."""
    if not path.exists():
        return None
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    with np.load(path) as saved:
        if 'quality_json' not in saved:
            # Legacy files without a sidecar cannot recover acceptance/timing.
            return None
        quality = json.loads(str(saved['quality_json'].item()))
    write_json(sidecar, quality)
    return quality


def record_execution(out, dates, slots, elapsed, status, completed_before):
    """Preserve prior timings when a resume only rebuilds the final report."""
    path = out/'execution.json'
    history = []
    if path.exists():
        previous = json.loads(path.read_text())
        history = previous.get('attempts', [{**previous, 'legacy': True}])
    completed_after = sum((out/'days'/date/'completed.json').exists() for date in dates)
    history.append(dict(seconds=elapsed, status=status, completed_before=completed_before,
                        completed_after=completed_after, finished_at=time.time()))
    write_json(path, dict(series=len(dates), spark_slots=slots, status=status,
                          seconds=sum(x['seconds'] for x in history),
                          last_attempt_seconds=elapsed, attempts=history))


def polish_map(fit, q, pg, groups):
    """Resolve early relative-objective stopping with damped Newton steps.

    The target, parameter bounds and positive-curvature check are identical
    to the original fit. This only refines an unresolved optimizer result.
    """
    if fit['covariance'] is not None and fit['newton_distance'] <= .01:
        return fit
    from scripts.liquidity.probe_csi300_fractional_changes import target
    point = np.asarray(fit['theta'])
    bounds = np.array([[-.99999999, .99999999], [-30., 10.], [-30., 20.], [-2., 2.]])
    result = dict(fit, initial_newton_distance=fit['newton_distance'], newton_polish_steps=0)
    for iteration in range(21):
        value, gradient = target(point, q, pg, groups)
        hessian = np.empty((4, 4))
        for j in range(4):
            delta = np.zeros(4)
            delta[j] = 1e-5
            hessian[:, j] = (target(point+delta, q, pg, groups)[1]
                            - target(point-delta, q, pg, groups)[1])/(2e-5)
        hessian = (hessian+hessian.T)/2
        eigen = np.linalg.eigvalsh(hessian)
        if not np.isfinite(eigen).all() or eigen.min() <= 0:
            return result
        step = np.linalg.solve(hessian, gradient)
        distance = float(np.sqrt(max(0., gradient@step)))
        result.update(theta=point.tolist(), newton_distance=distance,
                      hessian_eigenvalues=eigen.tolist(), covariance=np.linalg.inv(hessian).tolist(),
                      newton_polish_steps=iteration)
        if distance <= .01 or iteration == 20:
            return result
        fraction = 1.
        for _ in range(25):
            candidate = point-fraction*step
            if ((candidate >= bounds[:, 0]) & (candidate <= bounds[:, 1])).all():
                candidate_value = target(candidate, q, pg, groups)[0]
                if np.isfinite(candidate_value) and candidate_value <= value+1e-10:
                    point = candidate
                    break
            fraction *= .5
        else:
            return result
    return result


def run_shard(group, folder, config):
    folder = Path(folder)
    path = folder/'shards'/f'G{config["groups"]}_shard{group:03d}.npz'
    sidecar = path.with_suffix('.json')
    cached = saved_shard_quality(path)
    if cached is not None:
        return cached
    # Bound pool creation and compilation on one CPU, then release ALL threads.
    # Re-pinning only the main thread leaves reused JAX pools on old CPUs.
    allowed = config['worker_cpu_ids']
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {allowed[group % len(allowed)]})
    sys.path.insert(0, str(ROOT))
    from dts.mcmc import sampler, whittle_log_posterior_jit
    from scripts.liquidity.probe_csi300_fractional_changes import fit_spectrum, target
    started = time.monotonic()
    try:
        with np.load(folder/'frequencies'/f'shard{group:03d}.npz') as frequency:
            q, pg, omega = frequency['q'], frequency['periodogram'], frequency['omega']
        fit = polish_map(fit_spectrum(q, pg, groups=config['groups']), q, pg, config['groups'])
        if fit['covariance'] is None or fit['newton_distance'] > .01:
            raise ValueError(f'Unresolved shard MAP: {fit}')
        covariance = np.asarray(fit['covariance'])*(2.38**2/4)
        chol = np.linalg.cholesky(covariance)
        center = np.asarray(fit['theta'])
        warm = float(whittle_log_posterior_jit(center, 0, 1, pg, True, omega,
                     config['groups'], 0, 1.0, 3, False, None, None, 0.0, 1.0))
        target_error = abs(warm+target(center, q, pg, config['groups'])[0])
        if not np.isfinite(warm) or target_error > 1e-7:
            raise ValueError(f'Native posterior target mismatch: {target_error}')
        release_worker_threads(allowed)
        draws, acceptances = [], []
        for chain in range(config['chains']):
            seed = config['day_seed']+group*1000+chain
            initial = center+chol@np.random.default_rng(seed).normal(size=4)
            if not np.isfinite(target(initial, q, pg, config['groups'])[0]):
                initial = center.copy()
            samples, logp, acceptance = sampler(ar_order=0, ma_order=1, data=None,
                I_pg=pg, TFI_term=True, omega_shard=omega,
                n_samples=config['iterations'], Burn_in=config['burn'],
                paramsStar=initial, proposal_width=covariance, n_groups=config['groups'],
                mean_term=False, exact=False, random_state=seed, progress=False,
                backend=config['sampler_backend'])
            if not np.isfinite(samples).all() or not np.isfinite(logp).all():
                raise ValueError('Nonfinite posterior samples')
            draws.append(samples)
            acceptances.append(float(acceptance.mean()))
        draws = np.asarray(draws)
        quality = sampling_quality(draws)
        quality.update(group=int(group), ordinates=len(pg), acceptance=acceptances,
                       map=fit['theta'], newton_distance=fit['newton_distance'],
                       newton_polish_steps=fit.get('newton_polish_steps', 0),
                       native_target_error=target_error,
                       iterations=config['iterations'], burn=config['burn'],
                       sampler_backend=config['sampler_backend'],
                       status='sampled', seconds=time.monotonic()-started)
        save_npz(path, draws=draws, map=np.asarray(fit['theta']), proposal_covariance=covariance,
                 quality_json=json.dumps(quality))
        write_json(sidecar, quality)
        return quality
    except Exception as exc:
        failure = dict(group=int(group), status='failed', passed=False,
                       error=f'{type(exc).__name__}: {exc}', seconds=time.monotonic()-started)
        write_json(folder/'shards'/f'G{config["groups"]}_shard{group:03d}_failure.json', failure)
        return failure


def monitored_action(action, folder, pattern, total, description):
    """Run Spark while keeping actual checkpoint progress visible."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(action)
        with tqdm(total=total, desc=description, leave=False) as progress:
            while not future.done():
                count = min(total, len(list(folder.glob(pattern))))
                progress.update(max(0, count-progress.n))
                time.sleep(1)
            result = future.result()
            progress.update(total-progress.n)
    return result


def finish_day(reference, out, date, records, config, fft_checks, fft_seconds, sampling_seconds):
    from dts.aggregation import consensus
    from scripts.liquidity.compare_csi300_boundary_random_days import (
        posterior_comparison, forecasts_on_day)
    folder = out/'days'/date
    quality = pd.DataFrame(records).sort_values('group')
    quality.to_csv(folder/'shard_diagnostics.csv', index=False)
    sampled = quality[quality.status.eq('sampled')]
    summary = dict(date=date, groups=config['groups'], chains=config['chains'],
                   sampler_backend=config['sampler_backend'],
                   iterations=config['iterations'], burn=config['burn'],
                   n=config['n'], fft_partitions=config['actual_fft_partitions'],
                   fft_seconds=fft_seconds, sampling_seconds=sampling_seconds,
                   sampled_shards=len(sampled), failed_shards=int(quality.status.eq('failed').sum()),
                   passed_shards=int(quality.passed.sum()),
                   fft_max_absolute_error=max(x['max_absolute_error'] for x in fft_checks),
                   independent_chains_checked=bool(config['chains'] >= 2),
                   sampling_passed=False, posterior_accuracy_passed=None, forecast_scored=False)
    if len(sampled) == config['groups']:
        shards = [np.load(folder/'shards'/f'G{config["groups"]}_shard{g:03d}.npz')['draws']
                  for g in range(config['groups'])]
        shape = shards[0].shape
        combined = consensus([d.reshape(-1, 4) for d in shards], ridge=1e-12).reshape(shape)
        save_npy(folder/f'G{config["groups"]}_draws.npy', combined)
        combined_quality = sampling_quality(combined)
        combined_quality['shards_passed'] = bool(quality.passed.all())
        write_json(folder/'consensus_quality.json', combined_quality)
        summary['sampling_passed'] = bool(combined_quality['passed'] and quality.passed.all())
        summary['consensus_rhat_max'] = combined_quality['rhat_max']
        summary['consensus_ess_bulk_min'] = combined_quality['ess_bulk_min']
        summary['consensus_ess_tail_min'] = combined_quality['ess_tail_min']
        names = ['theta', 'log_lambda', 'log_sigma2', 'd']
        flat = combined.reshape(-1, 4)
        pd.DataFrame(dict(parameter=names, mean=flat.mean(axis=0), sd=flat.std(axis=0, ddof=1),
                          lower=np.quantile(flat, .025, axis=0), median=np.median(flat, axis=0),
                          upper=np.quantile(flat, .975, axis=0))).to_csv(folder/'posterior_summary.csv', index=False)
        previous = np.load(reference/'days'/date/'G2_draws.npy')
        write_json(folder/'dc_vs_previous_G2.json', posterior_comparison(previous, combined))
        full_path = reference/'days'/date/'G1_draws.npy'
        if full_path.exists():
            comparison = posterior_comparison(np.load(full_path), combined)
            write_json(folder/'dc_vs_full.json', comparison)
            summary['posterior_accuracy_passed'] = comparison['passed']
        if summary['sampling_passed']:
            saved_folder = reference/'days'/date
            fit = json.loads((saved_folder/'fit.json').read_text())
            for name in ['profile.npy', 'clock_mean.npy', 'clock_scale.npy']:
                target = folder/name
                if not target.exists():
                    target.symlink_to(saved_folder/name)
            tasks = pd.read_csv(reference/'forecast_tasks.csv', dtype={'date': str})
            tasks = tasks[tasks.date.eq(date)]
            forecast_config = dict(posterior_parameters=64, paths_per_parameter=8)
            forecasts_on_day(Path(config['input']), folder, tasks, fit, combined,
                             forecast_config, config['day_seed']+7)
            summary['forecast_scored'] = True
    write_json(folder/'completed.json', summary)
    return summary


def summarize(out, reference, dates, execution_seconds):
    records = [json.loads((out/'days'/date/'completed.json').read_text()) for date in dates]
    diagnostics = pd.DataFrame(records)
    diagnostics.to_csv(out/'daily_diagnostics.csv', index=False)
    forecast_dates = diagnostics.loc[diagnostics.forecast_scored, 'date'].tolist()
    groups = records[0]['groups']
    chains = records[0]['chains']
    sampler_backend = records[0].get('sampler_backend', 'vanilla')
    tables = []
    for date in forecast_dates:
        old = pd.read_parquet(reference/'days'/date/'forecasts.parquet')
        new = pd.read_parquet(out/'days'/date/'forecasts.parquet')
        old = old[old.model.eq('SpectralDC_ARTFIMA')].copy()
        old['model'] = 'SpectralDC_ARTFIMA_G2'
        new.loc[new.model.eq('SpectralDC_ARTFIMA'), 'model'] = f'Spark_SpectralDC_ARTFIMA_G{groups}'
        tables.extend([old, new])
    scores = None
    if tables:
        forecasts = pd.concat(tables, ignore_index=True)
        forecasts.to_parquet(out/'forecasts.parquet', index=False)
        scored = forecasts[np.isfinite(forecasts.actual)]
        scores = scored.groupby(['model', 'horizon']).agg(n=('se', 'size'), days=('date', 'nunique'),
            mse=('se', 'mean'), mae=('ae', 'mean'), crps=('crps', 'mean'), coverage95=('cover95', 'mean')).reset_index()
        scores['rmse'] = np.sqrt(scores.pop('mse'))
        scores.to_csv(out/'scores.csv', index=False)
    report = [f'CSI 300 Spark Spectral DC: {groups} frequency shards per training history',
              f'{len(dates)} expanding daily histories of the same aggregate liquidity series.',
              'Repository block-record Spark FFT and native dts.mcmc.sampler; fractional prior 1/G; fixed MAP-Hessian RWMH.',
              f'Sampler backend: {sampler_backend}.',
              'Exact original training lengths retained; P chosen as the largest power-of-two divisor <= requested P.',
              'Every positive-frequency ordinate is checked against the saved full-series FFT.',
              ('Sampling gate: all shards and consensus Rhat <=1.01; bulk/tail ESS >=400.'
               if chains >= 2 else
               'Production sampling gate: all shards and consensus bulk/tail ESS >=400. One chain per shard; independent-chain convergence is not checked; Rhat is blank.'),
              'Accuracy gate: full-reference mean shift <0.1 full SD and SD ratio within 0.9-1.1.',
              'Forecasts are produced only for dates passing the sampling gate; posterior accuracy is reported separately.',
              f'Execution wall seconds: {execution_seconds:.1f}', '', diagnostics.to_string(index=False)]
    if scores is not None:
        report.extend(['', 'Matched-date forecast scores:', scores.to_string(index=False)])
    (out/'REPORT.txt').write_text('\n'.join(report)+'\n')
    print(diagnostics.to_string(index=False), flush=True)
    if scores is not None:
        print(scores.to_string(index=False), flush=True)


def parse_arguments(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args(argv)
    defaults = {}
    if known.config:
        saved = json.loads(known.config.read_text())
        defaults = {k: saved[k] for k in RUN_ARGUMENTS if k in saved}
        if 'sampler_backend' not in saved and 'source_sha256' in saved:
            defaults['sampler_backend'] = 'vanilla'
    base = Path.home()/'running/dts/artifacts/liquidity'
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument('--reference', type=Path, default=base/'csi300_boundary_full1000_v1')
    parser.add_argument('--out', type=Path, default=base/'csi300_boundary_spark_g200_test10_v1')
    parser.add_argument('--series', type=int, default=10)
    parser.add_argument('--groups', type=int, default=200)
    parser.add_argument('--chains', type=int, default=1,
                        help='Chains per shard: 1 for production; use 4 for independent-chain diagnostics')
    parser.add_argument('--iterations', type=int, default=15000)
    parser.add_argument('--burn', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--spark-slots', type=int, default=100)
    parser.add_argument('--fft-partitions', type=int, default=256)
    parser.add_argument('--sampler-backend', choices=['compiled', 'vanilla'], default='compiled')
    parser.add_argument('--dates', nargs='+')
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.groups < 2 or args.chains < 1 or args.iterations-args.burn < 4 or args.burn < 0:
        parser.error('Requires distributed groups, >=1 chain and >=4 retained iterations')
    if args.spark_slots < 1 or args.series < 1:
        parser.error('Positive Spark slots and series count required')
    fft_partitions_for(2, args.fft_partitions)
    return args


def main():
    args = parse_arguments()
    args.reference, args.out = args.reference.expanduser().absolute(), args.out.expanduser().absolute()
    original = json.loads((args.reference/'protocol.json').read_text())
    input_config = json.loads((args.reference/'run_config.json').read_text())
    source = Path(input_config['input'])
    dates = args.dates or select_dates(original['dates'], original['references'], args.series, args.seed)
    if len(dates) != args.series or len(set(dates)) != len(dates) or not set(dates) <= set(original['dates']):
        raise ValueError('Dates must identify the requested number of distinct saved training histories')
    dates = sorted(dates)
    cpu_ids = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count()))
    config = {k: str(getattr(args, k)) if isinstance(getattr(args, k), Path) else getattr(args, k)
              for k in RUN_ARGUMENTS}
    config.update(dates=dates, input=str(source), worker_cpu_ids=cpu_ids,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), ROOT/'dts/dfft_blocks.py', ROOT/'dts/mcmc.py', ROOT/'dts/aggregation.py',
             ROOT/'scripts/liquidity/probe_csi300_fractional_changes.py',
             ROOT/'scripts/liquidity/compare_csi300_boundary_random_days.py']},
        input_manifest_sha256=hashlib.sha256((source/'input_manifest.json').read_bytes()).hexdigest(),
        reference_protocol_sha256=hashlib.sha256((args.reference/'protocol.json').read_bytes()).hexdigest(),
        model='ARTFIMA(0,d,lambda,1) on identical training-clock-standardized changes',
        inference='Spark block FFT; systematic frequency groups; native DTS sampler; prior 1/G; fixed MAP-Hessian RWMH; joint consensus',
        fft_policy='Largest power-of-two divisor <= fft_partitions; no observations discarded')
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out/'run_config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        config['source_history'] = previous.get('source_history', [])
        if previous != config:
            if compatible_worker_affinity_update(previous, config):
                reason = 'Release all initialized JAX threads from stale CPU affinity; same target, sampler and seeds'
                archive_name = 'worker_affinity'
            elif compatible_single_chain_update(previous, config):
                reason = 'One-chain production default and dynamic reporting; existing independent-chain settings, target, sampler and seeds unchanged'
                archive_name = 'single_chain_default'
            elif compatible_checkpoint_update(previous, config):
                reason = 'Atomic response/consensus writes and recovery of checkpoint metadata; same numerical target, sampler, settings and seeds'
                archive_name = 'atomic_checkpoints'
            else:
                raise ValueError('Use a new output directory after incompatible configuration/source changes')
            archive = args.out/'attempts'/archive_name
            archive.mkdir(parents=True, exist_ok=True)
            write_json(archive/'run_config.json', previous)
            config['source_history'] = config['source_history']+[dict(
                reason=reason,
                previous=previous['source_sha256'], current=config['source_sha256'])]
    else:
        config['source_history'] = []
    write_json(config_path, config)
    tasks = pd.read_csv(args.reference/'forecast_tasks.csv', dtype={'date': str})
    tasks[tasks.date.isin(dates)].to_csv(args.out/'forecast_tasks.csv', index=False)
    print(f'{len(dates)} histories; G={args.groups}; {args.chains} chains; {args.iterations}/{args.burn} iterations/burn', flush=True)
    from pyspark.sql import SparkSession
    slots = min(args.spark_slots, len(cpu_ids))
    completed_before = sum((args.out/'days'/date/'completed.json').exists() for date in dates)
    started = time.monotonic()
    spark = (SparkSession.builder.master(f'local[{slots}]').appName('csi300-spectral-dc-pilot')
             .config('spark.driver.host', '127.0.0.1').config('spark.driver.bindAddress', '127.0.0.1')
             .config('spark.ui.enabled', 'false').config('spark.ui.showConsoleProgress', 'false')
             .config('spark.driver.maxResultSize', '4g').config('spark.python.worker.reuse', 'true')
             .config('spark.pyspark.python', sys.executable)
             .config('spark.executorEnv.PYTHONPATH', str(ROOT))
             .config('spark.executorEnv.NUMBA_CACHE_DIR', os.environ['NUMBA_CACHE_DIR'])
             .config('spark.executorEnv.MPLCONFIGDIR', os.environ['MPLCONFIGDIR'])
             .config('spark.executorEnv.OPENBLAS_NUM_THREADS', '1')
             .config('spark.executorEnv.OMP_NUM_THREADS', '1').config('spark.executorEnv.MKL_NUM_THREADS', '1')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    status = 'interrupted'
    try:
        for date in tqdm(dates, desc='Spark liquidity histories'):
            folder = args.out/'days'/date
            folder.mkdir(parents=True, exist_ok=True)
            if (folder/'completed.json').exists():
                continue
            (folder/'frequencies').mkdir(exist_ok=True)
            (folder/'shards').mkdir(exist_ok=True)
            saved_folder = args.reference/'days'/date
            write_json(folder/'progress.json', dict(stage='prepare', time=time.time()))
            saved = prepare_response(source, saved_folder, folder)
            n = saved['n']
            p = fft_partitions_for(n, args.fft_partitions)
            daily = {**config, 'n': n, 'actual_fft_partitions': p,
                     'day_seed': args.seed+saved['train_stop_day']*10_000}
            print(f'{date}: n={n:,}, FFT P={p}, G={args.groups}, {n//2//args.groups:,} frequencies/shard', flush=True)
            write_json(folder/'progress.json', dict(stage='distributed_fft', time=time.time()))
            t = time.monotonic()
            fft_checks = recover_frequency_checks(folder, saved_folder, n, args.groups)
            if fft_checks is None:
                response_path = str(folder/'response.npy')
                blocks = spark.sparkContext.parallelize(range(p), p).map(
                    lambda b: load_input_block(b, response_path, n, p))
                transformed = spark_fft_contiguous_blocks(blocks, n, p)
                groups = transformed.flatMap(lambda record: frequency_chunks(record, n, p, args.groups)).groupByKey(args.groups)
                folder_text, saved_text = str(folder), str(saved_folder)
                result = groups.map(lambda item: cache_frequency_group(item, folder_text, saved_text, n, args.groups))
                fft_checks = monitored_action(result.collect, folder/'frequencies', 'shard*.json',
                                              args.groups, f'{date} FFT groups')
            fft_seconds = time.monotonic()-t
            write_json(folder/'fft_validation.json', dict(n=n, fft_partitions=p, groups=args.groups,
                       ordinates=sum(x['ordinates'] for x in fft_checks), all_frequencies_checked=True,
                       max_absolute_error=max(x['max_absolute_error'] for x in fft_checks), seconds=fft_seconds))
            write_json(folder/'progress.json', dict(stage='spark_mcmc', time=time.time()))
            t = time.monotonic()
            folder_text = str(folder)
            result = spark.sparkContext.parallelize(range(args.groups), args.groups).map(
                lambda group: run_shard(group, folder_text, daily))
            records = monitored_action(result.collect, folder/'shards', '*.json', args.groups, f'{date} MCMC shards')
            sampling_seconds = time.monotonic()-t
            write_json(folder/'progress.json', dict(stage='consensus', time=time.time()))
            finished = finish_day(args.reference, args.out, date, records, daily, fft_checks, fft_seconds, sampling_seconds)
            write_json(folder/'progress.json', dict(stage='completed', time=time.time()))
            print(json.dumps(finished), flush=True)
        status = 'completed'
    finally:
        try:
            spark.stop()
        finally:
            record_execution(args.out, dates, slots, time.monotonic()-started, status, completed_before)
    elapsed = time.monotonic()-started
    summarize(args.out, args.reference, dates, elapsed)
    print(f'Finished: {args.out}', flush=True)


if __name__ == '__main__':
    main()
