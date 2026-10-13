#!/usr/bin/env python3
"""Actual daily Spectral DC boundary forecasts on a seeded random date sample.

Uses the manuscript likelihood, systematic frequency groups, fractional prior,
fixed-Hessian Gaussian RWMH, and joint precision consensus. All training dates
precede the forecast date. No likelihood powers, session factors or latent states.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/csi300-boundary-forecast-mpl')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.fft import rfft
from tqdm import tqdm
from dts.aggregation import consensus
from scripts.liquidity.compare_csi300_forecasts import PHASES, score_samples
from scripts.liquidity.fit_csi300_observed_dc import diagnostics
from scripts.liquidity.probe_csi300_fractional_changes import (
    fit_spectrum, increment_predictor)
from scripts.liquidity.run_csi300_fractional_changes_dc import compiled_rwm, target_value
from scripts.liquidity._source_compat import (
    compatible_source_relocation, sources_before_relocation)

MODELS = ['SpectralDC_ARTFIMA', 'Matched_ARIMA011_MAP', 'LastValue', 'ClockLastValue']
RUN_ARGUMENTS = ['input', 'plan', 'out', 'days', 'seed', 'ray_actors', 'groups', 'chains',
                 'iterations', 'burn', 'posterior_parameters', 'paths_per_parameter']
LEGACY_SOURCE_SHA256 = {
    'compare_csi300_boundary_random_days.py': 'e1ac3caaa9b4c01ef946e0196d300a41bcb098ba7ba0e33bf51fe017d64531d5',
    'probe_csi300_fractional_changes.py': 'cb291905d95e639b6c080e26572fced566cbfd5da81fa86220d3bd9711fd8cb7',
    'run_csi300_fractional_changes_dc.py': 'e54b5b96328395e0affcf090fdd17611122075ce2f06b11ff893d089b6e94059'}
NULL_MAP_FIX_SHA256 = 'cd5201ea2920e54debc83e20a0653f7951fd3e7f3ab1ec83900b73483a60d2d5'


def compatible_null_map_update(previous, current):
    """Permit only the reviewed status-check fix with unchanged daily inference code."""
    current = sources_before_relocation(current)
    names = ['clock_calibration', 'standardized_training_changes', 'increment_kernel',
             'weighted_variance_diagonal', 'prepare_day', 'chain_quality', 'sample_shard',
             'sample_group', 'posterior_comparison', 'prediction_record',
             'forecasts_on_day', 'run_day', 'summarize']
    source = Path(__file__).read_text()
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    core_hash = hashlib.sha256('\n'.join(functions[name] for name in names).encode()).hexdigest()
    return (previous == LEGACY_SOURCE_SHA256 and
            current.get('probe_csi300_fractional_changes.py') == NULL_MAP_FIX_SHA256 and
            current.get('run_csi300_fractional_changes_dc.py') == LEGACY_SOURCE_SHA256['run_csi300_fractional_changes_dc.py'] and
            core_hash == 'ed26b8e7d2dd4859867923cf926ad9f156f71acd14eeeb1a34d9bd7727186eab')


def write_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def register_protocol(path, config):
    """Keep provenance for the reviewed numerical correction or source relocation."""
    if path.exists():
        previous = json.loads(path.read_text())
        history = previous.pop('source_history', [])
        cached = any((path.parent/'days').glob('*/fit.json'))
        if previous != config and cached:
            old_settings = {key: value for key, value in previous.items() if key != 'source_sha256'}
            new_settings = {key: value for key, value in config.items() if key != 'source_sha256'}
            reason = None
            if old_settings == new_settings:
                if compatible_null_map_update(previous['source_sha256'], config['source_sha256']):
                    reason = 'Validated null MAP stationarity instead of optimizer status; original valid estimates and daily inference unchanged'
                elif compatible_source_relocation(previous['source_sha256'], config['source_sha256']):
                    reason = 'Moved scripts into scripts/liquidity; daily inference and forecasting unchanged'
            if reason is None:
                raise ValueError('Use a new output directory after incompatible configuration/source changes')
            history.append(dict(time=time.time(), previous=previous['source_sha256'],
                                current=config['source_sha256'],
                                reason=reason))
        if history:
            config['source_history'] = history
    write_json(path, config)


def clock_calibration(train):
    """Frozen mean/scale for each one-second within-phase log change.

    Pairwise availability avoids differences of means calculated on different
    date subsets. No calibration uses the forecast date or its outcomes.
    """
    if (np.isfinite(train)&(train <= 0)).any():
        raise ValueError('Nonpositive observed training spread')
    profile = np.empty(14220)
    mean = np.zeros(14220)
    scale = np.ones(14220)
    audit = []
    for phase, lo, hi in PHASES:
        log = np.log(train[:, lo:hi])
        changes = np.diff(log, axis=1)
        count = np.isfinite(changes).sum(axis=0)
        if count.min() < 3:
            raise ValueError('Insufficient earlier clock calibration pairs')
        mu = np.nanmean(changes, axis=0)
        sd = np.sqrt(np.nanmean((changes-mu)**2, axis=0))
        floor = max(1e-6, .05*float(np.median(sd)))
        mean[lo+1:hi] = mu
        scale[lo+1:hi] = np.maximum(sd, floor)
        scale[lo] = max(float(np.median(sd)), floor)
        profile[lo:hi] = np.nanmean(log[:, 0])+np.r_[0., np.cumsum(mu)]
        audit.append(dict(phase=phase, min_pairs=int(count.min()), scale_floor=floor,
                          floored_positions=int((sd < floor).sum())))
    return profile, mean, scale, audit


def standardized_training_changes(train, mean, scale):
    log = np.log(train)
    changes = np.full_like(log, np.nan)
    changes[:, 1:] = np.diff(log, axis=1)
    changes[:, 7200] = np.nan  # Lunch is never a one-second change.
    changes -= mean
    changes /= scale
    return changes[np.isfinite(changes)]  # Date, AM, PM chronological order.


def increment_kernel(parameters, context, horizon, fractional=True):
    cumulative_weights, cumulative_covariance = increment_predictor(parameters, context, horizon, fractional)
    weights = np.diff(np.vstack([np.zeros((1, context)), cumulative_weights]), axis=0)
    covariance = np.diff(np.diff(np.pad(cumulative_covariance, ((1, 0), (1, 0))), axis=0), axis=1)
    covariance = (covariance+covariance.T)/2
    chol = np.linalg.cholesky(covariance+np.eye(horizon)*1e-12)
    return weights, covariance, chol


def weighted_variance_diagonal(covariance, scale):
    """Diagonal of R diag(scale) covariance diag(scale) R', exactly."""
    return np.cumsum(2*scale*(np.tril(covariance)@scale)-scale**2*np.diag(covariance))


def prepare_day(source, folder, date, day_index, expected_n):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'fit.json').exists():
        fit = json.loads((folder/'fit.json').read_text())
        if fit['n'] != expected_n or fit['train_stop_day'] != day_index:
            raise ValueError('Cached training cutoff/count changed')
        return fit
    started = time.monotonic()
    meta = json.loads((source/'input_manifest.json').read_text())
    raw = np.load(source/'spread.npy', mmap_mode='r')
    train = raw[:day_index]
    profile, clock_mean, clock_scale, audit = clock_calibration(train)
    y = standardized_training_changes(train, clock_mean, clock_scale)
    scale = float(y.std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid training change scale')
    removed = len(y) % 2
    if removed:
        y = y[1:]
    y /= scale
    if len(y) < 10_000_000 or len(y) != expected_n:
        raise ValueError(f'Training constraint/count failed: {len(y)} versus {expected_n}')
    ft = rfft(y, workers=1)[1:len(y)//2]
    pg = np.abs(ft)**2/(2*np.pi*len(y))
    q = 4*np.sin(np.pi*np.arange(1, len(y)//2)/len(y))**2
    fit = fit_spectrum(q, pg)
    if fit['covariance'] is None or fit['newton_distance'] > .01 or not fit['null_success']:
        raise ValueError(f'Unresolved full/null MAP: {fit}')
    fit.update(date=date, n=len(y), scale=scale, change_audit=audit,
               train_stop_day=day_index, train_first=meta['dates'][0],
               train_last=meta['dates'][day_index-1], earliest_change_removed=removed,
               preparation_seconds=time.monotonic()-started)
    np.save(folder/'profile.npy', profile)
    np.save(folder/'clock_mean.npy', clock_mean)
    np.save(folder/'clock_scale.npy', clock_scale)
    np.save(folder/'q.npy', q)
    np.save(folder/'periodogram.npy', pg)
    write_json(folder/'fit.json', fit)
    return fit


def chain_quality(draws):
    rh, bulk = diagnostics(draws)
    limits = np.quantile(draws.reshape(-1, draws.shape[-1]), [.05, .95], axis=0)
    indicators = np.concatenate([(draws <= limits[0]).astype(float),
                                 (draws >= limits[1]).astype(float)], axis=2)
    _, tail = diagnostics(indicators)
    return dict(rhat_max=float(rh.max()), ess_bulk_min=float(bulk.min()),
                ess_tail_min=float(tail.min()),
                passed=bool(rh.max() <= 1.01 and bulk.min() >= 400 and tail.min() >= 400))


def sample_shard(folder, groups, group, chains, iterations, burn, seed):
    path = folder/f'G{groups}_shard{group}_n{iterations}.npz'
    if path.exists():
        with np.load(path) as cached:
            return cached['draws'], json.loads(str(cached['quality']))
    q = np.ascontiguousarray(np.load(folder/'q.npy', mmap_mode='r')[group::groups])
    pg = np.ascontiguousarray(np.load(folder/'periodogram.npy', mmap_mode='r')[group::groups])
    fit = fit_spectrum(q, pg, groups)
    if fit['covariance'] is None or fit['newton_distance'] > .01:
        raise ValueError('Unresolved shard MAP')
    chol = np.linalg.cholesky(np.array(fit['covariance'])*(2.38**2/4))
    center = np.array(fit['theta'])
    def chain(index):
        chain_seed = seed+groups*100_000+group*1000+index
        initial = center+chol@np.random.default_rng(chain_seed).normal(size=4)
        if not np.isfinite(target_value(initial, q, pg, groups)):
            initial = center.copy()
        n = json.loads((folder/'fit.json').read_text())['n']
        return compiled_rwm(q, pg, groups, initial, chol, iterations, burn, chain_seed, n, group)
    # Two DC groups run in parallel, with serial chains inside each group.
    # A full reference has one group, so its four chains can run together.
    if groups == 1:
        with ThreadPoolExecutor(max_workers=chains) as pool:
            result = list(pool.map(chain, range(chains)))
    else:
        result = [chain(index) for index in range(chains)]
    draws = np.array([item[0] for item in result])
    quality = chain_quality(draws)
    quality.update(acceptance=[float(item[1]) for item in result],
                   iterations=iterations, burn=burn, groups=groups, shard=group)
    np.savez_compressed(path, draws=draws, quality=json.dumps(quality), fit=json.dumps(fit))
    return draws, quality


def sample_group(folder, groups, config, seed, minimum_iterations=None):
    minimum = config['iterations'] if minimum_iterations is None else minimum_iterations
    stages = sorted(set([minimum, 8000, 15000]))
    stages = [n for n in stages if n >= minimum]
    for total in stages:
        burn = config['burn'] if total == config['iterations'] else (2000 if total == 8000 else 5000)
        with ThreadPoolExecutor(max_workers=min(groups, 2)) as pool:
            output = list(pool.map(lambda g: sample_shard(folder, groups, g,
                config['chains'], total, burn, seed), range(groups)))
        shape = output[0][0].shape
        draws = consensus([item[0].reshape(-1, 4) for item in output], ridge=1e-12).reshape(shape)
        quality = chain_quality(draws)
        quality.update(groups=groups, iterations=total, burn=burn,
                       shards=[item[1] for item in output])
        quality['passed'] = quality['passed'] and all(item[1]['passed'] for item in output)
        np.save(folder/f'G{groups}_draws.npy', draws)
        write_json(folder/f'G{groups}_quality.json', quality)
        if quality['passed']:
            return draws, quality
    raise ValueError(f'Sampling diagnostics failed at 15000 iterations: {quality}')


def posterior_comparison(reference, candidate):
    full = reference.reshape(-1, 4)
    dc = candidate.reshape(-1, 4)
    sd = full.std(axis=0, ddof=1)
    shift = (dc.mean(axis=0)-full.mean(axis=0))/sd
    ratio = dc.std(axis=0, ddof=1)/sd
    return dict(parameters=['theta', 'log_lambda', 'log_sigma2', 'd'],
                mean_shift_reference_sd=shift.tolist(), sd_ratio=ratio.tolist(),
                passed=bool((np.abs(shift) < .1).all() and ((ratio >= .9)&(ratio <= 1.1)).all()))


def prediction_record(actual, point, paths):
    if not np.isfinite(point) or not np.isfinite(paths).all():
        raise ValueError('Nonfinite predictive distribution')
    record = score_samples(actual, point, paths)
    if not np.isfinite(actual):
        for key in ['ae', 'se', 'crps', 'cover80', 'cover95', 'interval80', 'interval95']:
            record[key] = np.nan
    return record


def forecasts_on_day(source, folder, tasks, fit, draws, config, seed):
    path = folder/'forecasts.parquet'
    if path.exists():
        return pd.read_parquet(path)
    raw = np.load(source/'spread.npy', mmap_mode='r')[int(tasks.day_index.iloc[0])]
    profile = np.load(folder/'profile.npy')
    clock_mean = np.load(folder/'clock_mean.npy')
    clock_scale = fit['scale']*np.load(folder/'clock_scale.npy')
    rng = np.random.default_rng(seed+7)
    per_chain = config['posterior_parameters']//draws.shape[0]
    parameters = np.concatenate([chain[rng.choice(len(chain), per_chain, replace=False)] for chain in draws])
    path_count = len(parameters)*config['paths_per_parameter']
    active = tasks[tasks.issued]
    origins = list(active.groupby('origin', sort=True))
    # Exact finite-context Gaussian covariance from the fitted spectrum.
    predictor = {}
    for context in sorted(active.context_changes.unique()):
        predictor[context] = []
        for theta in parameters:
            predictor[context].append(increment_kernel(theta, int(context), 300))
    null = [fit['null_theta'][0], 0., fit['null_theta'][1], 0.]
    null_predictor = {}
    for context in predictor:
        null_predictor[context] = increment_kernel(null, int(context), 300, fractional=False)
    rows = []
    for origin, group in origins:
        row = group.iloc[0]
        lo = 0 if row.phase == 'am' else 7200
        end = lo+int(row.elapsed)
        context = int(row.context_levels)
        horizon = int(group.horizon.max())
        history = (np.diff(np.log(raw[end-context:end]))-clock_mean[end-context+1:end])/clock_scale[end-context+1:end]
        anchor = np.log(raw[end-1])
        offset = profile[end:end+horizon]-profile[end-1]
        future_scale = clock_scale[end:end+horizon]
        candidate_paths, candidate_points = [], []
        for weights, covariance, chol in predictor[context-1]:
            logmean = anchor+offset+np.cumsum(future_scale*(weights[:horizon]@history))
            half = rng.normal(size=(config['paths_per_parameter']//2, horizon))
            normal = np.r_[half, -half]
            variance = weighted_variance_diagonal(covariance[:horizon, :horizon], future_scale)
            candidate_paths.append(np.exp(logmean+np.cumsum(future_scale*(normal@chol[:horizon, :horizon].T), axis=1)))
            candidate_points.append(np.exp(logmean+.5*variance))
        paths = {MODELS[0]: np.concatenate(candidate_paths)}
        points = {MODELS[0]: np.mean(candidate_points, axis=0)}
        half = rng.normal(size=(path_count//2, horizon))
        normal = np.r_[half, -half]
        weights, covariance, chol = null_predictor[context-1]
        null_mean = anchor+offset+np.cumsum(future_scale*(weights[:horizon]@history))
        variance = weighted_variance_diagonal(covariance[:horizon, :horizon], future_scale)
        paths[MODELS[1]] = np.exp(null_mean+np.cumsum(future_scale*(normal@chol[:horizon, :horizon].T), axis=1))
        points[MODELS[1]] = np.exp(null_mean+.5*variance)
        for model, clock in [(MODELS[2], np.zeros(horizon)), (MODELS[3], offset)]:
            points[model] = np.exp(anchor+clock)
            paths[model] = np.exp(anchor+clock+np.cumsum(future_scale*normal, axis=1))
        for task in group.itertuples():
            h = int(task.horizon)
            actual = float(raw[end:end+h].mean()) if task.target_ok else np.nan
            for model in MODELS:
                rows.append(dict(date=task.date, day_index=task.day_index, event=task.event,
                    origin=origin, horizon=h, context_levels=context, model=model,
                    train_fft_observations=fit['n'], target_ok=task.target_ok,
                    **prediction_record(actual, float(points[model][:h].mean()), paths[model][:, :h].mean(axis=1))))
    result = pd.DataFrame(rows)
    result.to_parquet(path, index=False)
    return result


def run_day(job):
    source, root, tasks, config, reference = job
    date = str(tasks.date.iloc[0])
    folder = root/'days'/date
    if (folder/'completed.json').exists():
        return json.loads((folder/'completed.json').read_text())
    started = time.monotonic()
    index = int(tasks.day_index.iloc[0])
    seed = config['seed']+index*10_000
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'progress.json', dict(stage='prepare', started=time.time()))
    fit = prepare_day(source, folder, date, index, int(tasks.train_fft_observations.iloc[0]))
    write_json(folder/'progress.json', dict(stage='dc_sampling', started=time.time()))
    draws, quality = sample_group(folder, config['groups'], config, seed)
    accuracy = None
    if reference:
        write_json(folder/'progress.json', dict(stage='full_reference', started=time.time()))
        full, full_quality = sample_group(folder, 1, config, seed, minimum_iterations=8000)
        accuracy = posterior_comparison(full, draws)
        for total in [8000, 15000]:
            if accuracy['passed']:
                break
            draws, quality = sample_group(folder, config['groups'], config, seed, minimum_iterations=total)
            if total == 15000:
                full, full_quality = sample_group(folder, 1, config, seed, minimum_iterations=15000)
            accuracy = posterior_comparison(full, draws)
        write_json(folder/'dc_vs_full.json', accuracy)
    write_json(folder/'progress.json', dict(stage='forecast', started=time.time()))
    results = forecasts_on_day(source, folder, tasks, fit, draws, config, seed)
    completed = dict(date=date, train_fft_observations=fit['n'], rows=len(results),
        scorable_tasks=int(tasks.scorable.sum()), iterations=quality['iterations'],
        sampling_passed=quality['passed'], reference_check=reference,
        posterior_accuracy_passed=accuracy['passed'] if accuracy else None,
        seconds=time.monotonic()-started)
    write_json(folder/'completed.json', completed)
    write_json(folder/'progress.json', dict(stage='completed', started=time.time()))
    return completed


class DayActor:
    """A daily task owns all of its cache, posterior and forecast files."""
    def process(self, job):
        return run_day(job)


def summarize(root, config):
    files = sorted((root/'days').glob('*/forecasts.parquet'))
    if not files:
        return
    a = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    keys = ['date', 'origin', 'horizon']
    if a.duplicated(keys+['model']).any() or not a.groupby(keys).model.nunique().eq(4).all():
        raise ValueError('Unmatched forecast comparisons')
    scored = a[np.isfinite(a.actual)].copy()
    if not scored.groupby(keys).actual.nunique().eq(1).all():
        raise ValueError('Different outcomes across models')
    a.to_parquet(root/'forecasts.parquet', index=False)
    def scores(group):
        s = group.agg(n=('ae', 'size'), days=('date', 'nunique'), mae=('ae', 'mean'),
                      mse=('se', 'mean'), crps=('crps', 'mean'), coverage95=('cover95', 'mean')).reset_index()
        s['rmse'] = np.sqrt(s.pop('mse'))
        return s
    total = scores(scored.groupby(['model', 'horizon']))
    event = scores(scored.groupby(['event', 'origin', 'model', 'horizon']))
    total.to_csv(root/'scores.csv', index=False)
    event.to_csv(root/'event_scores.csv', index=False)
    rng = np.random.default_rng(config['seed']+11)
    comparisons = []
    for h, g in scored.groupby('horizon'):
        wide = g.pivot(index=['day_index', 'origin'], columns='model', values='se')
        for baseline in MODELS[1:]:
            x = wide[[MODELS[0], baseline]].copy()
            x['cluster'] = x.index.get_level_values('day_index')//20
            cluster = x.groupby('cluster').agg({MODELS[0]: 'sum', baseline: 'sum'})
            count = x.groupby('cluster').size().to_numpy()
            values = cluster.to_numpy()
            indices = rng.integers(len(cluster), size=(2000, len(cluster)))
            summed = values[indices].sum(axis=1)/count[indices].sum(axis=1)[:, None]
            gains = 100*(1-np.sqrt(summed[:, 0]/summed[:, 1]))
            actual_gain = 100*(1-np.sqrt(x[MODELS[0]].mean()/x[baseline].mean()))
            comparisons.append(dict(horizon=int(h), baseline=baseline,
                rmse_reduction_pct=actual_gain, lower_pct=float(np.quantile(gains, .025)),
                upper_pct=float(np.quantile(gains, .975)), clusters=len(cluster)))
    pd.DataFrame(comparisons).to_csv(root/'paired_rmse.csv', index=False)
    completed = [json.loads(p.read_text()) for p in sorted((root/'days').glob('*/completed.json'))]
    pd.DataFrame(completed).to_csv(root/'daily_diagnostics.csv', index=False)
    report = ['Seeded random-date boundary forecasting comparison',
              'Actual manuscript Spectral DC posterior forecasts; matched ARIMA uses full Whittle MAP.',
              'Daily parameter posterior frozen; finite-context Gaussian conditional paths; second-specific clock change mean/scale fitted on earlier dates only.',
              'Last-value benchmark intervals use training-scaled log random walks; their point forecasts are raw/clock-adjusted persistence.',
              'Historical approximate weights, available-constituent selection and observed-time stationarity assumptions remain.',
              'Intervals resample nonoverlapping 20-archive-date clusters among sampled dates; exploratory, not a confirmatory test.',
              '', total.to_string(index=False), '', pd.DataFrame(comparisons).to_string(index=False),
              '', pd.DataFrame(completed).to_string(index=False)]
    (root/'REPORT.txt').write_text('\n'.join(report)+'\n')
    print(total.to_string(index=False), flush=True)


def parse_arguments(argv=None):
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=Path)
    initial, _ = pre_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home()/'running/dts/artifacts/liquidity'
    parser.add_argument('--config', type=Path, help='Load run_config.json; explicit command-line options override its values')
    parser.add_argument('--input', type=Path, default=base/'csi300_dc_1m')
    parser.add_argument('--plan', type=Path, default=base/'csi300_boundary_rolling_plan_usable')
    parser.add_argument('--out', type=Path, default=base/'csi300_boundary_random100_v2')
    parser.add_argument('--days', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--ray-actors', type=int, default=50)
    parser.add_argument('--groups', type=int, default=2)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=4000)
    parser.add_argument('--burn', type=int, default=1000)
    parser.add_argument('--posterior-parameters', type=int, default=64)
    parser.add_argument('--paths-per-parameter', type=int, default=8)
    parser.add_argument('--plan-only', action='store_true')
    if initial.config:
        try:
            settings = json.loads(initial.config.expanduser().read_text())
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot load configuration: {exc}')
        if not isinstance(settings, dict) or set(settings)-set(RUN_ARGUMENTS):
            parser.error('Configuration must contain only supported run arguments')
        for key, value in settings.items():
            if key in ['input', 'plan', 'out']:
                if not isinstance(value, str):
                    parser.error(f'Configuration {key} must be a path string')
            elif type(value) is not int:
                parser.error(f'Configuration {key} must be an integer')
        parser.set_defaults(**settings)
    args = parser.parse_args(argv)
    for key in ['input', 'plan', 'out']:
        setattr(args, key, getattr(args, key).expanduser().absolute())
    if args.groups < 2 or args.chains < 4 or not 1 <= args.days <= 1000 or args.iterations <= args.burn or args.burn < 0 or args.ray_actors < 1:
        parser.error('Need distributed groups, >=4 chains, 1-1000 dates and retained draws')
    if args.posterior_parameters < args.chains or args.posterior_parameters % args.chains or args.paths_per_parameter < 2 or args.paths_per_parameter % 2:
        parser.error('Parameter draws must divide across chains; path count must be even')
    return args


def run_configuration(args):
    settings = {key: getattr(args, key) for key in RUN_ARGUMENTS}
    return {key: str(value) if isinstance(value, Path) else value
            for key, value in settings.items()}


def main():
    args = parse_arguments()
    plan = pd.read_csv(args.plan/'daily_fits.csv', dtype={'date': str})
    if len(plan) != 1000 or not plan.fit_required.all():
        raise ValueError('Expected the agreed 1000 usable-date plan')
    selected = np.sort(np.random.default_rng(args.seed).choice(plan.date.to_numpy(), args.days, replace=False))
    tasks = pd.read_csv(args.plan/'forecast_tasks.csv', dtype={'date': str})
    tasks = tasks[tasks.date.isin(selected)].copy()
    if tasks.train_fft_observations.min() < 10_000_000:
        raise ValueError('Training sample shorter than 10 million')
    references = sorted(set(selected[[0, len(selected)//2, -1]]))
    config = {key: getattr(args, key) for key in ['days', 'seed', 'groups', 'chains', 'iterations', 'burn', 'posterior_parameters', 'paths_per_parameter']}
    config.update(dates=selected.tolist(), references=references,
        ray_actors=args.ray_actors,
        clock_calibration='Earlier-training pairwise mean and population SD of each clock-second log change; SD floor max(1e-6,0.05*phase median); frozen within day',
        response='Y=(within-session log change - frozen clock mean)/(frozen clock SD * global training scale)',
        inverse='log future spread = log origin spread + cumulative clock means + cumulative clock-scaled future Y',
        source=str(args.input.resolve()), input_manifest_sha256=hashlib.sha256((args.input/'input_manifest.json').read_bytes()).hexdigest(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), Path(__file__).with_name('probe_csi300_fractional_changes.py'), Path(__file__).with_name('run_csi300_fractional_changes_dc.py')]},
        inference='Native manuscript systematic spectral DC, fractional prior, fixed Hessian RWMH, joint consensus; no adaptive proposals',
        sampling_gate='All shard/combined Rhat<=1.01, bulk/tail ESS>=400; extend failed checks to 8000 then 15000 iterations',
        accuracy_gate='Full-reference absolute mean shift <0.1 full SD; SD ratio 0.9-1.1; extend failed accuracy checks using the unchanged target',
        forecasting='64 balanced joint posterior draws by default, 8 antithetic conditional paths per draw; Gaussian finite-context conditioning; daily fit frozen')
    args.out.mkdir(parents=True, exist_ok=True)
    cp = args.out/'protocol.json'
    register_protocol(cp, config)
    write_json(args.out/'run_config.json', run_configuration(args))
    tasks.to_csv(args.out/'forecast_tasks.csv', index=False)
    plan[plan.date.isin(selected)].to_csv(args.out/'sample_dates.csv', index=False)
    print(f'{len(selected)} dates: {selected[0]}-{selected[-1]}; {tasks.issued.sum()} issuable / {tasks.scorable.sum()} scorable tasks; training {tasks.train_fft_observations.min():,}-{tasks.train_fft_observations.max():,}', flush=True)
    if args.plan_only:
        return
    # Warm numerical kernels once before threaded jobs; no inference performed here.
    compiled_rwm(np.array([.1, .2]), np.array([.1, .2]), 1,
                 np.array([.1, -2., 0., .2]), np.eye(4)*.001, 5, 1, args.seed)
    jobs = [(args.input, args.out, tasks[tasks.date == date], config, date in references) for date in selected]
    jobs.sort(key=lambda job: not job[-1])
    import ray
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    actor_count = min(args.ray_actors, len(jobs))
    cpus_per_actor = min(2., available/actor_count)
    ray.init(num_cpus=available, include_dashboard=False, log_to_driver=False,
             object_store_memory=512*1024**2, _temp_dir='/tmp/ray-csi300-boundary-'+str(args.seed))
    actor_type = ray.remote(num_cpus=cpus_per_actor, max_restarts=0)(DayActor)
    actors = [actor_type.remote() for _ in range(actor_count)]
    remaining = iter(jobs)
    pending = {}
    for actor in actors:
        job = next(remaining, None)
        if job is not None:
            pending[actor.process.remote(job)] = (actor, str(job[2].date.iloc[0]))
    failures = []
    last_refresh = time.monotonic()
    with tqdm(total=len(jobs), desc='Random-date daily DC forecasts (Ray)', mininterval=10) as progress:
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=10)
            if time.monotonic()-last_refresh >= 30:
                stages = {}
                for file in (args.out/'days').glob('*/progress.json'):
                    stage = json.loads(file.read_text())['stage']
                    stages[stage] = stages.get(stage, 0)+1
                progress.set_postfix(stages, refresh=True)
                last_refresh = time.monotonic()
            for ref in ready:
                actor, date = pending.pop(ref)
                try:
                    print(json.dumps(ray.get(ref)), flush=True)
                except Exception as exc:
                    failures.append(dict(date=date, error=repr(exc)))
                    write_json(args.out/'failures.json', failures)
                    print(f'{date}: {exc!r}', flush=True)
                progress.update(1)
                progress.set_postfix(last=date, failures=len(failures), refresh=False)
                job = next(remaining, None)
                if job is not None:
                    pending[actor.process.remote(job)] = (actor, str(job[2].date.iloc[0]))
    ray.shutdown()
    write_json(args.out/'failures.json', failures)
    summarize(args.out, config)
    print(args.out, flush=True)


if __name__ == '__main__':
    main()
