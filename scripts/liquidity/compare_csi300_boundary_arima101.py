#!/usr/bin/env python3
"""Add ARIMA(1,0,1) on clock-adjusted log levels to a saved comparison.

Reuse its dates, historical cutoffs, clock profiles, conditioning windows and
targets. ARMA(1,1) is fitted with the full Whittle posterior on observed-time
levels, with no differencing. Original forecasts remain in the input folder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from numba import njit
from scipy.fft import rfft
from scipy.linalg import solve_toeplitz, toeplitz
from scipy.optimize import minimize
from tqdm import tqdm
from scripts.liquidity.compare_csi300_boundary_random_days import prediction_record, write_json

MODEL = 'ARIMA101_Levels_MAP'
RUN_ARGUMENTS = ['comparison', 'out', 'ray_actors', 'paths', 'seed']
PREVIOUS_SOURCE_SHA256 = '65a9320f68ce94dda28c1c1f3723d7ad673a4ee0ca81ac8bcb0caefa80fac6e1'


def comparison_reference(comparison, previous):
    """Accept the original comparison and comparisons with added benchmarks."""
    reference = Path(previous.get('reference', comparison))
    dates = previous.get('dates')
    if dates is None:
        dates = json.loads((comparison/'protocol.json').read_text())['dates']
    return reference, sorted(dates)


def log_product(coefficient, n):
    """Sum log|1-coefficient*exp(-i*omega)|^2 over interior frequencies."""
    root = coefficient**n
    value = np.log1p(-root)-np.log1p(-coefficient*coefficient)
    first = (-n*coefficient**(n-1)/(1-root)
             + 2*coefficient/(1-coefficient*coefficient))
    second = (-n*(n-1)*coefficient**(n-2)/(1-root)
              - n*n*coefficient**(2*n-2)/(1-root)**2
              + 2*(1+coefficient*coefficient)/(1-coefficient*coefficient)**2)
    return value, first, second


@njit(cache=True, nogil=True)
def arma_sums(phi, theta, q, pg):
    """Full-frequency ratio sum and its coefficient derivatives; no subsampling."""
    energy = p = t = pp = pt = tt = 0.
    for j in range(len(q)):
        denominator = (1-phi)**2+phi*q[j]
        numerator = (1+theta)**2-theta*q[j]
        dp = 2*(phi-1)+q[j]
        dt = 2*(1+theta)-q[j]
        weight = pg[j]/numerator
        slope = dt/numerator
        energy += weight*denominator
        p += weight*dp
        t -= weight*denominator*slope
        pp += 2*weight
        pt -= weight*dp*slope
        tt += weight*denominator*(2*slope*slope-2/numerator)
    return energy, p, t, pp, pt, tt


def arma_objective(parameters, q, pg, n):
    phi, theta, log_variance = parameters
    if n < 4 or n % 2 or len(pg) != n//2-1 or len(q) != len(pg):
        raise ValueError('Requires the complete interior positive-frequency spectrum of even n')
    if abs(phi) >= 1 or abs(theta) >= 1:
        raise ValueError('Requires stationary AR and invertible MA coefficients')
    ar, arp, arpp = log_product(phi, n)
    ma, map_, mapp = log_product(-theta, n)
    energy, p, t, pp, pt, tt = arma_sums(phi, theta, q, pg)
    multiplier = 2*np.pi*np.exp(-log_variance)
    value = (len(pg)*(log_variance-np.log(2*np.pi))+ma-ar+multiplier*energy
             + .5*log_variance**2+2*np.log(2)+.5*np.log(2*np.pi))
    gradient = np.array([-arp+multiplier*p, -map_+multiplier*t,
                         len(pg)-multiplier*energy+log_variance])
    hessian = np.array([[multiplier*pp-arpp, multiplier*pt, -multiplier*p],
                        [multiplier*pt, multiplier*tt+mapp, -multiplier*t],
                        [-multiplier*p, -multiplier*t, multiplier*energy+1]])
    return value, gradient, hessian


def fit_arma(q, pg, n, initial_theta=0.):
    sum_pg = float(np.sum(pg))
    phi = float(np.clip(1-float(q@pg)/(2*sum_pg), -.99, .999999))
    theta = float(np.clip(initial_theta, -.9, .9))
    energy = arma_sums(phi, theta, q, pg)[0]
    initial = np.array([np.arctanh(phi), np.arctanh(theta),
                        np.log(2*np.pi*energy/len(pg))])
    bound = np.arctanh(.99999999)

    def objective(z):
        parameters = np.r_[np.tanh(z[:2]), z[2]]
        value, gradient, _ = arma_objective(parameters, q, pg, n)
        jacobian = np.r_[1-parameters[:2]**2, 1.]
        # This is an optimization coordinate change; the native prior is unchanged.
        return value/len(pg), gradient*jacobian/len(pg)

    attempts = []
    best = None
    for start in [initial, np.array([np.arctanh(.99), 0., initial[2]])]:
        result = minimize(objective, start, jac=True, method='L-BFGS-B',
                          bounds=[(-bound, bound), (-bound, bound), (-30., 20.)],
                          options={'ftol': 1e-15, 'gtol': 1e-12, 'maxiter': 500, 'maxls': 100})
        parameters = np.r_[np.tanh(result.x[:2]), result.x[2]]
        value, gradient, hessian = arma_objective(parameters, q, pg, n)
        eigenvalues = np.linalg.eigvalsh(hessian)
        distance = (float(np.sqrt(max(0., gradient@np.linalg.solve(hessian, gradient))))
                    if eigenvalues.min() > 0 else float('inf'))
        candidate = dict(phi=float(parameters[0]), theta=float(parameters[1]),
                         log_sigma2=float(parameters[2]), sigma2=float(np.exp(parameters[2])),
                         n=int(n), optimizer_success=bool(result.success), message=str(result.message),
                         newton_distance=distance, hessian_eigenvalues=eigenvalues.tolist(),
                         objective=float(value), passed=bool(np.isfinite(value) and distance < .01))
        attempts.append(candidate)
        if best is None or candidate['objective'] < best['objective']:
            best = candidate
        if candidate['passed']:
            best = candidate
            break
    if not best['passed']:
        raise ValueError(f'Unresolved ARMA(1,1) MAP: {attempts}')
    return dict(best, attempts=attempts)


def arma_kernel(phi, theta, sigma2, context, horizon):
    """Exact stationary Gaussian ARMA(1,1) forecast given a finite history."""
    gamma = np.empty(context+horizon)
    gamma[0] = sigma2*(1+theta*theta+2*phi*theta)/(1-phi*phi)
    gamma[1:] = (sigma2*(phi+theta)*(1+phi*theta)/(1-phi*phi)
                 * phi**np.arange(context+horizon-1))
    lags = np.arange(1, horizon+1)[:, None]+np.arange(context-1, -1, -1)[None, :]
    cross = gamma[lags]
    weights = solve_toeplitz(gamma[:context], cross.T).T
    covariance = toeplitz(gamma[:horizon])-weights@cross.T
    covariance = (covariance+covariance.T)/2
    return weights, covariance


def run_day(reference, source, out, date, records, config):
    started = time.monotonic()
    folder = reference/'days'/date
    destination = out/'days'/date
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'completed.json').exists():
        return json.loads((destination/'completed.json').read_text())
    saved = json.loads((folder/'fit.json').read_text())
    tasks = pd.DataFrame(records)
    index = int(tasks.day_index.iloc[0])
    if (saved['n'] < 10_000_000 or saved['train_stop_day'] != index
            or str(saved['train_last']) >= date):
        raise ValueError('Training size or strictly earlier cutoff changed')
    raw = np.load(source/'spread.npy', mmap_mode='r')
    profile = np.load(folder/'profile.npy')
    fit_path = destination/'fit.json'
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
    else:
        if (np.isfinite(raw[:index]) & (raw[:index] <= 0)).any():
            raise ValueError('Nonpositive observed training spread')
        log = np.log(raw[:index])
        log -= profile
        levels = log[np.isfinite(log)]
        del log
        removed = len(levels) % 2
        if removed:
            levels = levels[1:]
        shift, scale = float(levels.mean()), float(levels.std())
        if len(levels) < 10_000_000 or not np.isfinite(scale) or scale <= 0:
            raise ValueError('Insufficient or degenerate historical level observations')
        levels -= shift
        levels /= scale
        n = len(levels)
        ft = rfft(levels, workers=1)[1:n//2]
        pg = np.abs(ft)**2/(2*np.pi*n)
        del ft, levels
        q = 4*np.sin(np.pi*np.arange(1, n//2)/n)**2
        fit = fit_arma(q, pg, n, initial_theta=saved['null_theta'][0])
        fit.update(date=date, level_shift=shift, level_scale=scale,
                   train_first=saved['train_first'], train_last=saved['train_last'],
                   train_stop_day=index, earliest_level_removed=removed,
                   source_change_observations=saved['n'])
        write_json(fit_path, fit)
    today = raw[index]
    predictors = {}
    for context in tasks[tasks.issued].context_levels.unique():
        weights, covariance = arma_kernel(fit['phi'], fit['theta'], fit['sigma2'], int(context), 300)
        chol = np.linalg.cholesky(covariance+np.eye(300)*1e-12)
        predictors[int(context)] = weights, covariance, chol
    rows = []
    for origin, group in tasks[tasks.issued].groupby('origin', sort=True):
        row = group.iloc[0]
        end = (0 if row.phase == 'am' else 7200)+int(row.elapsed)
        context, horizon = int(row.context_levels), int(group.horizon.max())
        observed = today[end-context:end]
        if not (np.isfinite(observed).all() and (observed > 0).all()):
            raise ValueError('Reference conditioning history changed')
        history = (np.log(observed)-profile[end-context:end]-fit['level_shift'])/fit['level_scale']
        weights, covariance, chol = predictors[context]
        logmean = (profile[end:end+horizon]+fit['level_shift']
                   + fit['level_scale']*(weights[:horizon]@history))
        variance = fit['level_scale']**2*np.diag(covariance[:horizon, :horizon])
        point = np.exp(logmean+.5*variance)
        rng = np.random.default_rng(config['seed']+index*100_000+end)
        half = rng.normal(size=(config['paths']//2, horizon))
        normal = np.r_[half, -half]
        paths = np.exp(logmean+fit['level_scale']*(normal@chol[:horizon, :horizon].T))
        for task in group.itertuples():
            h = int(task.horizon)
            target = today[end:end+h]
            ok = bool(np.isfinite(target).all() and (target > 0).all())
            if ok != bool(task.target_ok):
                raise ValueError('Reference target availability changed')
            actual = float(target.mean()) if ok else np.nan
            rows.append(dict(date=date, day_index=index, event=task.event, origin=origin,
                             horizon=h, context_levels=context, model=MODEL,
                             train_fft_observations=fit['n'], target_ok=ok,
                             **prediction_record(actual, float(point[:h].mean()), paths[:, :h].mean(axis=1))))
    result = pd.DataFrame(rows)
    result.to_parquet(destination/'forecasts.parquet', index=False)
    completed = dict(date=date, n=fit['n'], phi=fit['phi'], theta=fit['theta'], sigma2=fit['sigma2'],
                     newton_distance=fit['newton_distance'], rows=len(result),
                     scored=int(np.isfinite(result.actual).sum()), seconds=time.monotonic()-started)
    write_json(destination/'completed.json', completed)
    return completed


class DayActor:
    def process(self, reference, source, out, date, records, config):
        return run_day(reference, source, out, date, records, config)


def summarize(comparison, out, dates, config):
    original = pd.read_parquet(comparison/'forecasts.parquet')
    original['date'] = original.date.astype(str)
    new = pd.concat([pd.read_parquet(out/'days'/date/'forecasts.parquet') for date in dates], ignore_index=True)
    keys = ['date', 'origin', 'horizon']
    expected = original[original.model.eq('SpectralDC_ARTFIMA')]
    matched = new.merge(expected, on=keys, suffixes=('_new', '_old'), validate='one_to_one')
    if len(matched) != len(new) or len(matched) != len(expected):
        raise ValueError('Unmatched forecast origins')
    np.testing.assert_array_equal(matched.actual_new, matched.actual_old)
    forecasts = pd.concat([original, new], ignore_index=True)
    if forecasts.duplicated(keys+['model']).any():
        raise ValueError('Duplicate model/origin pairs')
    scored = forecasts[np.isfinite(forecasts.actual)].copy()
    scored['error'] = scored.prediction-scored.actual
    scored['width95'] = scored.upper95-scored.lower95
    if not scored.groupby(keys).model.nunique().eq(original.model.nunique()+1).all():
        raise ValueError('Unmatched scored outcomes')

    def scores(groups):
        result = groups.agg(n=('se', 'size'), days=('date', 'nunique'), mse=('se', 'mean'),
                            mae=('ae', 'mean'), bias=('error', 'mean'), crps=('crps', 'mean'),
                            coverage95=('cover95', 'mean'), width95=('width95', 'mean'),
                            interval_score95=('interval95', 'mean')).reset_index()
        result['rmse'] = np.sqrt(result.pop('mse'))
        return result

    total = scores(scored.groupby(['model', 'horizon']))
    event = scores(scored.groupby(['event', 'origin', 'model', 'horizon']))
    rng = np.random.default_rng(config['seed']+11)
    comparisons = []
    for horizon, group in scored.groupby('horizon'):
        wide = group.pivot(index=['day_index', 'origin'], columns='model', values='se')
        for candidate, baseline in [('SpectralDC_ARTFIMA', MODEL), (MODEL, 'Matched_ARIMA110_MAP'),
                                    (MODEL, 'Matched_ARIMA011_MAP')]:
            if candidate not in wide or baseline not in wide:
                continue
            pair = wide[[candidate, baseline]].copy()
            pair['cluster'] = pair.index.get_level_values('day_index')//20
            cluster = pair.groupby('cluster')
            values = cluster[[candidate, baseline]].sum().to_numpy()
            counts = cluster.size().to_numpy()
            indices = rng.integers(len(values), size=(2000, len(values)))
            mse = values[indices].sum(axis=1)/counts[indices].sum(axis=1)[:, None]
            gains = 100*(1-np.sqrt(mse[:, 0]/mse[:, 1]))
            comparisons.append(dict(horizon=int(horizon), candidate=candidate, baseline=baseline,
                                    rmse_reduction_pct=100*(1-np.sqrt(pair[candidate].mean()/pair[baseline].mean())),
                                    lower_pct=float(np.quantile(gains, .025)),
                                    upper_pct=float(np.quantile(gains, .975)), clusters=len(values)))
    paired = pd.DataFrame(comparisons)
    forecasts.to_parquet(out/'forecasts.parquet', index=False)
    total.to_csv(out/'scores.csv', index=False)
    event.to_csv(out/'event_scores.csv', index=False)
    paired.to_csv(out/'paired_rmse.csv', index=False)
    diagnostics = pd.DataFrame([json.loads((out/'days'/date/'completed.json').read_text()) for date in dates])
    diagnostics.to_csv(out/'daily_diagnostics.csv', index=False)
    report = [f'ARIMA(1,0,1) added to the saved {len(dates)}-date liquidity comparison',
              'ARMA(1,1) on training-clock-adjusted log-spread LEVELS, centered and scaled using earlier dates only.',
              'Same historical dates, frozen clock profile, recent level windows, origins and complete targets.',
              'Full positive-frequency Whittle MAP; uniform stationary/invertible coefficients and N(0,1) log variance.',
              'All frequencies are used; analytic log products accelerate the same Whittle likelihood.',
              'Level observations include session/run beginnings, so their count differs from retained changes.',
              'Valid levels are concatenated in observed time across gaps and session boundaries.',
              'This adds a stationary-level assumption and constant level innovation scale; the change models use clock-scaled changes.',
              'Existing Spectral DC and ARIMA forecasts are reused without changing them.',
              'MAP ARIMA forecasts omit parameter uncertainty; existing DC accuracy/data-construction limitations remain.',
              'Exploratory comparison; uncertainty intervals resample 20-archive-date clusters among sampled dates.',
              '', total.to_string(index=False), '', paired.to_string(index=False)]
    (out/'REPORT.txt').write_text('\n'.join(report)+'\n')
    print(total.to_string(index=False), flush=True)
    print(paired.to_string(index=False), flush=True)


def parse_arguments(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args(argv)
    defaults = {}
    if known.config:
        saved = json.loads(known.config.read_text())
        defaults = {key: saved[key] for key in RUN_ARGUMENTS if key in saved}
    base = Path.home()/'running/dts/artifacts/liquidity'
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument('--comparison', type=Path, default=base/'csi300_boundary_arima110_quick100_v1')
    parser.add_argument('--out', type=Path, default=base/'csi300_boundary_arima101_quick100_v1')
    parser.add_argument('--ray-actors', type=int, default=50)
    parser.add_argument('--paths', type=int, default=512)
    parser.add_argument('--seed', type=int)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.ray_actors < 1 or args.paths < 2 or args.paths % 2:
        parser.error('Positive actors and an even path count >= 2 are required')
    return args


def main():
    args = parse_arguments()
    previous = json.loads((args.comparison/'run_config.json').read_text())
    reference, dates = comparison_reference(args.comparison, previous)
    source = Path(json.loads((reference/'run_config.json').read_text())['input'])
    tasks = pd.read_csv(args.comparison/'forecast_tasks.csv', dtype={'date': str})
    if sorted(tasks.date.unique()) != dates:
        raise ValueError('Comparison date manifest changed')
    if args.seed is None:
        args.seed = previous['seed']
    config = {key: str(getattr(args, key)) if isinstance(getattr(args, key), Path)
              else getattr(args, key) for key in RUN_ARGUMENTS}
    config.update(dates=dates, model=MODEL, reference=str(reference),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  comparison_config_sha256=hashlib.sha256((args.comparison/'run_config.json').read_bytes()).hexdigest(),
                  comparison_forecasts_sha256=hashlib.sha256((args.comparison/'forecasts.parquet').read_bytes()).hexdigest(),
                  input_manifest_sha256=hashlib.sha256((source/'input_manifest.json').read_bytes()).hexdigest(),
                  response='Observed-time clock-adjusted log levels; training-only centering and scaling; no differencing',
                  target='Same complete next-60/300-second arithmetic spread means as saved comparison')
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out/'run_config.json'
    if path.exists():
        saved = json.loads(path.read_text())
        config['source_history'] = saved.get('source_history', [])
        if saved != config:
            left, right = dict(saved), dict(config)
            old_source, new_source = left.pop('source_sha256'), right.pop('source_sha256')
            left.pop('source_history', None)
            right.pop('source_history', None)
            if left != right or old_source != PREVIOUS_SOURCE_SHA256:
                raise ValueError('Use a new output directory after configuration/source changes')
            archive = args.out/'attempts/full_comparison_support'
            archive.mkdir(parents=True, exist_ok=True)
            write_json(archive/'run_config.json', saved)
            config['source_history'].append(dict(previous=old_source, current=new_source,
                reason='Original full-comparison input and optional benchmark summaries; fitting, seeds and forecasts unchanged'))
    else:
        config['source_history'] = []
    write_json(path, config)
    tasks.to_csv(args.out/'forecast_tasks.csv', index=False)
    remaining = [date for date in dates if not (args.out/'days'/date/'completed.json').exists()]
    print(f'{len(dates)} fixed comparison dates; {len(remaining)} ARIMA(1,0,1) fits to run', flush=True)
    started = time.monotonic()
    failures = {}
    if remaining:
        import ray
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        count = min(args.ray_actors, len(remaining), cpus)
        ray.init(num_cpus=cpus, include_dashboard=False, log_to_driver=False,
                 object_store_memory=512*1024**2, _temp_dir='/tmp/ray-csi300-arima101-'+str(args.seed))
        actors = [ray.remote(num_cpus=1)(DayActor).remote() for _ in range(count)]
        jobs, pending = iter(remaining), {}

        def submit(actor, date):
            records = tasks[tasks.date.eq(date)].to_dict('records')
            pending[actor.process.remote(reference, source, args.out, date, records, config)] = actor, date

        for actor in actors:
            submit(actor, next(jobs))
        with tqdm(total=len(dates), initial=len(dates)-len(remaining), desc='ARIMA(1,0,1) dates (Ray)') as progress:
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                actor, date = pending.pop(ready[0])
                try:
                    result = ray.get(ready[0])
                    progress.set_postfix(date=date, phi=f"{result['phi']:.6f}")
                except Exception as exc:
                    failures[date] = str(exc)
                    print(f'{date}: {exc}', flush=True)
                progress.update(1)
                following = next(jobs, None)
                if following:
                    submit(actor, following)
        ray.shutdown()
    write_json(args.out/'failures.json', failures)
    if failures:
        raise RuntimeError(f'{len(failures)} failed dates; completed dates can be resumed')
    summarize(args.comparison, args.out, dates, config)
    write_json(args.out/'execution.json', dict(dates=len(dates), newly_run=len(remaining),
                                              elapsed_seconds=time.monotonic()-started))
    print(f'Finished: {args.out}', flush=True)


if __name__ == '__main__':
    main()
