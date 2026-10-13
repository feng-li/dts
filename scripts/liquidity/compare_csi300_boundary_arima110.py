#!/usr/bin/env python3
"""Add an ARIMA(1,1,0) benchmark to saved daily liquidity forecasts.

AR(1) is fitted to the same standardized changes using the full positive-
frequency Whittle posterior. Its cumulative transformed series is ARIMA(1,1,0).
Existing Spectral DC fits and forecasts are read without modification.
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
from scipy.optimize import minimize
from tqdm import tqdm
from scripts.liquidity.compare_csi300_boundary_random_days import (
    MODELS, prediction_record, weighted_variance_diagonal, write_json)
from scripts.liquidity.compare_csi300_forecasts import kernel

MODEL = 'Matched_ARIMA110_MAP'
RUN_ARGUMENTS = ['reference', 'out', 'days', 'seed', 'ray_actors', 'paths']


def ar1_objective(parameters, n, sum_pg, sum_q_pg):
    """Exact Whittle posterior, gradient and Hessian from spectral sums.

For even n, the interior positive-frequency log denominator sums to
log(1-phi**n) - log(1-phi**2). Every periodogram ordinate contributes to
sum_pg and sum_q_pg; this identity introduces no frequency approximation.
Priors match the existing MA benchmark: uniform coefficient and N(0,1)
log innovation variance. The returned quantities are not divided by n.
"""
    phi, log_variance = parameters
    if n < 4 or n % 2 or abs(phi) >= 1:
        raise ValueError('Requires even n >= 4 and a stationary AR coefficient')
    m = n // 2 - 1
    root = phi**n
    log_product = np.log1p(-root) - np.log1p(-phi*phi)
    first = -n*phi**(n-1)/(1-root) + 2*phi/(1-phi*phi)
    second = (-n*(n-1)*phi**(n-2)/(1-root)
              - n*n*phi**(2*n-2)/(1-root)**2
              + 2*(1+phi*phi)/(1-phi*phi)**2)
    energy = (1-phi)**2*sum_pg + phi*sum_q_pg
    energy_first = 2*(phi-1)*sum_pg + sum_q_pg
    multiplier = 2*np.pi*np.exp(-log_variance)
    value = (m*(log_variance-np.log(2*np.pi)) - log_product
             + multiplier*energy + .5*log_variance**2
             + np.log(2) + .5*np.log(2*np.pi))
    gradient = np.array([-first+multiplier*energy_first,
                         m-multiplier*energy+log_variance])
    hessian = np.array([[2*multiplier*sum_pg-second, -multiplier*energy_first],
                        [-multiplier*energy_first, multiplier*energy+1]])
    return value, gradient, hessian


def fit_ar1(q, pg, n):
    if len(q) != n//2-1 or len(pg) != len(q):
        raise ValueError('Cached spectrum does not match the full training length')
    sum_pg = float(np.sum(pg, dtype=np.float64))
    sum_q_pg = float(np.dot(q, pg))
    if not np.isfinite([sum_pg, sum_q_pg]).all() or sum_pg <= 0:
        raise ValueError('Invalid training spectrum')
    phi = float(np.clip(1-sum_q_pg/(2*sum_pg), -.99, .99))
    energy = (1-phi)**2*sum_pg + phi*sum_q_pg
    initial = np.array([phi, np.log(2*np.pi*energy/len(pg))])

    def objective(parameters):
        value, gradient, _ = ar1_objective(parameters, n, sum_pg, sum_q_pg)
        return value/len(pg), gradient/len(pg)

    fit = minimize(objective, initial, jac=True, method='L-BFGS-B',
                   bounds=[(-.99999999, .99999999), (-30., 20.)],
                   options={'ftol': 1e-14, 'gtol': 1e-12, 'maxiter': 500})
    value, gradient, hessian = ar1_objective(fit.x, n, sum_pg, sum_q_pg)
    eigenvalues = np.linalg.eigvalsh(hessian)
    distance = (float(np.sqrt(max(0., gradient@np.linalg.solve(hessian, gradient))))
                if eigenvalues.min() > 0 else float('inf'))
    if not np.isfinite(value) or distance > .01:
        raise ValueError(f'Unresolved AR(1) MAP: distance={distance}, eigenvalues={eigenvalues}')
    return dict(phi=float(fit.x[0]), log_sigma2=float(fit.x[1]),
                sigma2=float(np.exp(fit.x[1])), n=int(n),
                sum_pg=sum_pg, sum_q_pg=sum_q_pg,
                optimizer_success=bool(fit.success), message=str(fit.message),
                newton_distance=distance, hessian_eigenvalues=eigenvalues.tolist(),
                covariance=np.linalg.inv(hessian).tolist())


def run_day(reference, source, out, date, records, config):
    started = time.monotonic()
    folder = reference/'days'/date
    destination = out/'days'/date
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/'completed.json').exists():
        return json.loads((destination/'completed.json').read_text())
    saved = json.loads((folder/'fit.json').read_text())
    tasks = pd.DataFrame(records)
    day_index = int(tasks.day_index.iloc[0])
    if (saved['n'] < 10_000_000 or saved['n'] != int(tasks.train_fft_observations.iloc[0])
            or saved['train_stop_day'] != day_index or str(saved['train_last']) >= date):
        raise ValueError('Training size or strictly earlier cutoff changed')
    fit_path = destination/'fit.json'
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
    else:
        fit = fit_ar1(np.load(folder/'q.npy', mmap_mode='r'),
                      np.load(folder/'periodogram.npy', mmap_mode='r'), saved['n'])
        fit.update(date=date, train_first=saved['train_first'], train_last=saved['train_last'],
                   train_stop_day=day_index)
        write_json(fit_path, fit)
    raw = np.load(source/'spread.npy', mmap_mode='r')[day_index]
    profile = np.load(folder/'profile.npy')
    clock_mean = np.load(folder/'clock_mean.npy')
    clock_scale = saved['scale']*np.load(folder/'clock_scale.npy')
    weights, covariance, _ = kernel('AR1', fit, 1, 300)
    chol = np.linalg.cholesky(covariance+np.eye(300)*1e-12)
    rows = []
    for origin, group in tasks[tasks.issued].groupby('origin', sort=True):
        row = group.iloc[0]
        end = (0 if row.phase == 'am' else 7200)+int(row.elapsed)
        context = int(row.context_levels)
        history_raw = raw[end-context:end]
        if not (np.isfinite(history_raw).all() and (history_raw > 0).all()):
            raise ValueError('Reference conditioning history is no longer available')
        history = (np.diff(np.log(history_raw))-clock_mean[end-context+1:end])/clock_scale[end-context+1:end]
        horizon = int(group.horizon.max())
        scale = clock_scale[end:end+horizon]
        mean = weights[:horizon, -1]*history[-1]
        logmean = (np.log(raw[end-1])+profile[end:end+horizon]-profile[end-1]
                   + np.cumsum(scale*mean))
        variance = weighted_variance_diagonal(covariance[:horizon, :horizon], scale)
        point = np.exp(logmean+.5*variance)
        rng = np.random.default_rng(config['seed']+day_index*100_000+end)
        half = rng.normal(size=(config['paths']//2, horizon))
        normal = np.r_[half, -half]
        paths = np.exp(logmean+np.cumsum(scale*(normal@chol[:horizon, :horizon].T), axis=1))
        for task in group.itertuples():
            h = int(task.horizon)
            target = raw[end:end+h]
            target_ok = bool(np.isfinite(target).all() and (target > 0).all())
            if target_ok != bool(task.target_ok):
                raise ValueError('Reference target availability changed')
            actual = float(target.mean()) if target_ok else np.nan
            rows.append(dict(date=date, day_index=day_index, event=task.event, origin=origin,
                             horizon=h, context_levels=context, model=MODEL,
                             train_fft_observations=fit['n'], target_ok=target_ok,
                             **prediction_record(actual, float(point[:h].mean()), paths[:, :h].mean(axis=1))))
    result = pd.DataFrame(rows)
    existing = pd.read_parquet(folder/'forecasts.parquet')
    existing['date'] = existing.date.astype(str)
    keys = ['date', 'origin', 'horizon']
    paired = result.merge(existing[existing.model.eq(MODELS[0])], on=keys,
                          suffixes=('_new', '_reference'), validate='one_to_one')
    if len(paired) != len(result):
        raise ValueError('Reference forecast origins changed')
    np.testing.assert_array_equal(paired.actual_new, paired.actual_reference)
    result.to_parquet(destination/'forecasts.parquet', index=False)
    completed = dict(date=date, n=fit['n'], phi=fit['phi'], sigma2=fit['sigma2'],
                     newton_distance=fit['newton_distance'], rows=len(result),
                     scored=int(np.isfinite(result.actual).sum()), seconds=time.monotonic()-started)
    write_json(destination/'completed.json', completed)
    return completed


class DayActor:
    def process(self, reference, source, out, date, records, config):
        return run_day(reference, source, out, date, records, config)


def summarize(reference, out, dates, config):
    frames = []
    for date in dates:
        for path in [reference/'days'/date/'forecasts.parquet', out/'days'/date/'forecasts.parquet']:
            frame = pd.read_parquet(path)
            frame['date'] = frame.date.astype(str)
            frames.append(frame)
    all_forecasts = pd.concat(frames, ignore_index=True)
    keys = ['date', 'origin', 'horizon']
    if (all_forecasts.duplicated(keys+['model']).any()
            or not all_forecasts.groupby(keys).model.nunique().eq(len(MODELS)+1).all()):
        raise ValueError('Unmatched model/origin pairs')
    scored = all_forecasts[np.isfinite(all_forecasts.actual)].copy()
    if not scored.groupby(keys).actual.nunique().eq(1).all():
        raise ValueError('Different target outcomes across models')
    all_forecasts['error'] = all_forecasts.prediction-all_forecasts.actual
    scored['error'] = scored.prediction-scored.actual
    scored['width95'] = scored.upper95-scored.lower95

    def scores(groups):
        result = groups.agg(n=('se', 'size'), days=('date', 'nunique'),
                            mse=('se', 'mean'), mae=('ae', 'mean'), bias=('error', 'mean'),
                            crps=('crps', 'mean'), coverage95=('cover95', 'mean'),
                            width95=('width95', 'mean')).reset_index()
        result['rmse'] = np.sqrt(result.pop('mse'))
        return result

    total = scores(scored.groupby(['model', 'horizon']))
    event = scores(scored.groupby(['event', 'origin', 'model', 'horizon']))
    rng = np.random.default_rng(config['seed']+11)
    comparisons = []
    for horizon, group in scored.groupby('horizon'):
        wide = group.pivot(index=['day_index', 'origin'], columns='model', values='se')
        for candidate, baseline in [(MODELS[0], MODEL), (MODEL, MODELS[1])]:
            paired = wide[[candidate, baseline]].copy()
            paired['cluster'] = paired.index.get_level_values('day_index')//20
            cluster = paired.groupby('cluster')
            values = cluster[[candidate, baseline]].sum().to_numpy()
            counts = cluster.size().to_numpy()
            indices = rng.integers(len(values), size=(2000, len(values)))
            mse = values[indices].sum(axis=1)/counts[indices].sum(axis=1)[:, None]
            gains = 100*(1-np.sqrt(mse[:, 0]/mse[:, 1]))
            comparisons.append(dict(horizon=int(horizon), candidate=candidate, baseline=baseline,
                                    rmse_reduction_pct=100*(1-np.sqrt(paired[candidate].mean()/paired[baseline].mean())),
                                    lower_pct=float(np.quantile(gains, .025)),
                                    upper_pct=float(np.quantile(gains, .975)), clusters=len(values)))
    comparisons = pd.DataFrame(comparisons)
    all_forecasts.to_parquet(out/'forecasts.parquet', index=False)
    total.to_csv(out/'scores.csv', index=False)
    event.to_csv(out/'event_scores.csv', index=False)
    comparisons.to_csv(out/'paired_rmse.csv', index=False)
    diagnostics = pd.DataFrame([json.loads((out/'days'/date/'completed.json').read_text()) for date in dates])
    diagnostics.to_csv(out/'daily_diagnostics.csv', index=False)
    report = ['ARIMA(1,1,0) quick benchmark on saved liquidity boundary forecasts',
              'AR(1) on the identical adjusted standardized changes; ARIMA(1,1,0) for their cumulative sum.',
              'Full positive-frequency Whittle MAP, uniform AR coefficient and normal log-variance prior; no drift.',
              'Same historical training, clock adjustment, conditioning windows, origins and complete outcomes.',
              'Existing Spectral DC forecasts are reused; no new DC MCMC or likelihood modification.',
              'ARIMA uses point-estimated parameters; ARTFIMA integrates parameter uncertainty.',
              'Intervals use antithetic Gaussian conditional paths; point forecasts integrate lognormal means.',
              'Exploratory random-date test; bootstrap uses 20-archive-date clusters among sampled dates.',
              'Historical approximate weights, observed-time concatenation and original DC accuracy limitations remain.',
              '', total.to_string(index=False), '', comparisons.to_string(index=False)]
    (out/'REPORT.txt').write_text('\n'.join(report)+'\n')
    print(total.to_string(index=False), flush=True)
    print(comparisons.to_string(index=False), flush=True)


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
    parser.add_argument('--reference', type=Path, default=base/'csi300_boundary_full1000_v1')
    parser.add_argument('--out', type=Path, default=base/'csi300_boundary_arima110_quick100_v1')
    parser.add_argument('--days', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--ray-actors', type=int, default=50)
    parser.add_argument('--paths', type=int, default=512)
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.days < 1 or args.ray_actors < 1 or args.paths < 2 or args.paths % 2:
        parser.error('Positive days/actors and an even path count >= 2 are required')
    return args


def main():
    args = parse_arguments()
    source_config = json.loads((args.reference/'run_config.json').read_text())
    source = Path(source_config['input'])
    tasks = pd.read_csv(args.reference/'forecast_tasks.csv', dtype={'date': str})
    choices = np.array(sorted(tasks.date.unique()))
    if args.days > len(choices):
        raise ValueError('Requested more dates than the saved experiment contains')
    dates = np.sort(np.random.default_rng(args.seed).choice(choices, args.days, replace=False)).tolist()
    config = {key: str(getattr(args, key)) if isinstance(getattr(args, key), Path)
              else getattr(args, key) for key in RUN_ARGUMENTS}
    config.update(dates=dates, model=MODEL,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference_protocol_sha256=hashlib.sha256((args.reference/'protocol.json').read_bytes()).hexdigest(),
                  input_manifest_sha256=hashlib.sha256((source/'input_manifest.json').read_bytes()).hexdigest(),
                  target='Next 60/300 raw spread seconds, arithmetic mean; same complete targets as reference',
                  inference='Full positive-frequency Whittle MAP on saved standardized changes; no drift',
                  prior='Uniform stationary AR coefficient; N(0,1) log innovation variance')
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Use a new output directory after configuration/source changes')
    write_json(config_path, config)
    selected = tasks[tasks.date.isin(dates)]
    selected.to_csv(args.out/'forecast_tasks.csv', index=False)
    started = time.monotonic()
    remaining = [date for date in dates if not (args.out/'days'/date/'completed.json').exists()]
    print(f'{len(dates)} sampled dates; {len(remaining)} to fit; reuse original Spectral DC forecasts', flush=True)
    failures = {}
    if remaining:
        import ray
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
        count = min(args.ray_actors, len(remaining), cpus)
        ray.init(num_cpus=cpus, include_dashboard=False, log_to_driver=False,
                 object_store_memory=512*1024**2, _temp_dir='/tmp/ray-csi300-arima110-'+str(args.seed))
        actors = [ray.remote(num_cpus=1)(DayActor).remote() for _ in range(count)]
        jobs = iter(remaining)
        pending = {}

        def submit(actor, date):
            records = selected[selected.date.eq(date)].to_dict('records')
            pending[actor.process.remote(args.reference, source, args.out, date, records, config)] = (actor, date)

        for actor in actors:
            submit(actor, next(jobs))
        with tqdm(total=len(dates), initial=len(dates)-len(remaining), desc='ARIMA(1,1,0) dates (Ray)') as progress:
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                actor, date = pending.pop(ready[0])
                try:
                    result = ray.get(ready[0])
                    progress.set_postfix(date=date, phi=f"{result['phi']:.3f}")
                except Exception as exc:
                    failures[date] = str(exc)
                    print(f'{date}: {exc}', flush=True)
                progress.update(1)
                next_date = next(jobs, None)
                if next_date:
                    submit(actor, next_date)
        ray.shutdown()
    write_json(args.out/'failures.json', failures)
    if failures:
        raise RuntimeError(f'{len(failures)} failed dates; successful dates can be resumed')
    summarize(args.reference, args.out, dates, config)
    write_json(args.out/'execution.json', dict(dates=len(dates), newly_run=len(remaining),
                                              elapsed_seconds=time.monotonic()-started))
    print(f'Finished: {args.out}', flush=True)


if __name__ == '__main__':
    main()
