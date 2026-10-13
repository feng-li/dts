#!/usr/bin/env python3
"""Audit a causal daily rolling-origin boundary forecast experiment.

The training window expands from 2018; frequency shards are constructed from
the complete transformed training series. This script plans tasks, not fits.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# End is an array boundary: end-1 is known, end is the first future second.
# These times are selected by the trading clock, not by realized costs.
ORIGINS = [
    ('opening', '09:31', 'am', 60), ('opening', '09:35', 'am', 300),
    ('pre_lunch', '11:25', 'am', 6900), ('pre_lunch', '11:29', 'am', 7140),
    ('reopening', '13:01', 'pm', 60), ('reopening', '13:05', 'pm', 300),
    ('end_of_grid', '14:52', 'pm', 6720), ('end_of_grid', '14:56', 'pm', 6960),
]
PHASES = {'am': (0, 7200), 'pm': (7200, 14220)}


def audit_tasks(raw, dates, daily_changes, minimum=10_000_000, progress=False):
    """Training counts exclude current/future dates and artificial boundaries."""
    if raw.shape != (len(dates), 14220) or len(daily_changes) != len(dates):
        raise ValueError('Input must match the original day/session grid')
    prior = np.r_[0, np.cumsum(daily_changes, dtype=np.int64)[:-1]]
    even_prior = prior - prior % 2
    tasks = []
    iterator = tqdm(range(len(dates)), desc='Audit rolling forecast dates', disable=not progress)
    for day in iterator:
        for event, clock, phase, elapsed in ORIGINS:
            lo, hi = PHASES[phase]
            end = lo + elapsed
            context = min(600, elapsed)
            history = raw[day, end-context:end]
            history_ok = bool(np.isfinite(history).all() and (history > 0).all())
            for horizon in [60, 300]:
                if end + horizon > hi:
                    continue
                target = raw[day, end:end+horizon]
                target_ok = bool(np.isfinite(target).all() and (target > 0).all())
                train_ok = bool(even_prior[day] >= minimum)
                issued = train_ok and history_ok
                tasks.append(dict(day_index=day, date=str(dates[day]), event=event,
                    origin=clock, phase=phase, elapsed=elapsed, horizon=horizon,
                    context_levels=context, context_changes=context-1,
                    train_first=str(dates[0]), train_last=str(dates[day-1]) if day else None,
                    train_stop_day=day, train_changes=int(prior[day]),
                    train_fft_observations=int(even_prior[day]), minimum_training_met=train_ok,
                    history_ok=history_ok, target_ok=target_ok, issued=issued,
                    scorable=issued and target_ok,
                    status=('short_training' if not train_ok else
                            'missing_history' if not history_ok else
                            'missing_target' if not target_ok else 'scorable')))
    return pd.DataFrame(tasks)


def select_dates(tasks, count=1000, policy='usable'):
    if count < 1:
        raise ValueError('Evaluation date count must be positive')
    # Selection uses training size and observable conditioning data, never
    # future targets. Missing future outcomes remain explicit tasks.
    if policy == 'usable':
        pool = tasks.groupby('date').issued.any()
        choices = pool.index[pool].to_list()
    elif policy == 'archived':
        choices = sorted(tasks.date.unique())
    else:
        raise ValueError('Unknown date policy')
    if len(choices) < count:
        raise ValueError(f'Only {len(choices)} {policy} dates available; requested {count}')
    selected = choices[-count:]
    return tasks[tasks.date.isin(selected)].copy().reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home() / 'running/dts/artifacts/liquidity'
    parser.add_argument('--input', type=Path, default=base / 'csi300_dc_1m')
    parser.add_argument('--out', type=Path, default=base / 'csi300_boundary_rolling_plan')
    parser.add_argument('--evaluation-days', type=int, default=1000)
    parser.add_argument('--date-policy', choices=['usable', 'archived'], default='usable')
    parser.add_argument('--minimum-training', type=int, default=10_000_000)
    args = parser.parse_args()
    if args.minimum_training < 10_000_000:
        parser.error('This experiment requires at least 10 million training observations')
    manifest_path = args.input / 'input_manifest.json'
    meta = json.loads(manifest_path.read_text())
    raw = np.load(args.input / 'spread.npy', mmap_mode='r')
    runs = pd.read_csv(args.input / 'contiguous_runs.csv')
    if meta['size_spec'] != 'notional_1e+06':
        raise ValueError('Expected the previously defined RMB 1m diagnostic')
    daily = np.bincount(runs.day_index, weights=runs.n-1, minlength=len(meta['dates'])).astype(np.int64)
    tasks = audit_tasks(raw, meta['dates'], daily, args.minimum_training, progress=True)
    selected = select_dates(tasks, args.evaluation_days, args.date_policy)
    fits = selected.groupby('date').agg(day_index=('day_index', 'first'),
        train_first=('train_first', 'first'), train_last=('train_last', 'first'),
        train_stop_day=('train_stop_day', 'first'), train_changes=('train_changes', 'first'),
        train_fft_observations=('train_fft_observations', 'first'),
        training_ok=('minimum_training_met', 'first'), issued_tasks=('issued', 'sum'),
        scorable_tasks=('scorable', 'sum')).reset_index()
    fits['fit_required'] = fits.training_ok & (fits.issued_tasks > 0)
    events = selected.groupby(['event', 'origin', 'horizon']).agg(
        scheduled=('date', 'size'), issued=('issued', 'sum'), scorable=('scorable', 'sum')).reset_index()
    summary = dict(date_policy=args.date_policy, evaluation_dates=len(fits),
        evaluation_first=selected.date.min(), evaluation_last=selected.date.max(),
        archive_dates_in_span=int(selected.day_index.max()-selected.day_index.min()+1),
        daily_fits=int(fits.fit_required.sum()), scheduled_tasks=len(selected),
        issued_tasks=int(selected.issued.sum()), scorable_tasks=int(selected.scorable.sum()),
        missing_history_tasks=int((selected.status == 'missing_history').sum()),
        missing_target_tasks=int((selected.status == 'missing_target').sum()),
        short_training_tasks=int((selected.status == 'short_training').sum()),
        min_training_fft_observations=int(fits.train_fft_observations.min()),
        max_training_fft_observations=int(fits.train_fft_observations.max()),
        counts_are='Plans and data-readiness counts, not completed forecasts')
    protocol = dict(source=str(args.input.resolve()),
        input_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evaluation_days=args.evaluation_days, date_policy=args.date_policy,
        minimum_training=args.minimum_training,
        training='Daily expanding history: every valid within-session adjusted log change before the forecast date; even T',
        refit='One daily fit shared by all origins and horizons on that date',
        dc='Full transformed-series FFT, systematic frequency partitions, prior power 1/G, fixed-Hessian RWMH, joint covariance-weighted consensus',
        model='ARTFIMA(0,d,lambda,1) on adjusted log changes; manuscript native priors',
        calibration='Clock profile and scale fitted on training dates only and frozen over the forecast date',
        clock_bins='60 seconds in first/last 10 minutes of each phase; 300 seconds elsewhere; training-only whole-minute profile',
        conditioning='min(600, elapsed-session-seconds) contiguous positive levels, identical across models',
        targets='Arithmetic mean of next 60/300 raw spread seconds; complete horizon within same phase',
        comparisons=['SpectralDC_ARTFIMA', 'Matched_ARIMA011', 'LastValue', 'ClockAdjustedLastValue'],
        missingness='Task issuance depends on history; target missingness is recorded separately and never imputed',
        selection='Usable dates have at least one issuable task; selection does not require an observed future outcome',
        reporting='Separate scores by event, origin and horizon; paired day-block intervals; origins and dates are dependent',
        caveats='Observed-time concatenation and residual stationarity require validation; approximate historical weights and available-constituent selection remain; no observations after 14:57',
        execution_status='Planned only; no new posterior or forecast estimates in this output')
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Use a new output directory after changing the source or plan')
    path.write_text(json.dumps(protocol, indent=2) + '\n')
    selected.to_csv(args.out / 'forecast_tasks.csv', index=False)
    fits.to_csv(args.out / 'daily_fits.csv', index=False)
    events.to_csv(args.out / 'event_counts.csv', index=False)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(events.to_string(index=False))
    print(args.out)


if __name__ == '__main__':
    main()
