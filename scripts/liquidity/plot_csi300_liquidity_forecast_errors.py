#!/usr/bin/env python3
"""Plot paired RMSE gains over matched ARIMA by forecast time and horizon."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'running/dts/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


EVENTS = [('opening', 'Opening', '#167c80'),
          ('pre_lunch', 'Pre-lunch', '#4c78a8'),
          ('reopening', 'Reopening', '#ce7930'),
          ('end_of_grid', 'Grid end', '#765ca2')]
MODELS = [('SpectralDC_ARTFIMA', 'Spectral DC ARTFIMA', '#167c80'),
          ('Matched_ARIMA011_MAP', 'Matched ARIMA', '#ce7930'),
          ('ClockLastValue', 'Clock-adjusted Naive', '#4c78a8'),
          ('LastValue', 'Naive', '#62748a')]
ORIGINS = ['09:31', '09:35', '11:25', '11:29', '13:01', '13:05', '14:52', '14:56']


def recent_forecasts(root, recent_start, manifest, dates):
    """Require completed forecasts and the same input archive as Figure 1."""
    protocol_path = root/'protocol.json'
    forecasts_path = root/'forecasts.parquet'
    protocol = json.loads(protocol_path.read_text())
    if protocol['input_manifest_sha256'] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError('Forecasts and session profiles use different input manifests')
    diagnostics = pd.read_csv(root/'daily_diagnostics.csv', dtype={'date': str})
    if (len(diagnostics) != protocol['days'] or
            set(diagnostics.date) != set(protocol['dates']) or
            not diagnostics.sampling_passed.eq(True).all()):
        raise ValueError('Expected a completed forecasting experiment with passed sampling checks')
    forecasts = pd.read_parquet(forecasts_path)
    forecasts['date'] = forecasts.date.astype(str)
    if set(forecasts.model) != {model[0] for model in MODELS}:
        raise ValueError('Expected the four matched models from the application')
    if forecasts.duplicated(['date', 'origin', 'horizon', 'model']).any():
        raise ValueError('Duplicate forecast targets')
    selected = forecasts.copy() if recent_start is None else forecasts[forecasts.date >= recent_start].copy()
    if not set(selected.date).issubset(set(protocol['dates'])):
        raise ValueError('Forecast dates are absent from the saved experiment protocol')
    if not np.array_equal(dates[selected.day_index.to_numpy(dtype=int)], selected.date.to_numpy()):
        raise ValueError('Forecast day indices do not match the spread archive')
    scored = selected[np.isfinite(selected.actual) & np.isfinite(selected.prediction)].copy()
    if scored.empty or not scored.target_ok.eq(True).all():
        raise ValueError('Expected complete observed targets')
    if not scored.horizon.isin([60, 300]).all() or set(scored.horizon) != {60, 300}:
        raise ValueError('Expected one- and five-minute forecast targets')
    if not scored.event.isin([event[0] for event in EVENTS]).all():
        raise ValueError('Unexpected trading boundary')
    if not scored.origin.isin(ORIGINS).all():
        raise ValueError('Unexpected forecast origin')
    groups = scored.groupby(['date', 'origin', 'horizon'])
    if not groups.model.nunique().eq(len(MODELS)).all() or not groups.actual.nunique().eq(1).all():
        raise ValueError('Models do not share identical complete forecast targets')
    if (scored[['actual', 'prediction']].to_numpy() <= 0).any():
        raise ValueError('Nonpositive observed or predicted spread')
    scored = scored.sort_values(['date', 'origin', 'horizon', 'model']).reset_index(drop=True)
    scored['error'] = scored.prediction-scored.actual
    targets = scored[scored.model == MODELS[0][0]]
    counts = [dict(horizon_seconds=int(h), targets_per_model=len(g), dates=int(g.date.nunique()),
                   first=str(g.date.min()), last=str(g.date.max()))
              for h, g in targets.groupby('horizon')]
    metadata = dict(root=str(root.resolve()), models=[model[0] for model in MODELS], samples=counts,
                    input_manifest_sha256=protocol['input_manifest_sha256'],
                    forecasts_sha256=hashlib.sha256(forecasts_path.read_bytes()).hexdigest(),
                    protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                    target='Arithmetic average of the next 60 or 300 raw one-second spreads, in bps',
                    error='Saved point prediction minus observed horizon average, in bps',
                    recent_start=recent_start,
                    date_selection='All saved test dates unless --recent-start is supplied; no refitting',
                    daily_coverage='No additional whole-day coverage filter; every target is complete',
                    excluded_incomplete_targets=int(selected[['date', 'origin', 'horizon']].drop_duplicates().shape[0]
                                                    - scored[['date', 'origin', 'horizon']].drop_duplicates().shape[0]),
                    rendering='Vector RMSE-reduction markers, paired bootstrap intervals, text and axes',
                    clock='Eight chronological forecast origins at equal spacing; not elapsed time',
                    sampling_checks='Saved daily sampling gates passed; this plot does not establish posterior accuracy',
                    groups=protocol['groups'],
                    inference_source='Existing Table 3 forecast archive; not the subsequently repaired G=20 fits')
    return scored, metadata


def errors_by_origin(forecasts):
    rows = []
    for model, _, _ in MODELS:
        for horizon in [60, 300]:
            for origin in ORIGINS:
                errors = forecasts.loc[(forecasts.model == model) &
                                       (forecasts.horizon == horizon) &
                                       (forecasts.origin == origin), 'error'].to_numpy()
                rows.append(dict(model=model, horizon=horizon, origin=origin, n=len(errors),
                                 mean_error=float(errors.mean()) if len(errors) else np.nan,
                                 mae=float(np.abs(errors).mean()) if len(errors) else np.nan,
                                 rmse=float(np.sqrt(np.mean(errors**2))) if len(errors) else np.nan))
    return pd.DataFrame(rows)


def forecast_interval_label(origin, horizon):
    hour, minute = map(int, origin.split(':'))
    endpoint = 60*hour+minute+horizon//60
    return f'{origin}-{endpoint//60:02d}:{endpoint%60:02d}'


def _summarize_rmse_bootstrap(row, sums, weights):
    """Use the same paired resamples in serial execution and Ray actors."""
    sampled = weights@sums
    valid = sampled[:, 1] > 0
    gains = 100*(1-np.sqrt(sampled[valid, 0]/sampled[valid, 1]))
    if len(gains) < .99*len(weights) or sums[:, 1].sum() <= 0:
        raise ValueError(f'Insufficient positive benchmark errors: {row["horizon"]}, {row["origin"]}')
    lo, hi = np.quantile(gains, [.025, .975])
    totals = sums.sum(axis=0)
    row.update(rmse_candidate=float(np.sqrt(totals[0]/row['n'])),
               rmse_benchmark=float(np.sqrt(totals[1]/row['n'])),
               rmse_reduction_pct=float(100*(1-np.sqrt(totals[0]/totals[1]))),
               lower_pct=float(lo), upper_pct=float(hi),
               excludes_zero=bool(lo > 0 or hi < 0), bootstrap_valid=len(gains))
    return row


class RMSEBootstrapActor:
    def __init__(self, weights):
        self.weights = weights

    def summarize(self, row, sums):
        return _summarize_rmse_bootstrap(row, sums, self.weights)


def _bootstrap_with_ray(jobs, weights, ray_actors):
    import ray
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
    workers = min(ray_actors, len(jobs), available)
    owns_runtime = not ray.is_initialized()
    actors = []
    try:
        if owns_runtime:
            ray.init(num_cpus=workers, include_dashboard=False,
                     runtime_env={'env_vars': {key: '1' for key in
                                  ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']}})
        else:
            workers = min(workers, int(ray.cluster_resources().get('CPU', 0)))
            if workers < 1:
                raise ValueError('The existing Ray runtime has no CPU resources')
        shared_weights = ray.put(weights)
        actor_type = ray.remote(num_cpus=1)(RMSEBootstrapActor)
        for _ in range(workers):
            actors.append(actor_type.remote(shared_weights))
        pending = {actors[i % workers].summarize.remote(row, sums): index
                   for i, (index, row, sums) in enumerate(jobs)}
        results = {}
        with tqdm(total=len(jobs), desc='Paired RMSE bootstrap (Ray)', unit='comparison') as progress:
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                for ref in ready:
                    results[pending.pop(ref)] = ray.get(ref)
                    progress.update(1)
        return results
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        if owns_runtime:
            ray.shutdown()


def paired_rmse_gains(forecasts, block_days=20, replicates=10000, seed=20261014, ray_actors=1):
    """Paired percentile intervals using the archive-date clusters of Table 3.

    Cluster boundaries use the original day_index, not the retained-row rank.
    One set of cluster multiplicities is shared by every time and horizon.
    Missing targets remain missing; both methods always use the same outcomes.
    """
    if block_days < 1 or replicates < 2:
        raise ValueError('Positive block length and at least two replicates required')
    if ray_actors < 1:
        raise ValueError('ray_actors must be at least 1')
    candidate, benchmark = MODELS[0][0], MODELS[1][0]
    paired = forecasts[forecasts.model.isin([candidate, benchmark])].copy()
    paired['squared_error'] = paired.error**2
    wide = paired.pivot(index=['day_index', 'date', 'origin', 'horizon'],
                        columns='model', values='squared_error')
    if wide.empty or not np.isfinite(wide[[candidate, benchmark]]).all().all():
        raise ValueError('Expected finite paired model errors')
    wide = wide.reset_index()
    wide['cluster'] = wide.day_index//block_days
    clusters = np.sort(wide.cluster.unique())
    if len(clusters) < 2:
        raise ValueError('At least two archive-date clusters required for uncertainty')
    rng = np.random.default_rng(seed)
    # Multiplicities avoid a large replicates x clusters x targets temporary.
    draws = rng.integers(len(clusters), size=(replicates, len(clusters)))
    weights = np.zeros((replicates, len(clusters)), dtype=np.int64)
    np.add.at(weights, (np.arange(replicates)[:, None], draws), 1)
    rows, jobs = [], []
    groups = [(h, o) for h in [60, 300] for o in ORIGINS+['Pooled']]
    for index, (horizon, origin) in enumerate(groups):
        g = wide[wide.horizon == horizon]
        if origin != 'Pooled':
            g = g[g.origin == origin]
        row = dict(horizon=horizon, origin=origin, n=len(g), dates=int(g.date.nunique()),
                   clusters=int(g.cluster.nunique()))
        rows.append(row)
        if not g.empty:
            sums = g.groupby('cluster')[[candidate, benchmark]].sum().reindex(clusters, fill_value=0)
            jobs.append((index, row, sums.to_numpy()))
    if ray_actors == 1:
        for index, row, sums in tqdm(jobs, desc='Paired RMSE bootstrap', unit='comparison'):
            rows[index] = _summarize_rmse_bootstrap(row, sums, weights)
    else:
        for index, row in _bootstrap_with_ray(jobs, weights, ray_actors).items():
            rows[index] = row
    return pd.DataFrame(rows)


def draw(gains):
    plt.rcParams.update({'font.size': 11, 'axes.labelsize': 11,
                         'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
                         'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral'],
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(6.14, 4.8), sharey=True)
    finite = gains[gains.origin != 'Pooled'].dropna(subset=['lower_pct', 'upper_pct'])
    lower, upper = min(0, finite.lower_pct.min()), max(0, finite.upper_pct.max())
    pad = max(.8, (upper-lower)*.1)
    for ax, horizon, letter in zip(axes, [60, 300], ['a', 'b']):
        data = gains[(gains.horizon == horizon) & (gains.origin != 'Pooled')].set_index('origin').loc[ORIGINS]
        available = data.n.to_numpy() > 0
        x = np.arange(len(ORIGINS))[available]
        data = data[available]
        color = '#167c80'
        # Draw interval endpoints directly: percentile intervals need not contain
        # the original estimate, so asymmetric errorbar lengths can be negative.
        ax.vlines(x, data.lower_pct, data.upper_pct, color=color, lw=1.5, zorder=3)
        for endpoint in [data.lower_pct, data.upper_pct]:
            ax.hlines(endpoint, x-.08, x+.08, color=color, lw=1.5, zorder=3)
        ax.scatter(x, data.rmse_reduction_pct, s=35, color=color, edgecolors='white', lw=.6, zorder=4)
        ax.axhline(0, color='#777777', ls='--', lw=.8, zorder=1)
        ax.set_ylim(lower-pad, upper+pad)
        ax.set_xlim(-.5, len(ORIGINS)-.5)
        origins = np.asarray(ORIGINS)[available]
        # Two-line interval labels keep five-minute endpoints legible at 11 pt.
        labels = [forecast_interval_label(o, horizon).replace('-', '-\n') for o in origins] if horizon == 300 else origins
        ax.set_xticks(x, labels)
        ax.set_title(f'({letter}) {horizon//60}-minute average', loc='left', fontsize=11, pad=7)
        ax.set_ylabel('RMSE reduction (%)')
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.12, right=.98, top=.94, bottom=.13, hspace=.55)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home()/'running/dts'
    parser.add_argument('--input', type=Path, default=base/'artifacts/liquidity/csi300_dc_1m')
    parser.add_argument('--root', type=Path,
                        default=base/'artifacts/liquidity/csi300_boundary_full1000_v1')
    parser.add_argument('--recent-start', help='Optional YYYYMMDD cutoff; default uses the complete test sample')
    parser.add_argument('--block-days', type=int, default=20)
    parser.add_argument('--bootstrap-replicates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--ray-actors', type=int, default=1,
                        help='Bootstrap actors; 1 runs serially, larger values use Ray (up to 16 comparisons)')
    parser.add_argument('--out', type=Path,
                        default=base/'artifacts/liquidity/csi300_liquidity_rmse_reduction_full1000')
    parser.add_argument('--pdf', type=Path,
                        default=Path(__file__).resolve().parents[2]/'docs/figures/csi300_liquidity_forecast_errors_intraday_full1000.pdf')
    args = parser.parse_args()
    if args.ray_actors < 1:
        parser.error('--ray-actors must be at least 1')
    manifest = args.input/'input_manifest.json'
    meta = json.loads(manifest.read_text())
    if meta['size_spec'] != 'notional_1e+06':
        raise ValueError('Expected the RMB 1m spread archive')
    forecasts, protocol = recent_forecasts(args.root, args.recent_start, manifest, np.asarray(meta['dates']))
    summaries = errors_by_origin(forecasts)
    gains = paired_rmse_gains(forecasts, args.block_days, args.bootstrap_replicates, args.seed,
                             ray_actors=args.ray_actors)
    protocol.update(metric='100 * (1 - RMSE_SpectralDC_ARTFIMA / RMSE_Matched_ARIMA011_MAP)',
                    benchmark='Matched ARIMA(0,1,1), full-data Whittle MAP',
                    positive_values='Lower RMSE for Spectral DC ARTFIMA',
                    bootstrap=dict(method='Paired nonoverlapping archive-date cluster percentile bootstrap',
                                   cluster='floor(day_index / block_days), preserving archive gaps',
                                   block_days=args.block_days, replicates=args.bootstrap_replicates,
                                   seed=args.seed, confidence=.95,
                                   pairing='Same cluster multiplicities for both methods and all origins/horizons',
                                   recomputation='Ratio of square roots of resampled sums of squared errors; matched counts cancel',
                                   scope='Pointwise, not simultaneous; conditional on saved fits and sample'),
                    missing_slots='Unavailable origins retain blank slots without ticks or labels',
                    five_minute_tick_labels='Origin to horizon endpoint; excludes origin second, includes endpoint',
                    panel_layout='One-minute reductions above five-minute reductions; common percentage scale',
                    typography='Times New Roman; 11 pt labels, 10.5 pt ticks at manuscript width',
                    ray_actors=args.ray_actors,
                    time_separators=False)
    args.out.mkdir(parents=True, exist_ok=True)
    forecasts[['date', 'origin', 'event', 'horizon', 'model', 'actual', 'prediction', 'error']].to_csv(
        args.out/'forecast_errors.csv', index=False)
    summaries.to_csv(args.out/'errors_by_origin.csv', index=False)
    gains.to_csv(args.out/'rmse_reduction_by_origin.csv', index=False)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    fig = draw(gains)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    temporary_pdf = args.pdf.with_name(f'.{args.pdf.stem}.{os.getpid()}.tmp.pdf')
    fig.savefig(temporary_pdf, metadata={'Title': 'CSI 300 RMSE reductions relative to matched ARIMA',
                                   'Subject': 'Spectral DC ARTFIMA; paired 95% archive-date cluster bootstrap intervals',
                                   'Creator': Path(__file__).name})
    temporary_pdf.replace(args.pdf)
    plt.close(fig)
    print(args.pdf)
    print(json.dumps(protocol['samples'], indent=2))
    print(gains.to_string(index=False))


if __name__ == '__main__':
    main()
