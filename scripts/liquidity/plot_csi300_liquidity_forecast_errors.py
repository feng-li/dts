#!/usr/bin/env python3
"""Compare saved liquidity forecast errors across scheduled trading times."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'running/dts/cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


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
    selected = forecasts[forecasts.date >= recent_start].copy()
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
                    date_selection='Saved random-date experiment, filtered by recent_start; no refitting',
                    daily_coverage='No additional whole-day coverage filter; every target is complete',
                    excluded_incomplete_targets=int(selected[['date', 'origin', 'horizon']].drop_duplicates().shape[0]
                                                    - scored[['date', 'origin', 'horizon']].drop_duplicates().shape[0]),
                    rendering='Vector kernel-density violins, individual-error markers, mean-error markers, text and axes',
                    clock='Eight chronological forecast origins at equal spacing; clock labels are not an elapsed-time scale',
                    mean_markers='Unconnected means at scheduled origins only; unavailable origins remain missing',
                    vertical_range='Symmetric limits shared across models within each horizon; all errors included',
                    sampling_checks='All daily fits passed; DC accuracy qualifications remain unchanged')
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


def draw(forecasts, summaries):
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(6.14, 5.0))
    grid = fig.add_gridspec(2, 1)
    model_offsets = [(model, .16*(i-1.5)) for i, model in enumerate(MODELS)]
    panels = [(fig.add_subplot(grid[0, :]), 60,
               model_offsets,
               '(a) All models | 1 min', True),
              (fig.add_subplot(grid[1, :]), 300,
               model_offsets,
               '(b) All models | 5 min', True)]
    x = np.arange(len(ORIGINS))
    jitter = np.random.default_rng(20261009)
    for ax, horizon, models, title, bottom in panels:
        sns.violinplot(data=forecasts[forecasts.horizon == horizon],
                       x='origin', y='error', hue='model', order=ORIGINS,
                       hue_order=[model[0] for model, _ in models],
                       palette={model[0]: model[2] for model, _ in models},
                       saturation=1, width=.64, gap=.1, dodge=True,
                       inner=None, cut=0, bw_method='scott', bw_adjust=1,
                       density_norm='width', common_norm=False,
                       linewidth=.5, legend=False, ax=ax)
        for violin in ax.collections:
            violin.set_alpha(.3)
            violin.set_zorder(2)
        for (model, name, color), offset in models:
            data = forecasts[(forecasts.model == model) & (forecasts.horizon == horizon)]
            means = summaries[(summaries.model == model) & (summaries.horizon == horizon)].set_index('origin').loc[ORIGINS]
            for i, origin in enumerate(ORIGINS):
                errors = data.loc[data.origin == origin, 'error'].to_numpy()
                if len(errors):
                    ax.scatter(i+offset+jitter.uniform(-.02, .02, len(errors)),
                               errors, s=3, color=color, alpha=.45,
                               edgecolors='none', zorder=3)
            ax.plot(x+offset, means.mean_error.to_numpy(), color=color,
                    marker='D', ms=3, linestyle='None', zorder=4)
        group_center = (models[0][1]+models[-1][1])/2
        if len(models) > 1:
            ax.legend(handles=[Patch(facecolor=model[2], alpha=.3, label=label)
                               for (model, _), label in zip(models, ['Spectral DC ARTFIMA', 'ARIMA',
                                                                    'Clock-adjusted Naive', 'Naive'])],
                      loc='upper right', ncol=4, frameon=False, fontsize=7,
                      handletextpad=.4, columnspacing=1.1)
        ax.axhline(0, color='#777777', ls='--', lw=.65, zorder=1)
        extent = max(1, np.ceil(np.abs(forecasts.loc[forecasts.horizon == horizon, 'error']).max()))
        ax.set_ylim(-extent*1.04, extent*1.04)
        ax.set_xlim(-.69, len(ORIGINS)-.31)
        available = means.n.to_numpy() > 0
        tick_origins = np.asarray(ORIGINS)[available]
        tick_labels = ([forecast_interval_label(origin, horizon) for origin in tick_origins]
                       if horizon == 300 else tick_origins)
        ax.set_xticks((x+group_center)[available], tick_labels, fontsize=6.5)
        ax.set_title(title, loc='left', fontsize=8, pad=5)
        ax.tick_params(axis='y', labelsize=7, labelleft=True)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel('Error (bps)', fontsize=8)
        if bottom:
            ax.set_xlabel('Five-minute forecast interval' if horizon == 300 else
                          'Scheduled forecast origin', fontsize=8, labelpad=5)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    handles = [Patch(facecolor='#62748a', alpha=.3, label='Error density (violin)'),
               Line2D([], [], color='#62748a', marker='o', lw=0, ms=3, alpha=.45,
                      label='Individual forecast error'),
               Line2D([], [], color='#62748a', marker='D', lw=0, ms=3,
                      label='Mean error at origin')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .005),
               ncol=3, frameon=False, fontsize=7, handletextpad=.4, columnspacing=1.5)
    fig.subplots_adjust(left=.095, right=.985, top=.94, bottom=.16, hspace=.55)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home()/'running/dts'
    parser.add_argument('--input', type=Path, default=base/'artifacts/liquidity/csi300_dc_1m')
    parser.add_argument('--root', type=Path,
                        default=base/'artifacts/liquidity/csi300_boundary_random100_v2')
    parser.add_argument('--recent-start', default='20241107')
    parser.add_argument('--out', type=Path,
                        default=base/'artifacts/liquidity/csi300_liquidity_forecast_errors_intraday')
    parser.add_argument('--pdf', type=Path,
                        default=base/'output/pdf/csi300_liquidity_forecast_errors_intraday.pdf')
    args = parser.parse_args()
    manifest = args.input/'input_manifest.json'
    meta = json.loads(manifest.read_text())
    if meta['size_spec'] != 'notional_1e+06':
        raise ValueError('Expected the RMB 1m spread archive')
    forecasts, protocol = recent_forecasts(args.root, args.recent_start, manifest, np.asarray(meta['dates']))
    summaries = errors_by_origin(forecasts)
    protocol.update(normalization='None; signed errors in original basis points', clipping=False,
                    missing_slots='Unavailable origins retain blank, unshaded slots without ticks or labels',
                    axis_count_labels=False,
                    five_minute_tick_labels='Origin to horizon endpoint; interval excludes the origin second and includes the endpoint',
                    paired_panel='All four models at each origin in both panels, ordered left to right as Spectral DC ARTFIMA, ARIMA, Clock-adjusted Naive and Naive, with category offsets -0.24, -0.08, 0.08 and 0.24; available clock ticks are centered on each group',
                    panel_layout='One-minute errors on top; five-minute errors at the bottom',
                    violin=dict(library=f'seaborn {sns.__version__}', bandwidth='Scott',
                                bandwidth_adjustment=1, cut=0,
                                support='Observed error range for each model and origin',
                                normalization='Equal maximum width; sample counts recorded in errors_by_origin.csv',
                                group_width=.64, gap=.1, inner=None,
                                points='All individual errors; horizontal uniform jitter only',
                                jitter_half_width=.02, jitter_seed=20261009),
                    time_separators=False)
    args.out.mkdir(parents=True, exist_ok=True)
    forecasts[['date', 'origin', 'event', 'horizon', 'model', 'actual', 'prediction', 'error']].to_csv(
        args.out/'forecast_errors.csv', index=False)
    summaries.to_csv(args.out/'errors_by_origin.csv', index=False)
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    fig = draw(forecasts, summaries)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.pdf, metadata={'Title': 'Recent CSI 300 forecast errors by trading time',
                                   'Subject': 'Matched models; one- and five-minute average-cost errors at scheduled trading times',
                                   'Creator': Path(__file__).name})
    plt.close(fig)
    print(args.pdf)
    print(json.dumps(protocol['samples'], indent=2))


if __name__ == '__main__':
    main()
