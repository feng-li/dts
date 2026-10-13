#!/usr/bin/env python3
"""Render session profiles for all/recent history or the forecast train/test split."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', str(Path.home()/'running/dts/cache/matplotlib'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scripts.liquidity.report_csi300_session_boundaries import intraday_profile


def forecast_sample_masks(dates, forecast_root, manifest_sha256):
    """Use the initial training window and the actual evaluated dates, without overlap."""
    plan_path = forecast_root/'sample_dates.csv'
    protocol_path = forecast_root/'protocol.json'
    plan = pd.read_csv(plan_path, dtype={'date': str, 'train_first': str, 'train_last': str})
    protocol = json.loads(protocol_path.read_text())
    if protocol['input_manifest_sha256'] != manifest_sha256:
        raise ValueError('Forecast evaluation and plot input have different manifests')
    test_dates = plan['date'].to_numpy()
    if (not len(test_dates) or len(np.unique(test_dates)) != len(test_dates)
            or not np.all(test_dates[:-1] < test_dates[1:])
            or list(test_dates) != protocol['dates']):
        raise ValueError('Expected unique, ordered evaluation dates matching the forecast protocol')
    if not np.isin(test_dates, dates).all():
        raise ValueError('An evaluated date is missing from the plot input')
    first = plan.iloc[0]
    train = (dates >= first['train_first']) & (dates <= first['train_last'])
    test = np.isin(dates, test_dates)
    if not train.any() or dates[train][-1] >= test_dates[0] or np.any(train & test):
        raise ValueError('Initial training must end strictly before the first test date')
    stop = int(first['train_stop_day'])
    if stop != int(first['day_index']) or dates[stop] != test_dates[0] or dates[stop-1] != first['train_last']:
        raise ValueError('Initial training boundary differs from the saved forecast plan')
    return [('Initial training', train), ('Test', test)], dict(
        forecast_root=str(forecast_root.resolve()),
        sample_dates_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        forecast_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        first_forecast_date=str(test_dates[0]), evaluated_dates=list(test_dates),
        initial_train_first=str(first['train_first']), initial_train_last=str(first['train_last']),
        training='Panel (a) is the initial training window; earlier test dates enter later expanding training windows',
        test='Only dates in the saved evaluation plan, subject to the same descriptive daily coverage filter')


def draw(profiles, compact, raw=None, y_scale='linear', vector=False):
    plt.rcParams.update({'font.size': 8 if compact else 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(6.14, 3.7) if compact else (11.7, 8.3),
                             sharey=raw is None or y_scale == 'log')
    if raw is not None and y_scale == 'log':
        low = min(p[1]['min_spread_bps'] for p in profiles)
        high = max(p[1]['max_spread_bps'] for p in profiles)
        powers = np.arange(int(np.floor(np.log2(low/5))),
                           int(np.ceil(np.log2(high/5)))+1)
        log_ticks = 5*2.**powers
    x = np.arange(237)+.5
    for i, (ax, (label, profile, date_range)) in enumerate(zip(axes, profiles)):
        if raw is None:
            for lo, hi in [(0, 120), (120, 237)]:
                ax.fill_between(x[lo:hi], profile['lower'][lo:hi], profile['upper'][lo:hi],
                                color='#167c80', alpha=.17, lw=.5)
                ax.plot(x[lo:hi], profile['median'][lo:hi], color='#167c80',
                        lw=1.2 if compact else 1.8)
        else:
            # Every finite second is drawn. Chunking bounds temporary memory;
            # Vector/raster storage does not change the observations plotted.
            indices = np.flatnonzero(profile['eligible'])
            for start in range(0, len(indices), 128):
                values = np.asarray(raw[indices[start:start+128]])
                observed = np.isfinite(values)
                seconds = np.nonzero(observed)[1]
                ax.scatter((seconds+1)/60, values[observed], s=.5 if compact else .35,
                           alpha=.06, color='#167c80', edgecolors='none', rasterized=not vector)
        ax.axvline(120, color='#888888', ls=':', lw=.8)
        if raw is None:
            ax.axhline(1, color='#888888', ls='--', lw=.65)
        ax.set_xticks([0, 60, 120, 180, 237],
                      ['09:30', '10:30', '11:30 / 13:00', '14:00', '14:57'])
        ax.get_xticklabels()[-1].set_ha('right')
        ax.set_xlim(0, 237)
        if raw is None:
            ax.set_ylim(.6, 2.2)
            ax.set_yticks([.8, 1., 1.4, 1.8, 2.2])
        elif y_scale == 'log':
            ax.set_yscale('log', base=2)
            ax.set_ylim(log_ticks[0], log_ticks[-1])
            ax.set_yticks(log_ticks)
            ax.yaxis.set_major_formatter(ScalarFormatter())
        else:
            ax.set_ylim(0, profile['max_spread_bps']*1.04)
        ax.grid(axis='y', alpha=.15)
        ax.set_ylabel(('Spread (bps; log scale)' if y_scale == 'log' else 'Spread (bps)') if raw is not None else
                      ('Spread / daily mean' if compact else 'Spread / same-day mean'))
        panel_label = f'Since {date_range.split(" to ")[0]}' if compact and label == 'Recent period' else label
        title = f'({chr(97+i)}) {panel_label}'
        ax.set_title(title, loc='left', fontsize=8.5 if compact else 11, pad=5)
        if not compact:
            ax.text(1., 1.015, date_range, transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=9, color='#62748a')
    if compact:
        fig.subplots_adjust(left=.11, right=.985, top=.94, bottom=.105, hspace=.53)
    else:
        fig.subplots_adjust(left=.08, right=.96, top=.95, bottom=.075, hspace=.40)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home()/'running/dts'
    parser.add_argument('--input', type=Path, default=base/'artifacts/liquidity/csi300_dc_1m')
    parser.add_argument('--recent-start', default='20241107')
    parser.add_argument('--forecast-root', type=Path,
                        help='Use this evaluation\'s sample_dates.csv to show initial training and test panels')
    parser.add_argument('--style', choices=['profile', 'scatter'], default='profile')
    parser.add_argument('--y-scale', choices=['linear', 'log'], default='linear')
    parser.add_argument('--vector', action='store_true',
                        help='Export full vector scatter to separate _vector files (large)')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--paper-pdf', type=Path)
    parser.add_argument('--skip-paper', action='store_true', help='Generate only the full-size figure used by the application')
    args = parser.parse_args()
    if args.y_scale == 'log' and args.style != 'scatter':
        parser.error('--y-scale log requires --style scatter')
    stem = 'csi300_liquidity_session_profile'+('_scatter' if args.style == 'scatter' else '')
    if args.y_scale == 'log':
        stem += '_log'
    if args.forecast_root is not None:
        stem += '_train_test'
    if args.vector and args.style == 'scatter':
        stem += '_vector'
    args.out = args.out or base/'artifacts/liquidity'/stem
    args.pdf = args.pdf or base/'output/pdf'/f'{stem}.pdf'
    args.paper_pdf = args.paper_pdf or base/'output/pdf'/f'{stem}_paper.pdf'
    manifest = args.input/'input_manifest.json'
    meta = json.loads(manifest.read_text())
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    dates = np.asarray(meta['dates'])
    if not np.all(dates[:-1] < dates[1:]):
        raise ValueError('Expected unique, ordered input dates')
    raw = np.load(args.input/'spread.npy', mmap_mode='r')
    if raw.shape != (len(dates), 14220) or meta['size_spec'] != 'notional_1e+06':
        raise ValueError('Expected the existing RMB 1m one-second cache')
    if args.forecast_root is not None:
        masks, split = forecast_sample_masks(dates, args.forecast_root, manifest_sha256)
    else:
        masks = [('All history', np.ones(len(dates), dtype=bool)),
                 ('Recent period', dates >= args.recent_start)]
        split = dict(recent_start=args.recent_start)
    profiles, rows, counts = [], [], []
    for label, mask in masks:
        if args.style == 'scatter':
            eligible = mask & (np.isfinite(raw).sum(axis=1) >= .95*14220)
            if not eligible.any():
                raise ValueError('No eligible dates for the scatter plot')
            values = np.asarray(raw[eligible])
            profile = dict(eligible=eligible, plotted_seconds=int(np.isfinite(values).sum()),
                           min_spread_bps=float(np.nanmin(values)),
                           max_spread_bps=float(np.nanmax(values)))
            if (values[np.isfinite(values)] <= 0).any():
                raise ValueError('Nonpositive raw spread')
        else:
            profile = intraday_profile(raw, mask)
        picked = dates[mask]
        date_range = f'{picked[0][:4]}-{picked[0][4:6]}-{picked[0][6:]} to {picked[-1][:4]}-{picked[-1][4:6]}-{picked[-1][6:]}'
        profiles.append((label, profile, date_range))
        counts.append(dict(sample=label, scheduled_days=int(mask.sum()), eligible_days=int(profile['eligible'].sum()),
                           first=str(picked[0]), last=str(picked[-1])))
        if args.style == 'scatter':
            counts[-1].update(plotted_seconds=profile['plotted_seconds'],
                              min_spread_bps=profile['min_spread_bps'], max_spread_bps=profile['max_spread_bps'])
        else:
            counts[-1]['min_contributing_days'] = int(profile['contributing_days'].min())
        for j in range(237) if args.style == 'profile' else []:
            rows.append(dict(sample=label, minute_index=j, session='am' if j < 120 else 'pm',
                             median=profile['median'][j], lower=profile['lower'][j], upper=profile['upper'][j],
                             contributing_days=int(profile['contributing_days'][j])))
    args.out.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(args.out/'profiles.csv', index=False)
    protocol = dict(input=str(args.input.resolve()), sample_split=split,
                    input_manifest_sha256=manifest_sha256, samples=counts,
                    style=args.style, y_scale=args.y_scale, daily_coverage=.95,
                    figure_title=False, footer_notes=False, panel_day_counts=False,
                    normalization=('None; raw basis-point values' if args.style == 'scatter' else
                                   'Observed spread divided by its same-day observed-grid mean, then arithmetic minute mean'),
                    clock='Lunch removed; AM and PM drawn separately; no data after 14:57',
                    use='Descriptive retrospective chart; same-day normalizer is not a forecasting feature')
    if args.style == 'scatter':
        protocol.update(points='Every finite observed second on eligible dates', subsampling=False,
                        rendering=('Vector scatter markers and labels' if args.vector else
                                   'Rasterized scatter markers at 200 dpi; vector labels'),
                        vertical_range=('Common log range spanning all observed values, with bps tick labels'
                                        if args.y_scale == 'log' else
                                        'Each panel includes all observed values; panel ranges differ'))
    else:
        protocol.update(minute_coverage='At least one observed second; identical to the reference chart',
                        band='25th and 75th percentiles across eligible dates; not a confidence or forecast interval')
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    outputs = [(False, args.pdf)]
    if not args.skip_paper:
        outputs.append((True, args.paper_pdf))
    for compact, path in outputs:
        print(f'Rendering {path.name}', flush=True)
        fig = draw(profiles, compact, raw=raw if args.style == 'scatter' else None,
                   y_scale=args.y_scale, vector=args.vector)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200 if args.style == 'scatter' else 100,
                    metadata={'Title': 'Liquidity around session boundaries | RMB 1m spread',
                                    'Subject': f'{"Initial training and evaluated test dates" if args.forecast_root else "All history and recent period from " + args.recent_start}; {args.style}; {args.y_scale} axis',
                                    'Creator': Path(__file__).name})
        plt.close(fig)
        print(path, flush=True)
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
