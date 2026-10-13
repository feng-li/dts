#!/usr/bin/env python3
"""Describe opening, lunch and end-of-grid spreads; no forecasting model fitted."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/csi300-boundary-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from tqdm import tqdm

WINDOWS = {
    'open': (0, 300, '09:30-09:35'),
    'open_reference': (600, 900, '09:40-09:45'),
    'pre_lunch': (6900, 7200, '11:25-11:30'),
    'pre_lunch_reference': (6300, 6600, '11:15-11:20'),
    'reopen': (7200, 7500, '13:00-13:05'),
    'reopen_reference': (7800, 8100, '13:10-13:15'),
    'end_grid': (13920, 14220, '14:52-14:57'),
    'end_reference': (13320, 13620, '14:42-14:47'),
    'last_minute': (14160, 14220, '14:56-14:57'),
    'previous_minute': (14100, 14160, '14:55-14:56'),
}
PAIRS = [('open', 'open_reference'), ('pre_lunch', 'pre_lunch_reference'),
         ('reopen', 'reopen_reference'), ('end_grid', 'end_reference'),
         ('reopen', 'pre_lunch'), ('last_minute', 'previous_minute')]


def intraday_profile(raw, mask, minimum_daily_coverage=.95):
    """Reference chart: day-normalized minute means, then day quantiles.

    Normalization uses each observed day's entire grid for a descriptive
    profile. It is separate from causal forecast calibration. Missing seconds
    are retained; minute means use their available observations.
    """
    if raw.ndim != 2 or raw.shape[1] != 14220 or len(mask) != len(raw):
        raise ValueError('Expected a date mask and the original 14,220-second grid')
    if not 0 < minimum_daily_coverage <= 1:
        raise ValueError('Invalid daily coverage threshold')
    eligible = np.asarray(mask, dtype=bool) & (
        np.isfinite(raw).sum(axis=1) >= minimum_daily_coverage*14220)
    if not eligible.any():
        raise ValueError('No dates pass the descriptive-profile coverage rule')
    values = np.asarray(raw[eligible]).copy()
    normalizer = np.nanmean(values, axis=1)
    if (normalizer <= 0).any() or not np.isfinite(normalizer).all():
        raise ValueError('Nonpositive or nonfinite daily spread mean')
    values /= normalizer[:, None]
    blocks = values.reshape(len(values), 237, 60)
    count = np.isfinite(blocks).sum(axis=2)
    minutes = np.divide(np.nansum(blocks, axis=2), count,
                        out=np.full(count.shape, np.nan), where=count > 0)
    lower, median, upper = np.nanquantile(minutes, [.25, .5, .75], axis=0)
    return dict(eligible=eligible, median=median, lower=lower, upper=upper,
                contributing_days=np.isfinite(minutes).sum(axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home() / 'running/dts'
    parser.add_argument('--input', type=Path, default=base / 'artifacts/liquidity/csi300_dc_1m')
    parser.add_argument('--out', type=Path, default=base / 'artifacts/liquidity/csi300_session_boundaries')
    parser.add_argument('--pdf', type=Path, default=base / 'output/pdf/csi300_session_boundaries.pdf')
    parser.add_argument('--recent-start', default='20241107')
    args = parser.parse_args()
    meta = json.loads((args.input / 'input_manifest.json').read_text())
    dates = np.array(meta['dates'])
    raw = np.load(args.input / 'spread.npy', mmap_mode='r')
    if raw.shape != (len(dates), 14220) or meta['size_spec'] != 'notional_1e+06':
        raise ValueError('Expected the existing RMB 1m second grid')
    args.out.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    daily = pd.DataFrame({'date': dates})
    for name, (lo, hi, _) in tqdm(WINDOWS.items(), desc='Boundary windows'):
        x = raw[:, lo:hi]
        count = np.isfinite(x).sum(axis=1)
        mean = np.divide(np.nansum(x, axis=1), count,
                         out=np.full(len(dates), np.nan), where=count > 0)
        mean[count < .95 * (hi - lo)] = np.nan
        daily[name] = mean
        daily[name + '_valid_seconds'] = count
    rows = []
    rng = np.random.default_rng(20261008)
    masks = [('all_history', np.ones(len(dates), dtype=bool)),
             ('recent_evaluation', dates >= args.recent_start)]
    masks += [(year, np.char.startswith(dates, year)) for year in sorted(set(d[:4] for d in dates))]
    for period, mask in masks:
        for name, reference in PAIRS:
            good = mask & np.isfinite(daily[name]) & np.isfinite(daily[reference]) & (daily[reference] > 0) & (daily[name] > 0)
            event = daily.loc[good, name].to_numpy()
            control = daily.loc[good, reference].to_numpy()
            log_ratio = np.log(event / control)
            n = len(event)
            if not n:
                continue
            starts = rng.integers(n, size=(2000, (n + 4) // 5))
            index = ((starts[:, :, None] + np.arange(5)) % n).reshape(2000, -1)[:, :n]
            ci = 100 * np.expm1(np.quantile(log_ratio[index].mean(axis=1), [.025, .975]))
            rows.append(dict(period=period, event=name, reference=reference,
                             event_window=WINDOWS[name][2], reference_window=WINDOWS[reference][2],
                             scheduled_days=int(mask.sum()), paired_days=n,
                             event_mean_bps=float(event.mean()), reference_mean_bps=float(control.mean()),
                             geometric_change_pct=float(100 * np.expm1(log_ratio.mean())),
                             lower_pct=ci[0], upper_pct=ci[1], higher_days_pct=100 * (log_ratio > 0).mean()))
    summary = pd.DataFrame(rows)
    daily.to_csv(args.out / 'daily_windows.csv', index=False)
    summary.to_csv(args.out / 'paired_windows.csv', index=False)
    protocol = dict(source=str(args.input), first=dates[0], last=dates[-1],
                    recent_start=args.recent_start, target=meta['target'], size_spec=meta['size_spec'],
                    window_valid_fraction=.95, bootstrap='2000 circular blocks of five retained paired dates',
                    change='Geometric mean of same-day event/reference arithmetic-window-mean ratios',
                    grid='09:30:01 through 11:30:00; 13:00:01 through 14:57:00',
                    scope='Descriptive retrospective available-constituent diagnostic, not causal or predictive evidence; no missing-value imputation; no observations after 14:57 and no 15:00 close')
    (args.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    with PdfPages(args.pdf) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(11.7, 8.3))
        for ax, (period, mask) in zip(axes, masks[:2]):
            profile = intraday_profile(raw, mask)
            eligible = profile['eligible']
            median, lower, upper = [profile[key] for key in ['median', 'lower', 'upper']]
            x = np.arange(237) + .5
            for lo, hi in [(0, 120), (120, 237)]:
                ax.plot(x[lo:hi], median[lo:hi], color='#167c80', lw=1.7)
                ax.fill_between(x[lo:hi], lower[lo:hi], upper[lo:hi], color='#167c80', alpha=.16)
            ax.axvline(120, color='#888888', ls=':', lw=1)
            ax.axhline(1, color='#888888', ls='--', lw=.7)
            ax.set_xticks([0, 60, 120, 180, 237], ['09:30', '10:30', '11:30 / 13:00', '14:00', '14:57'])
            ax.set_xlim(0, 237)
            ax.set_ylabel('Spread / same-day mean')
            ax.set_title(f'{period.replace("_", " ").title()} | {int(eligible.sum())} days with >=95% full-grid coverage', loc='left')
            ax.grid(axis='y', alpha=.15)
        fig.suptitle('Liquidity around session boundaries | RMB 1m spread', fontsize=17, x=.08, ha='left')
        fig.text(.08, .035, 'Median and interquartile range across days of daily-normalized one-minute means. Lunch removed; sessions drawn separately.', fontsize=9)
        fig.text(.08, .013, 'Available-constituent research approximation. Missing seconds stay missing. The input ends at 14:57; no 15:00 close is observed.', fontsize=9)
        fig.subplots_adjust(left=.08, right=.96, top=.89, bottom=.13, hspace=.35)
        pdf.savefig(fig)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        labels = ['Opening (5 min)', 'Before lunch (5 min)', 'After reopening (5 min)',
                  'End of grid (5 min)', 'Reopening vs pre-lunch', 'Last minute vs previous']
        for offset, period, color in [(-.13, 'all_history', '#167c80'), (.13, 'recent_evaluation', '#ce7930')]:
            g = summary[summary.period == period]
            y = np.arange(len(PAIRS)) + offset
            ax.errorbar(g.geometric_change_pct, y,
                        xerr=np.vstack([g.geometric_change_pct-g.lower_pct, g.upper_pct-g.geometric_change_pct]),
                        fmt='o', color=color, capsize=4, label=period.replace('_', ' ').title())
            for row, yy in zip(g.itertuples(), y):
                ax.annotate(f'n={row.paired_days}', (row.upper_pct + 1, yy), fontsize=9, va='center', color=color)
        ax.axvline(0, color='#888888', ls='--', lw=1)
        ax.set_yticks(np.arange(len(PAIRS)), labels)
        ax.invert_yaxis()
        ax.set_xlabel('Same-day event/reference change (%) | geometric mean and 95% block-bootstrap interval')
        ax.set_title('Paired boundary windows', loc='left', fontsize=17, pad=22)
        ax.legend(loc='lower right', frameon=False)
        ax.grid(axis='x', alpha=.15)
        ax.set_xlim(-6, 48)
        fig.text(.08, .065, 'References, in order: 09:40-09:45; 11:15-11:20; 13:10-13:15; 14:42-14:47; 11:25-11:30; 14:55-14:56.', fontsize=9)
        fig.text(.08, .04, 'Each paired window needs >=95% valid seconds. Recent period: 2024-11-07 through 2026-07-24. Counts vary by comparison.', fontsize=9)
        fig.text(.08, .015, 'These are clock-pattern descriptions. Existing forecast scores at 30/60/90 minutes into each session do not test these windows.', fontsize=9)
        fig.subplots_adjust(left=.24, right=.95, top=.87, bottom=.19)
        pdf.savefig(fig)
        plt.close(fig)
    print(summary[summary.period == 'recent_evaluation'].to_string(index=False))
    print(args.pdf)


if __name__ == '__main__':
    main()
