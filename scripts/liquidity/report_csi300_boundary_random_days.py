#!/usr/bin/env python3
"""Render the completed Spectral DC comparison as a PDF for any sample size."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/csi300-boundary-report-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

LABELS = {'SpectralDC_ARTFIMA': 'Spectral DC ARTFIMA',
          'Matched_ARIMA011_MAP': 'Matched ARIMA on transformed series',
          'LastValue': 'Naive', 'ClockLastValue': 'Clock-adjusted Naive'}
COLORS = ['#167c80', '#ce7930', '#765ca2', '#62748a']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home()/'running/dts'
    parser.add_argument('--root', type=Path, default=base/'artifacts/liquidity/csi300_boundary_random100_v2')
    parser.add_argument('--pdf', type=Path, default=base/'output/pdf/csi300_boundary_random100_comparison.pdf')
    args = parser.parse_args()
    protocol = json.loads((args.root/'protocol.json').read_text())
    total = pd.read_csv(args.root/'scores.csv')
    paired = pd.read_csv(args.root/'paired_rmse.csv')
    forecasts = pd.read_parquet(args.root/'forecasts.parquet')
    diagnostics = pd.read_csv(args.root/'daily_diagnostics.csv')
    plan = pd.read_csv(args.root/'forecast_tasks.csv')
    failures = json.loads((args.root/'failures.json').read_text())
    refs = []
    for date in protocol['references']:
        path = args.root/'days'/date/'dc_vs_full.json'
        if path.exists():
            refs.append((date, json.loads(path.read_text())))
    accuracy_passed = len(refs) == len(protocol['references']) and all(v['passed'] for _, v in refs)
    verdict = ('Representative DC accuracy checks passed' if accuracy_passed else
               f'{sum(v["passed"] for _, v in refs)} of {len(protocol["references"])} DC accuracy checks passed; forecasts remain exploratory')
    models = list(LABELS)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.pdf) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
        for i, model in enumerate(models):
            g = total[total.model == model].set_index('horizon')
            axes[0].bar(np.arange(2)+(i-1.5)*.18, g.loc[[60, 300], 'rmse'],
                        width=.18, color=COLORS[i], label=LABELS[model])
        axes[0].set_xticks([0, 1], ['1-minute average', '5-minute average'])
        axes[0].set_ylabel('RMSE (bps)')
        axes[0].set_title('Same origins and outcomes', loc='left')
        axes[0].legend(frameon=False, fontsize=8, loc='upper left')
        axes[0].grid(axis='y', alpha=.15)
        for offset, h, color in [(-.12, 60, COLORS[0]), (.12, 300, COLORS[1])]:
            g = paired[paired.horizon == h].set_index('baseline').loc[models[1:]]
            y = np.arange(3)+offset
            axes[1].hlines(y, g.lower_pct, g.upper_pct, color=color)
            axes[1].plot(g.rmse_reduction_pct, y, 'o', color=color, label=f'{h//60}-minute')
        axes[1].axvline(0, color='#888888', ls='--', lw=1)
        axes[1].set_yticks(np.arange(3), [LABELS[m] for m in models[1:]], fontsize=8)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('ARTFIMA RMSE reduction (%)')
        axes[1].set_title('Paired comparisons | 95% intervals', loc='left')
        axes[1].legend(frameon=False, loc='lower right')
        axes[1].grid(axis='x', alpha=.15)
        fig.suptitle(f'Liquidity boundary forecasts | {protocol["days"]:,}-date evaluation',
                     x=.08, ha='left', fontsize=18, y=.96)
        fig.text(.08, .90, f'{len(diagnostics)} completed daily fits; {forecasts.date.nunique()} issued dates; {len(failures)} failed daily tasks.\n{verdict}.', fontsize=9)
        fig.text(.08, .08, 'Clock change means/scales use earlier training only. ARTFIMA uses manuscript frequency partitions and joint posterior forecasts.', fontsize=9)
        fig.text(.08, .055, 'Matched null: MA(1) on standardized changes, ARIMA(0,1,1) on their cumulative sum; full Whittle MAP, without parameter uncertainty.', fontsize=9)
        fig.text(.08, .03, 'Historical weights and available-constituent series remain research approximations. Confidence intervals resample 20-archive-date clusters.', fontsize=9)
        fig.subplots_adjust(left=.08, right=.96, top=.80, bottom=.20, wspace=.62)
        pdf.savefig(fig)
        plt.close(fig)

        scored = forecasts[np.isfinite(forecasts.actual)]
        grouped = scored.groupby(['event', 'model', 'horizon']).agg(mse=('se', 'mean'), n=('se', 'size')).reset_index()
        grouped['rmse'] = np.sqrt(grouped.mse)
        fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
        events = ['opening', 'pre_lunch', 'reopening', 'end_of_grid']
        names = ['Opening | 09:31 / 09:35', 'Before lunch | 11:25 / 11:29',
                 'Reopening | 13:01 / 13:05', 'End of available grid | 14:52 / 14:56']
        for ax, event, title in zip(axes.flat, events, names):
            for i, model in enumerate(models):
                g = grouped[(grouped.event == event)&(grouped.model == model)].set_index('horizon')
                ax.bar(np.arange(2)+(i-1.5)*.18, g.loc[[60, 300], 'rmse'], width=.18,
                       color=COLORS[i], label=LABELS[model])
            counts = grouped[(grouped.event == event)&(grouped.model == models[0])].set_index('horizon')
            ax.set_xticks([0, 1], [f'1 minute\nn={int(counts.loc[60, "n"])}', f'5 minutes\nn={int(counts.loc[300, "n"])}'])
            ax.set_title(title, loc='left', fontsize=11)
            ax.set_ylabel('RMSE (bps)')
            ax.grid(axis='y', alpha=.15)
        fig.suptitle('Forecast performance by trading boundary', x=.08, ha='left', fontsize=18, y=.97)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.52, .93), ncol=2, frameon=False, fontsize=9)
        fig.text(.08, .03, 'Origins have 60, 300 or 600 contiguous past levels, identically across models. Targets never cross lunch or the 14:57 endpoint.', fontsize=9)
        fig.subplots_adjust(left=.08, right=.96, top=.83, bottom=.13, hspace=.45, wspace=.25)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
        coverage = plan.groupby(['origin', 'horizon']).agg(issued=('issued', 'sum'), scored=('scorable', 'sum')).reset_index()
        y = np.arange(len(coverage))
        axes[0].barh(y, coverage.issued, color='#d8dfe5', label='Issuable')
        axes[0].barh(y, coverage.scored, color=COLORS[0], label='Complete target')
        axes[0].set_yticks(y, [f'{r.origin} | {r.horizon//60}m' for r in coverage.itertuples()], fontsize=9)
        axes[0].invert_yaxis()
        axes[0].set_xlim(0, protocol['days']+3)
        axes[0].set_xlabel('Dates in the fixed evaluation sample')
        axes[0].set_title('Data availability', loc='left')
        axes[0].legend(frameon=False, loc='upper left', bbox_to_anchor=(0, -.12), ncol=2, fontsize=9)
        axes[1].axhline(.1, color='#888888', ls='--', lw=1)
        axes[1].axhline(-.1, color='#888888', ls='--', lw=1)
        for i, (date, reference) in enumerate(refs):
            axes[1].plot(np.arange(4)+(i-1)*.07, reference['mean_shift_reference_sd'],
                         'o-', color=COLORS[i], lw=1, label=date)
        axes[1].set_xticks(np.arange(4), ['MA', 'log lambda', 'log variance', 'd'], rotation=15)
        axes[1].set_ylabel('DC minus full mean / full posterior SD')
        axes[1].set_title('Representative full-versus-DC checks', loc='left')
        axes[1].legend(frameon=False, fontsize=9)
        axes[1].grid(axis='y', alpha=.15)
        fig.suptitle('Coverage and posterior accuracy', x=.08, ha='left', fontsize=18, y=.96)
        fig.text(.08, .90, verdict+'. SD ratios must also lie between 0.9 and 1.1.', fontsize=10)
        selection = ('All last 1,000 usable forecast dates.' if protocol['days'] == 1000 else
                     f'Seed: {protocol["seed"]}. Dates sampled without replacement from the last 1,000 usable dates.')
        fig.text(.08, .08, selection+' No replacement of failed dates.', fontsize=9)
        fig.text(.08, .055, 'Missing outcomes remain missing. The closing 14:57 endpoint causes substantial target loss; endpoint sensitivity is a separate analysis.', fontsize=9)
        fig.text(.08, .03, 'Stationarity of the concatenated observed-time changes is a working assumption. Passing numerical diagnostics does not establish it.', fontsize=9)
        fig.subplots_adjust(left=.12, right=.96, top=.80, bottom=.20, wspace=.45)
        pdf.savefig(fig)
        plt.close(fig)
    print(args.pdf)


if __name__ == '__main__':
    main()
