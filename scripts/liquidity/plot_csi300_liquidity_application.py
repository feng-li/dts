#!/usr/bin/env python3
"""Create the two-panel manuscript figure from the completed random-date test."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/csi300-liquidity-application-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path.home()/'running/dts'
    parser.add_argument('--root', type=Path,
                        default=base/'artifacts/liquidity/csi300_boundary_random100_v2')
    parser.add_argument('--out', type=Path,
                        default=base/'output/pdf/csi300_liquidity_application_figure.pdf')
    args = parser.parse_args()
    protocol = json.loads((args.root/'protocol.json').read_text())
    completed = pd.read_csv(args.root/'daily_diagnostics.csv')
    if len(completed) != protocol['days'] or not completed.sampling_passed.all():
        raise ValueError('The figure requires a completed experiment with passed sampling checks')
    forecasts = pd.read_parquet(args.root/'forecasts.parquet')
    scored = forecasts[np.isfinite(forecasts.actual)]
    rmse = scored.groupby(['event', 'horizon', 'model']).se.mean().pow(.5).unstack('model')
    gain = 100*(1-rmse.SpectralDC_ARTFIMA/rmse.Matched_ARIMA011_MAP)
    events = ['opening', 'pre_lunch', 'reopening', 'end_of_grid']
    references = [(date, json.loads((args.root/'days'/date/'dc_vs_full.json').read_text()))
                  for date in protocol['references']]
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(6.14, 2.1))
    x = np.arange(4)
    for offset, horizon, color in [(-.18, 60, '#167c80'), (.18, 300, '#ce7930')]:
        axes[0].bar(x+offset, [gain.loc[(event, horizon)] for event in events],
                    width=.34, color=color, label=f'{horizon//60} minute')
    axes[0].axhline(0, color='#555555', lw=.7)
    axes[0].set_xticks(x, ['Opening', 'Pre-lunch', 'Reopening', 'Grid end'], fontsize=7)
    axes[0].set_ylabel('RMSE reduction (%)')
    axes[0].set_ylim(-4.5, 15)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
    axes[0].set_title('(a) Forecast improvement', loc='left', fontsize=9)
    axes[1].axhspan(-.1, .1, color='#d8e6e8', alpha=.5)
    for i, (date, reference) in enumerate(references):
        label = f'{date[:4]}-{date[4:6]}-{date[6:]}'
        axes[1].plot(x+(i-1)*.06, reference['mean_shift_reference_sd'],
                     'o-', ms=3, lw=.8, label=label,
                     color=['#167c80', '#ce7930', '#765ca2'][i])
    axes[1].axhline(.1, color='#777777', lw=.6, ls='--')
    axes[1].axhline(-.1, color='#777777', lw=.6, ls='--')
    axes[1].set_xticks(x, [r'$\vartheta$', r'$\log\lambda$', r'$\log\sigma^2$', '$d$'])
    axes[1].set_ylabel('Mean shift / full SD')
    axes[1].set_ylim(-.13, .26)
    axes[1].legend(frameon=False, fontsize=6.5, loc='upper right')
    axes[1].set_title('(b) DC versus full posterior', loc='left', fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.095, right=.99, bottom=.2, top=.84, wspace=.57)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, metadata={'Title': 'CSI 300 liquidity forecast and DC accuracy assessment',
                                   'Subject': f'Random-date test; seed {protocol["seed"]}; source {args.root}',
                                   'Creator': Path(__file__).name})
    plt.close(fig)
    print(args.out)


if __name__ == '__main__':
    main()
