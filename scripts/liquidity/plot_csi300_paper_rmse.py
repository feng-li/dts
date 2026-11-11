#!/usr/bin/env python3
"""Reproduce the paper's G=20 RMSE figure from its audited comparison output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from scripts.liquidity.plot_csi300_liquidity_forecast_errors import ORIGINS, draw
import matplotlib.pyplot as plt


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_summary(root, review):
    """Keep the saved G=20 forecasts and paired-bootstrap summary matched."""
    config = json.loads((root / 'run_config.json').read_text())
    audit = json.loads((review / 'audit.json').read_text())
    protocol = json.loads((review / 'manuscript_update/figure_protocol.json').read_text())
    if ((config['groups'], config['chains']) != (20, 1) or
            (protocol['groups'], protocol['chains']) != (20, 1)):
        raise ValueError('Expected the audited G=20, one-chain paper experiment')
    if not audit['all_daily_sampling_mode_consensus_covariance_forecast_gates_pass']:
        raise ValueError('The saved experiment did not pass its audit')
    forecast_sha = digest(root / 'forecasts.parquet')
    if not (forecast_sha == audit['new_forecast_sha256'] == protocol['source_forecasts_sha256']):
        raise ValueError('Forecasts differ from the audited paper experiment')
    comparison = review / 'paired_rmse_reductions.csv'
    if digest(comparison) != protocol['comparison_csv_sha256']:
        raise ValueError('Paired-bootstrap results differ from the paper figure record')

    gains = pd.read_csv(comparison)
    gains = gains[(gains.baseline == 'Matched_ARIMA011_MAP') &
                  (gains.scope == 'origin')].copy()
    gains = gains.rename(columns={'label': 'origin', 'reduction_pct': 'rmse_reduction_pct'})
    expected = {(horizon, origin) for horizon in [60, 300] for origin in ORIGINS
                if horizon == 60 or origin not in ['11:29', '14:56']}
    keys = list(zip(gains.horizon, gains.origin))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('Expected the 14 scorable paper origin/horizon combinations')
    values = gains[['rmse_reduction_pct', 'lower_pct', 'upper_pct']].to_numpy()
    if (not np.isfinite(values).all() or not gains.n.gt(0).all() or
            not gains.lower_pct.le(gains.upper_pct).all()):
        raise ValueError('Invalid saved RMSE reductions or bootstrap intervals')
    full = pd.MultiIndex.from_product([[60, 300], ORIGINS], names=['horizon', 'origin'])
    gains = gains.set_index(['horizon', 'origin']).reindex(full).reset_index()
    gains['n'] = gains.n.fillna(0).astype(int)
    return gains, protocol


def main():
    base = Path.home() / 'running/dts'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=base / 'artifacts/liquidity/csi300_boundary_spark_g20_full1000_onechain_v5')
    parser.add_argument('--review', type=Path, help='Audit/summary directory; default: ROOT/review_20261016')
    parser.add_argument('--out', type=Path, default=base / 'artifacts/liquidity/csi300_paper_rmse')
    parser.add_argument('--pdf', type=Path, default=base / 'output/pdf/csi300_liquidity_forecast_errors_intraday_full1000.pdf')
    args = parser.parse_args()
    review = args.review or args.root / 'review_20261016'
    gains, original = load_summary(args.root, review)
    args.out.mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.pdf.with_name(f'.{args.pdf.stem}.{os.getpid()}.partial.pdf')
    fig = draw(gains)
    try:
        fig.savefig(temporary, metadata={
            'Title': 'CSI 300 RMSE reductions relative to matched ARIMA',
            'Subject': 'Stabilized G=20; paired 95% archive-date cluster bootstrap intervals',
            'Creator': Path(__file__).name})
        temporary.replace(args.pdf)
    finally:
        plt.close(fig)
    gains.to_csv(args.out / 'figure_values.csv', index=False)
    metadata = dict(root=str(args.root.resolve()), review=str(review.resolve()),
                    groups=20, chains=1, bootstrap=original['bootstrap'],
                    source_forecasts_sha256=original['source_forecasts_sha256'],
                    comparison_csv_sha256=original['comparison_csv_sha256'],
                    drawing_script_sha256=digest(Path(__file__).with_name('plot_csi300_liquidity_forecast_errors.py')),
                    script_sha256=digest(Path(__file__)), pdf_sha256=digest(args.pdf),
                    operation='Plot saved audited summaries; no refitting or bootstrap recomputation')
    (args.out / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(args.pdf)


if __name__ == '__main__':
    main()
