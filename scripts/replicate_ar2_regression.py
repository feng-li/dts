#!/usr/bin/env python3
"""Replicate the AR(2) regression divide-and-conquer diagnostic table."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dts.runtime import suppress_numeric_warnings

suppress_numeric_warnings()

from dts.aggregation import credible_interval, wasserstein_quantile_distance
from dts.regression import (
    RegressionSettings,
    RegressionSpec,
    fit_regression_frequency_divide_and_conquer,
    load_ar2_regression,
    regression_parameter_names,
    transform_regression_draws,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "DC-BATS_AR2_try.npy")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "ar2_regression")
    parser.add_argument("--preset", choices=["quick", "paper"], default="quick")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--groups", type=int, default=None)
    parser.add_argument("--max-observations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--basinhopping", action="store_true")
    parser.add_argument("--basinhopping-iter", type=int, default=25)
    parser.add_argument("--no-progress", action="store_true", help="disable progress bars")
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> tuple[RegressionSettings, int, int | None]:
    if args.preset == "paper":
        samples = 15000
        burn_in = 5000
        groups = 16
        max_observations = None
        optimize = True
    else:
        samples = 160
        burn_in = 60
        groups = 2
        max_observations = 1501
        optimize = False

    if args.samples is not None:
        samples = args.samples
    if args.burn_in is not None:
        burn_in = args.burn_in
    if args.groups is not None:
        groups = args.groups
    if args.max_observations is not None:
        max_observations = args.max_observations
    if args.no_optimize:
        optimize = False

    return (
        RegressionSettings(
            n_samples=samples,
            burn_in=burn_in,
            seed=args.seed,
            optimize=optimize,
            basinhopping=args.basinhopping,
            basinhopping_iter=args.basinhopping_iter,
            progress=not args.no_progress,
        ),
        groups,
        max_observations,
    )


def write_summary(path: Path, method: str, spec: RegressionSpec, draws: np.ndarray, reference: np.ndarray | None):
    transformed = transform_regression_draws(draws, spec)
    ref_transformed = transform_regression_draws(reference, spec) if reference is not None else None
    distances = wasserstein_quantile_distance(ref_transformed, transformed) if ref_transformed is not None else None
    intervals = credible_interval(transformed)
    names = regression_parameter_names(spec)
    file_exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "parameter", "mean", "q025", "q50", "q975", "wasserstein"],
        )
        if not file_exists:
            writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow(
                {
                    "method": method,
                    "parameter": name,
                    "mean": float(np.mean(transformed[:, index])),
                    "q025": float(intervals[index, 0]),
                    "q50": float(intervals[index, 1]),
                    "q975": float(intervals[index, 2]),
                    "wasserstein": "" if distances is None else float(distances[index]),
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ar2_ci_summary.csv"
    if summary_path.exists():
        summary_path.unlink()

    settings, groups, max_observations = settings_from_args(args)
    x, y = load_ar2_regression(args.data)
    if max_observations is not None:
        x = x[-max_observations:]
        y = y[-max_observations:]

    spec = RegressionSpec(q=2, p=0, n_exog=x.shape[1], tfi_term=False)
    result = fit_regression_frequency_divide_and_conquer(
        x=x,
        y=y,
        spec=spec,
        settings=settings,
        n_groups=groups,
        include_full=True,
    )

    reference = result.full.draws if result.full else None
    write_summary(summary_path, "full", spec, reference, reference=None)
    write_summary(summary_path, f"frequency_G{groups}", spec, result.consensus_draws, reference=reference)
    write_summary(summary_path, f"average_G{groups}", spec, result.average_draws, reference=reference)

    manifest = {
        "data": str(args.data),
        "preset": args.preset,
        "n_obs": int(len(y)),
        "groups": groups,
        "samples": settings.n_samples,
        "burn_in": settings.burn_in,
        "optimize": settings.optimize,
        "basinhopping": settings.basinhopping,
        "acceptance_rates": [item.acceptance_rate for item in result.shards],
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
