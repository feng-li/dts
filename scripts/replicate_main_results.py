#!/usr/bin/env python3
"""Replicate the main numerical results from docs/Manuscript_2026_Zi.tex."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-dts")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/dts-cache")

import matplotlib.pyplot as plt

from dts.aggregation import (
    credible_interval,
    parameter_names,
    transform_partial_draws,
    wasserstein_quantile_distance,
)
from dts.experiments import (
    MCMCSettings,
    fit_frequency_divide_and_conquer,
    fit_full_whittle,
    fit_time_domain_as_frequency_shards,
    load_series,
)
from dts.mcmc import ModelSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["quick", "paper"],
        default="quick",
        help="quick validates the pipeline; paper uses manuscript-scale settings",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["combination"],
        choices=["combination", "group-size", "partition", "time-frequency", "all"],
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "replication")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-observations", type=int, default=None)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--basinhopping", action="store_true")
    parser.add_argument("--basinhopping-iter", type=int, default=25)
    return parser.parse_args()


def settings_from_args(args: argparse.Namespace) -> tuple[MCMCSettings, int, int | None, bool]:
    if args.preset == "paper":
        samples = 15000
        burn_in = 5000
        groups = 10
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
        MCMCSettings(
            n_samples=samples,
            burn_in=burn_in,
            seed=args.seed,
            optimize=optimize,
            basinhopping=args.basinhopping,
            basinhopping_iter=args.basinhopping_iter,
        ),
        groups,
        max_observations,
        optimize,
    )


def maybe_truncate(data: np.ndarray, max_observations: int | None) -> np.ndarray:
    if max_observations is None:
        return data
    return data[:max_observations]


def save_draw_summary(
    csv_path: Path,
    dataset: str,
    method: str,
    model: ModelSpec,
    draws: np.ndarray,
    reference: np.ndarray | None = None,
) -> None:
    transformed = transform_partial_draws(draws, model.q, model.p, model.tfi_term)
    names = parameter_names(model.q, model.p, model.tfi_term)
    intervals = credible_interval(transformed)
    distances = None
    if reference is not None:
        ref_transformed = transform_partial_draws(reference, model.q, model.p, model.tfi_term)
        distances = wasserstein_quantile_distance(ref_transformed, transformed)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "method", "parameter", "mean", "q025", "q50", "q975", "wasserstein"],
        )
        if not file_exists:
            writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow(
                {
                    "dataset": dataset,
                    "method": method,
                    "parameter": name,
                    "mean": float(np.mean(transformed[:, index])),
                    "q025": float(intervals[index, 0]),
                    "q50": float(intervals[index, 1]),
                    "q975": float(intervals[index, 2]),
                    "wasserstein": "" if distances is None else float(distances[index]),
                }
            )


def plot_marginals(
    path: Path,
    dataset: str,
    model: ModelSpec,
    full_draws: np.ndarray | None,
    consensus_draws: np.ndarray,
    average_draws: np.ndarray | None = None,
    shard_draws: list[np.ndarray] | None = None,
) -> None:
    names = parameter_names(model.q, model.p, model.tfi_term)
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)

    def transformed(draws: np.ndarray) -> np.ndarray:
        return transform_partial_draws(draws, model.q, model.p, model.tfi_term)

    full_t = transformed(full_draws) if full_draws is not None else None
    consensus_t = transformed(consensus_draws)
    average_t = transformed(average_draws) if average_draws is not None else None
    shard_t = [transformed(item) for item in shard_draws or []]

    for index, name in enumerate(names):
        ax = axes[index // cols][index % cols]
        for shard in shard_t:
            ax.hist(shard[:, index], bins=35, density=True, histtype="step", color="0.72", linewidth=0.8)
        if full_t is not None:
            ax.hist(full_t[:, index], bins=40, density=True, histtype="step", color="#1f77b4", linewidth=1.8, label="full")
        ax.hist(consensus_t[:, index], bins=40, density=True, histtype="step", color="#d62728", linewidth=1.8, label="consensus")
        if average_t is not None:
            ax.hist(average_t[:, index], bins=40, density=True, histtype="step", color="#ff7f0e", linewidth=1.2, label="average")
        ax.set_title(name)
        ax.tick_params(labelsize=8)
    for index in range(len(names), rows * cols):
        axes[index // cols][index % cols].axis("off")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(dataset)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_combination(args, settings: MCMCSettings, groups: int, max_observations: int | None) -> list[dict]:
    specs = [
        ("sim_artfima", args.data_dir / "SimARTFIMA11.txt", ModelSpec(q=1, p=1, tfi_term=True)),
        ("vancouver", args.data_dir / "Vancouver.npy", ModelSpec(q=1, p=2, tfi_term=False)),
    ]
    records = []
    summary_csv = args.output_dir / "posterior_summary.csv"
    for dataset, path, model in specs:
        data = maybe_truncate(load_series(path), max_observations)
        result = fit_frequency_divide_and_conquer(
            data,
            model,
            settings,
            n_groups=groups,
            partition="systematic",
            include_full=True,
        )
        plot_marginals(
            args.output_dir / f"{dataset}_combination.png",
            dataset,
            model,
            result.full.draws if result.full else None,
            result.consensus_draws,
            average_draws=result.average_draws,
            shard_draws=[item.draws for item in result.shards],
        )
        reference = result.full.draws if result.full else None
        save_draw_summary(summary_csv, dataset, "full", model, reference)
        save_draw_summary(summary_csv, dataset, "consensus", model, result.consensus_draws, reference=reference)
        save_draw_summary(summary_csv, dataset, "average", model, result.average_draws, reference=reference)
        records.append(
            {
                "experiment": "combination",
                "dataset": dataset,
                "n_obs": int(len(data)),
                "groups": groups,
                "acceptance_rates": [item.acceptance_rate for item in result.shards],
            }
        )
    return records


def run_group_size(args, settings: MCMCSettings, max_observations: int | None) -> list[dict]:
    data = maybe_truncate(load_series(args.data_dir / "SimARTFIMA11.txt"), max_observations)
    group_values = [2, 4] if args.preset == "quick" else [10, 100, 1000]
    models = [
        ("arma11", ModelSpec(q=1, p=1, tfi_term=False)),
        ("artfima11", ModelSpec(q=1, p=1, tfi_term=True)),
    ]
    records = []
    summary_csv = args.output_dir / "posterior_summary.csv"
    for model_name, model in models:
        full = fit_full_whittle(data, model, settings)
        for groups in group_values:
            if groups >= (len(data) - 1) // 2:
                continue
            result = fit_frequency_divide_and_conquer(
                data,
                model,
                settings,
                n_groups=groups,
                partition="systematic",
                include_full=False,
            )
            dataset = f"group_size_{model_name}_G{groups}"
            save_draw_summary(summary_csv, dataset, "consensus", model, result.consensus_draws, reference=full.draws)
            records.append({"experiment": "group-size", "model": model_name, "groups": groups, "n_obs": int(len(data))})
    return records


def run_partition(args, settings: MCMCSettings, groups: int, max_observations: int | None) -> list[dict]:
    data = maybe_truncate(load_series(args.data_dir / "SimARTFIMA11.txt"), max_observations)
    model = ModelSpec(q=1, p=1, tfi_term=True)
    full = fit_full_whittle(data, model, settings)
    summary_csv = args.output_dir / "posterior_summary.csv"
    records = []
    for partition in ["systematic", "sequential"]:
        result = fit_frequency_divide_and_conquer(
            data,
            model,
            settings,
            n_groups=groups,
            partition=partition,
            include_full=False,
        )
        save_draw_summary(summary_csv, f"partition_{partition}", "consensus", model, result.consensus_draws, reference=full.draws)
        records.append({"experiment": "partition", "partition": partition, "groups": groups, "n_obs": int(len(data))})
    return records


def run_time_frequency(args, settings: MCMCSettings, groups: int, max_observations: int | None) -> list[dict]:
    data = maybe_truncate(load_series(args.data_dir / "Vancouver.npy"), max_observations)
    model = ModelSpec(q=1, p=0, tfi_term=False)
    full = fit_full_whittle(data, model, settings)
    summary_csv = args.output_dir / "posterior_summary.csv"
    records = []
    frequency = fit_frequency_divide_and_conquer(
        data,
        model,
        settings,
        n_groups=groups,
        partition="systematic",
        include_full=False,
    )
    time_split = fit_time_domain_as_frequency_shards(data, model, settings, n_groups=groups, include_full=False)
    for label, result in [("frequency", frequency), ("time", time_split)]:
        save_draw_summary(summary_csv, f"time_frequency_{label}", "consensus", model, result.consensus_draws, reference=full.draws)
        records.append({"experiment": "time-frequency", "method": label, "groups": groups, "n_obs": int(len(data))})
    plot_marginals(
        args.output_dir / "vancouver_time_frequency.png",
        "vancouver_time_frequency",
        model,
        full.draws,
        frequency.consensus_draws,
        average_draws=time_split.consensus_draws,
        shard_draws=None,
    )
    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "posterior_summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    settings, groups, max_observations, optimize = settings_from_args(args)
    experiments = args.experiments
    if "all" in experiments:
        experiments = ["combination", "group-size", "partition", "time-frequency"]

    manifest = {
        "preset": args.preset,
        "samples": settings.n_samples,
        "burn_in": settings.burn_in,
        "groups": groups,
        "seed": settings.seed,
        "max_observations": max_observations,
        "optimize": optimize,
        "experiments": experiments,
        "source_manuscript": "docs/Manuscript_2026_Zi.tex",
        "records": [],
    }

    if "combination" in experiments:
        manifest["records"].extend(run_combination(args, settings, groups, max_observations))
    if "group-size" in experiments:
        manifest["records"].extend(run_group_size(args, settings, max_observations))
    if "partition" in experiments:
        manifest["records"].extend(run_partition(args, settings, groups, max_observations))
    if "time-frequency" in experiments:
        manifest["records"].extend(run_time_frequency(args, settings, groups, max_observations))

    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
