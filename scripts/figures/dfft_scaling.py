#!/usr/bin/env python3
"""Plot wall-clock time and speed-up from the DFFT scaling benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/dfft_scaling/dfft_scaling_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dfft_scaling/dfft_scaling.pdf"),
    )
    return parser.parse_args()


def length_label(length: int) -> str:
    exponent = np.log2(length)
    if np.isclose(exponent, round(exponent)):
        return rf"$T=2^{{{int(round(exponent))}}}$"
    return rf"$T={length:,}$"


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    required = {
        "series_length",
        "fft_partitions",
        "median_seconds",
        "q25_seconds",
        "q75_seconds",
        "baseline_partitions",
        "speedup",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    colors = ["#0B6E75", "#D95D39", "#334E68", "#C49A00", "#7A5195"]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    for color, (length, group) in zip(colors, data.groupby("series_length", sort=True)):
        group = group.sort_values("fft_partitions")
        partitions = group["fft_partitions"].to_numpy()
        seconds = group["median_seconds"].to_numpy()
        lower = seconds - group["q25_seconds"].to_numpy()
        upper = group["q75_seconds"].to_numpy() - seconds
        label = length_label(int(length))
        axes[0].errorbar(
            partitions,
            seconds,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.7,
            capsize=2.5,
            color=color,
            label=label,
        )
        axes[1].plot(
            partitions,
            group["speedup"],
            marker="o",
            linewidth=1.7,
            color=color,
            label=label,
        )

    baseline = int(data["baseline_partitions"].min())
    all_partitions = np.sort(data["fft_partitions"].unique())
    axes[1].plot(
        all_partitions,
        all_partitions / baseline,
        linestyle="--",
        color="#555555",
        linewidth=1.1,
        label="Ideal",
    )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(all_partitions)
        axis.set_xticklabels([str(value) for value in all_partitions])
        axis.set_xlabel("DFFT partitions, $P$")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Wall-clock time (seconds)")
    axes[0].set_title("Strong-scaling runtime", loc="left", fontweight="bold")
    axes[1].set_ylabel("Speed-up")
    axes[1].set_title("Speed-up relative to baseline", loc="left", fontweight="bold")
    axes[1].legend(frameon=False, ncol=1, fontsize=8)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
