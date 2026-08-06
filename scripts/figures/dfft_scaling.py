#!/usr/bin/env python3
"""Create the fixed-T DFFT scaling table and figure."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "dfft_scaling_t28",
    )
    return parser


def read_spark_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["engine"].startswith("spark")]
    if len(rows) != 1:
        raise ValueError(f"Expected one Spark summary row in {path}, found {len(rows)}")
    return rows[0]


def main() -> None:
    args = build_parser().parse_args()
    results_dir = args.results_dir.resolve()
    design = json.loads((results_dir / "scaling_design.json").read_text())
    series_length = int(design["series_length"])

    rows: list[dict[str, Any]] = []
    for configuration in design["configurations"]:
        partitions = int(configuration["partitions"])
        cores = int(configuration["local_cores"])
        run_dir = results_dir / f"p{partitions:03d}_c{cores:03d}"
        summary = read_spark_row(run_dir / "dfft_vs_numpy_summary.csv")
        if int(summary["series_length"]) != series_length:
            raise ValueError(f"Unexpected series length in {run_dir}")
        rows.append(
            {
                "partitions": partitions,
                "local_cores": cores,
                "runs": int(summary["runs"]),
                "median_seconds": float(summary["median_seconds"]),
            }
        )

    rows.sort(key=lambda row: row["partitions"])
    baseline_time = rows[0]["median_seconds"]
    baseline_partitions = rows[0]["partitions"]
    baseline_cores = rows[0]["local_cores"]
    for row in rows:
        row["observed_speedup"] = baseline_time / row["median_seconds"]
        row["partition_ratio"] = row["partitions"] / baseline_partitions
        row["core_ratio"] = row["local_cores"] / baseline_cores
        row["partition_efficiency"] = (
            row["observed_speedup"] / row["partition_ratio"]
        )
        row["core_efficiency"] = row["observed_speedup"] / row["core_ratio"]

    csv_path = results_dir / "dfft_scaling_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "partitions",
            "local_cores",
            "runs",
            "median_seconds",
            "observed_speedup",
            "partition_ratio",
            "partition_efficiency",
            "core_ratio",
            "core_efficiency",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    exponent = int(math.log2(series_length))
    if 2**exponent != series_length:
        raise ValueError("The series length must be a power of two")
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        (
            rf"\caption{{Coupled partition/core scaling of the block-record Spark FFT at "
            rf"$T=2^{{{exponent}}}$. Speed-ups are relative to $P_0={baseline_partitions}$ "
            rf"with {baseline_cores} local task slots. Entries are medians of "
            rf"{rows[0]['runs']} timed runs after one warm-up.}}"
        ),
        r"\label{tab:dfft-scaling}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        "$P$ & Local cores & Time (s) & Observed & $P/P_0$ & Achieved (\\%) \\\\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['partitions']} & {row['local_cores']} & "
            f"{row['median_seconds']:.3f} & {row['observed_speedup']:.2f} & "
            f"{row['partition_ratio']:.1f} & "
            f"{100.0 * row['partition_efficiency']:.1f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (results_dir / "dfft_scaling_table.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="ascii"
    )

    partitions = [row["partitions"] for row in rows]
    times = [row["median_seconds"] for row in rows]
    observed = [row["observed_speedup"] for row in rows]
    partition_ratio = [row["partition_ratio"] for row in rows]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 13,
            "text.color": "black",
            "axes.labelsize": 15,
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.fontsize": 12,
            "legend.labelcolor": "black",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    axes[0].plot(partitions, times, color="darkblue", marker="o", linewidth=2.2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(partitions)
    axes[0].set_xticklabels([str(value) for value in partitions])
    axes[0].set_xlabel("Logical DFFT partitions, $P$")
    axes[0].set_ylabel("Median wall-clock time (seconds)")
    axes[0].set_title("Wall-clock time", loc="left")

    axes[1].plot(
        partitions,
        observed,
        color="darkblue",
        marker="o",
        linewidth=2.4,
        label="Observed",
    )
    axes[1].plot(
        partitions,
        partition_ratio,
        color="#D62728",
        linestyle="--",
        linewidth=1.8,
        label=r"Partition reference, $P/P_0$",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xticks(partitions)
    axes[1].set_xticklabels([str(value) for value in partitions])
    axes[1].set_xlabel("Logical DFFT partitions, $P$")
    axes[1].set_ylabel("Speed-up relative to baseline")
    axes[1].set_title("Observed versus nominal scaling", loc="left")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", direction="out", length=4, width=0.8)
    figure.suptitle(rf"Distributed FFT scaling at $T=2^{{{exponent}}}$", x=0.06, ha="left")
    figure.tight_layout()
    figure.savefig(results_dir / "dfft_scaling.pdf", bbox_inches="tight")
    figure.savefig(results_dir / "dfft_scaling.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
