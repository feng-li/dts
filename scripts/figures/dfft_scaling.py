#!/usr/bin/env python3
"""Create the fixed-T, fixed-P DFFT task-slot scaling table and figure."""

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
    partitions = int(design["fft_partitions"])

    rows: list[dict[str, Any]] = []
    for task_slots in design["concurrent_task_slots"]:
        task_slots = int(task_slots)
        run_dir = results_dir / f"p{partitions:03d}_c{task_slots:03d}"
        summary = read_spark_row(run_dir / "dfft_vs_numpy_summary.csv")
        if int(summary["series_length"]) != series_length:
            raise ValueError(f"Unexpected series length in {run_dir}")
        if int(summary["partitions"]) != partitions:
            raise ValueError(f"Unexpected partition count in {run_dir}")
        rows.append(
            {
                "partitions": partitions,
                "concurrent_task_slots": task_slots,
                "runs": int(summary["runs"]),
                "median_seconds": float(summary["median_seconds"]),
            }
        )

    rows.sort(key=lambda row: row["concurrent_task_slots"])
    baseline_time = rows[0]["median_seconds"]
    baseline_task_slots = rows[0]["concurrent_task_slots"]
    for row in rows:
        row["observed_speedup"] = baseline_time / row["median_seconds"]
        row["nominal_speedup"] = (
            row["concurrent_task_slots"] / baseline_task_slots
        )
        row["parallel_efficiency"] = (
            row["observed_speedup"] / row["nominal_speedup"]
        )

    csv_path = results_dir / "dfft_scaling_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "partitions",
            "concurrent_task_slots",
            "runs",
            "median_seconds",
            "observed_speedup",
            "nominal_speedup",
            "parallel_efficiency",
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
            rf"\caption{{Concurrent-task-slot scaling of the block-record Spark FFT at "
            rf"$T=2^{{{exponent}}}$ with $P={partitions}$ fixed. Speed-ups are "
            rf"relative to $C_0={baseline_task_slots}$ task slots. Entries are medians of "
            rf"{rows[0]['runs']} timed runs after one warm-up.}}"
        ),
        r"\label{tab:dfft-scaling}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        "$C$ & Time (s) & Observed & $C/C_0$ & Achieved (\\%) \\\\",
        r"\midrule",
    ]
    for row in rows:
        tex_lines.append(
            f"{row['concurrent_task_slots']} & {row['median_seconds']:.3f} & "
            f"{row['observed_speedup']:.2f} & {row['nominal_speedup']:.1f} & "
            f"{100.0 * row['parallel_efficiency']:.1f} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (results_dir / "dfft_scaling_table.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="ascii"
    )

    task_slots = [row["concurrent_task_slots"] for row in rows]
    times = [row["median_seconds"] for row in rows]
    observed = [row["observed_speedup"] for row in rows]
    nominal = [row["nominal_speedup"] for row in rows]

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
    axes[0].plot(task_slots, times, color="darkblue", marker="o", linewidth=2.2)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(task_slots)
    axes[0].set_xticklabels([str(value) for value in task_slots])
    axes[0].set_xlabel("Concurrent Spark task slots, $C$")
    axes[0].set_ylabel("Wall-clock time (seconds)")

    axes[1].plot(
        task_slots,
        observed,
        color="darkblue",
        marker="o",
        linewidth=2.4,
        label="Observed",
    )
    axes[1].plot(
        task_slots,
        nominal,
        color="#D62728",
        linestyle="--",
        linewidth=1.8,
        label=r"Ideal linear, $C/C_0$",
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log", base=2)
    axes[1].set_xticks(task_slots)
    axes[1].set_xticklabels([str(value) for value in task_slots])
    axes[1].set_xlabel("Concurrent Spark task slots, $C$")
    axes[1].set_ylabel("Speed-up relative to baseline")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", direction="out", length=4, width=0.8)
    figure.tight_layout()
    figure.savefig(results_dir / "dfft_scaling.pdf", bbox_inches="tight")
    figure.savefig(results_dir / "dfft_scaling.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
