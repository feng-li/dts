#!/usr/bin/env python3
"""Create the 8 GiB FFT capacity comparison table and figure."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_rows(paths: list[Path]) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="ascii") as handle:
            for row in csv.DictReader(handle):
                rows[int(row["series_length"])] = row
    return rows


def seconds_text(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def input_size_text(exponent: int) -> str:
    size_bytes = 2**exponent * 8
    if size_bytes >= 1024**3:
        return f"{size_bytes // 1024**3} GiB"
    if size_bytes >= 1024**2:
        return f"{size_bytes // 1024**2} MiB"
    return f"{size_bytes // 1024} KiB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpy-summary", type=Path, action="append", required=True)
    parser.add_argument("--spark-summary", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/dfft_memory_8gb/combined")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    numpy_rows = load_rows(args.numpy_summary)
    spark_rows = load_rows(args.spark_summary)
    lengths = sorted(set(numpy_rows) | set(spark_rows))
    if set(lengths) != set(numpy_rows) or set(lengths) != set(spark_rows):
        raise ValueError("NumPy and Spark summaries must contain identical series lengths")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "dfft_memory_8gb_table.csv"
    tex_path = args.output_dir / "dfft_memory_8gb_table.tex"
    pdf_path = args.output_dir / "dfft_memory_8gb_walltime.pdf"
    png_path = args.output_dir / "dfft_memory_8gb_walltime.png"

    table_rows: list[dict[str, Any]] = []
    for length in lengths:
        exponent = int(math.log2(length))
        if 2**exponent != length:
            raise ValueError(f"series length is not a power of two: {length}")
        numpy_row = numpy_rows[length]
        spark_row = spark_rows[length]
        numpy_status = numpy_row.get("status", "ok")
        numpy_seconds = (
            float(numpy_row["median_seconds"]) if numpy_status == "ok" else None
        )
        spark_seconds = float(spark_row["median_seconds"])
        table_rows.append(
            {
                "log2_T": exponent,
                "series_length": length,
                "numpy_status": numpy_status,
                "numpy_median_seconds": "" if numpy_seconds is None else numpy_seconds,
                "spark_median_seconds": spark_seconds,
                "numpy_time_over_spark_time": (
                    "" if numpy_seconds is None else numpy_seconds / spark_seconds
                ),
            }
        )

    fields = [
        "log2_T",
        "series_length",
        "numpy_status",
        "numpy_median_seconds",
        "spark_median_seconds",
        "numpy_time_over_spark_time",
    ]
    with csv_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Wall-clock time for a single-process NumPy FFT limited to 8 GiB and the block-record Spark FFT with 256 partitions on 64 one-core, 8 GiB workers. Entries are medians of three runs after one warm-up.}",
        r"\label{tab:dfft-memory-8gb}",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$T$ & NumPy (s) & Spark (s) & NumPy/Spark \\",
        r"\midrule",
    ]
    for row in table_rows:
        if row["numpy_status"] == "ok":
            numpy_value = seconds_text(float(row["numpy_median_seconds"]))
            ratio = f"{float(row['numpy_time_over_spark_time']):.3f}"
        else:
            numpy_value = r"\textsc{oom}"
            ratio = "--"
        tex_lines.append(
            f"$2^{{{row['log2_T']}}}$ & {numpy_value} & "
            f"{seconds_text(float(row['spark_median_seconds']))} & {ratio} \\\\"
        )
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="ascii")

    exponents = [int(row["log2_T"]) for row in table_rows]
    series_lengths = [int(row["series_length"]) for row in table_rows]
    spark_medians = [float(row["spark_median_seconds"]) for row in table_rows]
    numpy_lengths = [
        int(row["series_length"])
        for row in table_rows
        if row["numpy_status"] == "ok"
    ]
    numpy_medians = [
        float(row["numpy_median_seconds"])
        for row in table_rows
        if row["numpy_status"] == "ok"
    ]
    oom_lengths = [
        int(row["series_length"])
        for row in table_rows
        if row["numpy_status"] != "ok"
    ]

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
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "xtick.color": "black",
            "ytick.color": "black",
            "legend.fontsize": 12,
            "legend.labelcolor": "black",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    axis.plot(
        series_lengths,
        spark_medians,
        color="darkblue",
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        label=r"Spark block FFT, $P=256$",
    )
    axis.plot(
        numpy_lengths,
        numpy_medians,
        color="#D62728",
        marker="s",
        linewidth=2.2,
        markersize=5.0,
        label="NumPy FFT, 8 GiB limit",
    )
    if oom_lengths:
        first_oom = min(oom_lengths)
        axis.axvspan(
            first_oom / math.sqrt(2.0),
            max(series_lengths) * math.sqrt(2.0),
            color="#D62728",
            alpha=0.08,
            linewidth=0,
        )

    axis.set_yscale("log")
    axis.set_xscale("log", base=10)
    axis.set_xlim(
        min(series_lengths) / math.sqrt(2.0),
        max(series_lengths) * math.sqrt(2.0),
    )
    decade_exponents = range(
        math.floor(math.log10(min(series_lengths))),
        math.floor(math.log10(max(series_lengths))) + 1,
    )
    decade_exponents = list(decade_exponents)
    axis.set_xticks(
        [10**value for value in decade_exponents] + [max(series_lengths)]
    )
    axis.set_xticklabels(
        [rf"$10^{{{value}}}$" for value in decade_exponents]
        + [r"$2.68\times10^8$"]
    )
    axis.set_xlabel(r"$T$")
    size_exponents = [
        value for value in exponents if value % 2 == 0 or value >= max(exponents) - 2
    ]
    size_axis = axis.secondary_xaxis("top")
    size_axis.set_xticks([2**value for value in size_exponents])
    size_axis.set_xticklabels([input_size_text(value) for value in size_exponents])
    axis.set_ylabel("Wall-clock time (seconds, log scale)")
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="both", direction="out", length=4, width=0.8)
    size_axis.tick_params(
        axis="x", direction="out", length=4, width=0.8, labelsize=8
    )
    axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
    )
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"Table CSV: {csv_path}")
    print(f"Table TeX: {tex_path}")
    print(f"Figure PDF: {pdf_path}")
    print(f"Figure PNG: {png_path}")


if __name__ == "__main__":
    main()
