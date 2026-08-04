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
        r"\caption{Wall-clock time for a single-process NumPy FFT limited to 8 GiB and the block-record Spark FFT with 128 partitions on 64 one-core, 8 GiB workers. Entries are medians of three runs after one warm-up.}",
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
    spark_medians = [float(row["spark_median_seconds"]) for row in table_rows]
    numpy_exponents = [
        int(row["log2_T"]) for row in table_rows if row["numpy_status"] == "ok"
    ]
    numpy_medians = [
        float(row["numpy_median_seconds"])
        for row in table_rows
        if row["numpy_status"] == "ok"
    ]
    oom_exponents = [
        int(row["log2_T"]) for row in table_rows if row["numpy_status"] != "ok"
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.titleweight": "bold",
            "axes.edgecolor": "#263238",
            "axes.labelcolor": "#263238",
            "xtick.color": "#263238",
            "ytick.color": "#263238",
        }
    )
    figure, axis = plt.subplots(figsize=(9.2, 5.8), facecolor="#f4f0e6")
    axis.set_facecolor("#fbf9f3")
    axis.plot(
        exponents,
        spark_medians,
        color="#0b5d6b",
        marker="o",
        linewidth=2.4,
        markersize=5.5,
        label="Spark block FFT, P=128",
    )
    axis.plot(
        numpy_exponents,
        numpy_medians,
        color="#c7462d",
        marker="s",
        linewidth=2.2,
        markersize=5.0,
        label="NumPy FFT, 8 GiB limit",
    )
    if oom_exponents:
        first_oom = min(oom_exponents)
        axis.axvspan(
            first_oom - 0.45,
            max(exponents) + 0.45,
            color="#c7462d",
            alpha=0.10,
            linewidth=0,
        )
        axis.text(
            first_oom + 0.1,
            max(spark_medians) * 0.55,
            "NumPy exceeds\n8 GiB",
            color="#8f2f20",
            fontsize=10,
            fontweight="bold",
        )

    axis.set_yscale("log")
    axis.set_xlim(min(exponents) - 0.5, max(exponents) + 0.5)
    axis.set_xticks(exponents)
    axis.set_xticklabels([rf"$2^{{{value}}}$" for value in exponents], rotation=45)
    axis.set_xlabel("Series length T")
    axis.set_ylabel("Median wall-clock time (seconds, log scale)")
    axis.set_title("Distributed FFT capacity beyond one 8 GiB worker", loc="left")
    axis.grid(which="both", axis="y", color="#9aa4a6", alpha=0.28, linewidth=0.7)
    axis.legend(frameon=False, loc="upper left")
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
