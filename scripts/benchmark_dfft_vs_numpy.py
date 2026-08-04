#!/usr/bin/env python3
"""Strong-scaling benchmark for the distributed FFT versus NumPy.

The default experiment fixes the FFT parallelism at P = 128, uses 64
concurrent Spark task slots, and varies T from 2^20 through 2^26. The Spark
tasks therefore deliberately oversubscribe the available CPU cores by a
factor of two. Input construction and materialization are excluded from both
timings.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyspark
from pyspark import StorageLevel
from pyspark.sql import SparkSession

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dts.dfft_blocks import spark_fft_contiguous_blocks


DEFAULT_LENGTHS = "1048576,4194304,16777216,67108864"  # 2^20, 2^22, 2^24, 2^26
DEFAULT_PARTITIONS = "128"
LCG_MULTIPLIER = 48271
LCG_MODULUS = 2147483647


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def numpy_input(series_length: int) -> np.ndarray:
    indices = np.arange(series_length, dtype=np.int64)
    np.multiply(indices, LCG_MULTIPLIER, out=indices)
    np.remainder(indices, LCG_MODULUS, out=indices)
    values = indices.astype(np.float64)
    values /= LCG_MODULUS
    values -= 0.5
    return values


def spark_input_block(block_id: int, block_size: int) -> tuple[int, np.ndarray]:
    start = block_id * block_size
    indices = np.arange(start, start + block_size, dtype=np.int64)
    np.multiply(indices, LCG_MULTIPLIER, out=indices)
    np.remainder(indices, LCG_MODULUS, out=indices)
    values = indices.astype(np.float64)
    values /= LCG_MODULUS
    values -= 0.5
    return block_id, values


def timed_numpy_fft(values: np.ndarray) -> float:
    gc.collect()
    started = time.perf_counter()
    transformed = np.fft.fft(values)
    elapsed = time.perf_counter() - started
    if transformed.size != values.size:
        raise RuntimeError("NumPy FFT returned an unexpected number of coefficients")
    del transformed
    return elapsed


def timed_spark_fft(block_rdd: Any, series_length: int, partitions: int) -> float:
    started = time.perf_counter()
    transformed = spark_fft_contiguous_blocks(block_rdd, series_length, partitions)
    output_count = transformed.values().map(lambda block: int(block[1].size)).sum()
    elapsed = time.perf_counter() - started
    if output_count != series_length:
        raise RuntimeError(
            f"Spark FFT returned {output_count} coefficients; expected {series_length}"
        )
    return elapsed


def visible_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def write_results(
    output_dir: Path,
    raw_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "dfft_vs_numpy_raw.csv"
    summary_path = output_dir / "dfft_vs_numpy_summary.csv"
    metadata_path = output_dir / "dfft_vs_numpy_metadata.json"
    paths = (raw_path, summary_path, metadata_path)
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError(f"output exists in {output_dir}; pass --overwrite to replace it")

    raw_fields = ["engine", "series_length", "partitions", "repetition", "seconds"]
    with raw_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in raw_rows:
        key = (row["engine"], row["series_length"], row["partitions"])
        grouped.setdefault(key, []).append(row["seconds"])

    numpy_medians = {
        length: statistics.median(seconds)
        for (engine, length, _), seconds in grouped.items()
        if engine == "numpy"
    }
    spark_p1_medians = {
        length: statistics.median(seconds)
        for (engine, length, partitions), seconds in grouped.items()
        if engine == "spark" and partitions == 1
    }

    summary_rows: list[dict[str, Any]] = []
    for (engine, length, partitions), seconds in sorted(grouped.items()):
        median_seconds = statistics.median(seconds)
        spark_speedup = ""
        numpy_over_engine = ""
        if engine == "spark" and length in spark_p1_medians:
            spark_speedup = spark_p1_medians[length] / median_seconds
        if length in numpy_medians:
            numpy_over_engine = numpy_medians[length] / median_seconds
        summary_rows.append(
            {
                "engine": engine,
                "series_length": length,
                "partitions": partitions,
                "runs": len(seconds),
                "median_seconds": median_seconds,
                "min_seconds": min(seconds),
                "max_seconds": max(seconds),
                "spark_speedup_vs_p1": spark_speedup,
                "numpy_time_over_engine_time": numpy_over_engine,
            }
        )

    summary_fields = [
        "engine",
        "series_length",
        "partitions",
        "runs",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "spark_speedup_vs_p1",
        "numpy_time_over_engine_time",
    ]
    with summary_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--master", default="local[64]", help="Spark master (default: local[64])"
    )
    parser.add_argument("--lengths", type=parse_int_list, default=parse_int_list(DEFAULT_LENGTHS))
    parser.add_argument(
        "--partitions", type=parse_int_list, default=parse_int_list(DEFAULT_PARTITIONS)
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--spark-warmups", type=int, default=0)
    parser.add_argument("--numpy-warmups", type=int, default=1)
    parser.add_argument("--skip-numpy", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/dfft_vs_numpy"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions < 1 or args.spark_warmups < 0 or args.numpy_warmups < 0:
        raise ValueError("repetitions must be positive and warm-up counts must be non-negative")
    for length in args.lengths:
        if not is_power_of_two(length):
            raise ValueError(f"series length must be a power of two: {length}")
        for partitions in args.partitions:
            if not is_power_of_two(partitions) or length % partitions:
                raise ValueError(
                    f"P must be a power of two that divides T: T={length}, P={partitions}"
                )

    builder = SparkSession.builder.appName("dfft-vs-numpy")
    if args.master:
        builder = builder.master(args.master)
    spark = builder.config("spark.ui.enabled", "false").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    raw_rows: list[dict[str, Any]] = []
    metadata = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pyspark_version": pyspark.__version__,
        "spark_version": spark.version,
        "spark_master": spark.sparkContext.master,
        "spark_default_parallelism": spark.sparkContext.defaultParallelism,
        "visible_cpu_count": visible_cpu_count(),
        "series_lengths": args.lengths,
        "partitions": args.partitions,
        "repetitions": args.repetitions,
        "spark_warmups": args.spark_warmups,
        "numpy_warmups": args.numpy_warmups,
        "signal": "((index * 48271) % 2147483647) / 2147483647 - 0.5",
        "timing_boundary": {
            "numpy": "np.fft.fft(values), with values already resident in driver memory",
            "spark": "spark_fft_contiguous_blocks(...), with contiguous array blocks persisted",
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }

    try:
        for length in args.lengths:
            if not args.skip_numpy:
                print(f"Preparing NumPy input T={length}", flush=True)
                values = numpy_input(length)
                for _ in range(args.numpy_warmups):
                    timed_numpy_fft(values)
                for repetition in range(1, args.repetitions + 1):
                    seconds = timed_numpy_fft(values)
                    raw_rows.append(
                        {
                            "engine": "numpy",
                            "series_length": length,
                            "partitions": 0,
                            "repetition": repetition,
                            "seconds": seconds,
                        }
                    )
                    print(
                        f"engine=numpy T={length} rep={repetition} seconds={seconds:.3f}",
                        flush=True,
                    )
                del values
                gc.collect()

            for partitions in args.partitions:
                print(f"Preparing Spark input T={length} P={partitions}", flush=True)
                block_size = length // partitions
                block_rdd = (
                    spark.sparkContext.parallelize(range(partitions), partitions)
                    .map(lambda block_id: spark_input_block(block_id, block_size))
                    .persist(StorageLevel.MEMORY_AND_DISK)
                )
                input_count = block_rdd.values().map(lambda block: int(block.size)).sum()
                if input_count != length:
                    raise RuntimeError(f"Spark input has {input_count} rows; expected {length}")
                for _ in range(args.spark_warmups):
                    timed_spark_fft(block_rdd, length, partitions)
                for repetition in range(1, args.repetitions + 1):
                    seconds = timed_spark_fft(block_rdd, length, partitions)
                    raw_rows.append(
                        {
                            "engine": "spark",
                            "series_length": length,
                            "partitions": partitions,
                            "repetition": repetition,
                            "seconds": seconds,
                        }
                    )
                    print(
                        f"engine=spark T={length} P={partitions} "
                        f"rep={repetition} seconds={seconds:.3f}",
                        flush=True,
                    )
                block_rdd.unpersist(blocking=True)
    finally:
        spark.stop()

    paths = write_results(args.output_dir, raw_rows, metadata, args.overwrite)
    print(f"Raw timings: {paths[0]}")
    print(f"Summary: {paths[1]}")
    print(f"Metadata: {paths[2]}")


if __name__ == "__main__":
    main()
