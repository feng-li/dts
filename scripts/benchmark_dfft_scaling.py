#!/usr/bin/env python3
"""Benchmark strong scaling of the Spark distributed FFT without MCMC."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dts.dfft import spark_fft_indexed_rdd


RAW_FIELDS = [
    "timestamp_utc",
    "series_length",
    "fft_partitions",
    "repetition",
    "wall_seconds",
    "output_count",
    "checksum",
    "throughput_obs_per_second",
    "status",
    "error",
]


def parse_integer_list(value: str, label: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError(f"{label} must contain positive integers")
    return list(dict.fromkeys(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        default="1048576,4194304,16777216",
        help="comma-separated series lengths T",
    )
    parser.add_argument(
        "--partitions",
        default="1,2,4,8,16,32,64",
        help="comma-separated power-of-two DFFT partition counts P",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--storage-level",
        choices=["memory-and-disk", "memory-only", "disk-only"],
        default="memory-and-disk",
        help="persistence used for the untimed generated input",
    )
    parser.add_argument("--master", default=None, help="optional Spark master URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/dfft_scaling"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_design(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    lengths = parse_integer_list(args.lengths, "--lengths")
    partitions = parse_integer_list(args.partitions, "--partitions")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be nonnegative")
    for partition_count in partitions:
        if partition_count & (partition_count - 1):
            raise ValueError(f"P={partition_count} is not a power of two")
    invalid = [
        (length, partition_count)
        for length in lengths
        for partition_count in partitions
        if length % partition_count
    ]
    if invalid:
        length, partition_count = invalid[0]
        raise ValueError(f"T={length} is not divisible by P={partition_count}")
    return lengths, partitions


def deterministic_value(index: int) -> float:
    """Bounded deterministic signal generated independently on executors."""

    phase = index % 1048576
    return (
        math.sin(2.0 * math.pi * phase / 1024.0)
        + 0.5 * math.cos(2.0 * math.pi * phase / 4096.0)
        + 0.125 * math.sin(2.0 * math.pi * phase / 16384.0)
    )


def storage_level(name: str):
    from pyspark import StorageLevel

    return {
        "memory-and-disk": StorageLevel.MEMORY_AND_DISK,
        "memory-only": StorageLevel.MEMORY_ONLY,
        "disk-only": StorageLevel.DISK_ONLY,
    }[name]


def generated_indexed_rdd(spark, length: int, partitions: int, persistence):
    indexed = (
        spark.range(0, length, numPartitions=partitions)
        .rdd.map(lambda row: (int(row[0]), deterministic_value(int(row[0]))))
        .persist(persistence)
    )
    materialized_count = indexed.count()
    if materialized_count != length:
        raise RuntimeError(
            f"generated input contains {materialized_count} rows; expected {length}"
        )
    return indexed


def evaluate_fft(indexed, length: int, partitions: int) -> tuple[int, float]:
    fft_rdd = spark_fft_indexed_rdd(indexed, length, partitions)

    def sequence(accumulator, item):
        value = item[1]
        return (
            accumulator[0] + 1,
            accumulator[1] + value.real * value.real + value.imag * value.imag,
        )

    def combine(left, right):
        return left[0] + right[0], left[1] + right[1]

    return fft_rdd.treeAggregate((0, 0.0), sequence, combine, depth=3)


def append_result(path: Path, row: dict) -> None:
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def write_summary(raw_path: Path, summary_path: Path) -> None:
    with raw_path.open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    successful = [row for row in raw if row["status"] == "ok"]
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in successful:
        key = int(row["series_length"]), int(row["fft_partitions"])
        grouped.setdefault(key, []).append(float(row["wall_seconds"]))

    medians = {key: float(np.median(values)) for key, values in grouped.items()}
    baseline_by_length = {}
    for length, partition_count in sorted(medians):
        baseline_by_length.setdefault(length, partition_count)

    fields = [
        "series_length",
        "fft_partitions",
        "repetitions",
        "median_seconds",
        "q25_seconds",
        "q75_seconds",
        "median_throughput_obs_per_second",
        "baseline_partitions",
        "speedup",
        "parallel_efficiency",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (length, partition_count), values in sorted(grouped.items()):
            baseline_partitions = baseline_by_length[length]
            baseline_seconds = medians[(length, baseline_partitions)]
            median_seconds = medians[(length, partition_count)]
            speedup = baseline_seconds / median_seconds
            relative_partitions = partition_count / baseline_partitions
            writer.writerow(
                {
                    "series_length": length,
                    "fft_partitions": partition_count,
                    "repetitions": len(values),
                    "median_seconds": median_seconds,
                    "q25_seconds": float(np.percentile(values, 25)),
                    "q75_seconds": float(np.percentile(values, 75)),
                    "median_throughput_obs_per_second": length / median_seconds,
                    "baseline_partitions": baseline_partitions,
                    "speedup": speedup,
                    "parallel_efficiency": speedup / relative_partitions,
                }
            )


def selected_spark_metadata(spark, args: argparse.Namespace) -> dict:
    conf = spark.sparkContext.getConf()

    def get(key: str):
        return conf.get(key, None)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spark_version": spark.version,
        "spark_master": spark.sparkContext.master,
        "spark_app_id": spark.sparkContext.applicationId,
        "spark_default_parallelism": spark.sparkContext.defaultParallelism,
        "spark_executor_instances": get("spark.executor.instances"),
        "spark_executor_cores": get("spark.executor.cores"),
        "spark_executor_memory": get("spark.executor.memory"),
        "spark_driver_memory": get("spark.driver.memory"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "benchmark_arguments": vars(args) | {"output_dir": str(args.output_dir)},
        "timed_region": "spark_fft_indexed_rdd plus output count/checksum action",
        "excluded_region": "deterministic Spark input generation and persistence",
    }


def main() -> None:
    from pyspark.sql import SparkSession

    args = parse_args()
    lengths, partitions = validate_design(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "dfft_scaling_raw.csv"
    summary_path = args.output_dir / "dfft_scaling_summary.csv"
    metadata_path = args.output_dir / "dfft_scaling_metadata.json"
    if raw_path.exists() and not args.overwrite:
        raise FileExistsError(f"{raw_path} exists; pass --overwrite to replace it")
    if args.overwrite:
        raw_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    builder = SparkSession.builder.appName("dts-dfft-strong-scaling")
    if args.master:
        builder = builder.master(args.master)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    persistence = storage_level(args.storage_level)

    try:
        with metadata_path.open("w") as handle:
            json.dump(selected_spark_metadata(spark, args), handle, indent=2)

        for length in lengths:
            for partition_count in partitions:
                print(f"Preparing T={length} P={partition_count}", flush=True)
                indexed = generated_indexed_rdd(
                    spark,
                    length,
                    partition_count,
                    persistence,
                )
                try:
                    for warmup in range(args.warmups):
                        spark.sparkContext.setJobGroup(
                            f"warmup-T{length}-P{partition_count}-{warmup + 1}",
                            f"DFFT warmup T={length} P={partition_count}",
                        )
                        evaluate_fft(indexed, length, partition_count)

                    for repetition in range(1, args.repetitions + 1):
                        spark.sparkContext.setJobGroup(
                            f"timed-T{length}-P{partition_count}-{repetition}",
                            f"DFFT timed T={length} P={partition_count} repetition={repetition}",
                        )
                        started = time.perf_counter()
                        try:
                            output_count, checksum = evaluate_fft(
                                indexed,
                                length,
                                partition_count,
                            )
                            elapsed = time.perf_counter() - started
                            if output_count != length:
                                raise RuntimeError(
                                    f"FFT output has {output_count} rows; expected {length}"
                                )
                            row = {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "series_length": length,
                                "fft_partitions": partition_count,
                                "repetition": repetition,
                                "wall_seconds": elapsed,
                                "output_count": output_count,
                                "checksum": checksum,
                                "throughput_obs_per_second": length / elapsed,
                                "status": "ok",
                                "error": "",
                            }
                            print(
                                f"T={length} P={partition_count} rep={repetition} "
                                f"seconds={elapsed:.3f} checksum={checksum:.6e}",
                                flush=True,
                            )
                        except Exception as error:
                            elapsed = time.perf_counter() - started
                            row = {
                                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                                "series_length": length,
                                "fft_partitions": partition_count,
                                "repetition": repetition,
                                "wall_seconds": elapsed,
                                "output_count": "",
                                "checksum": "",
                                "throughput_obs_per_second": "",
                                "status": "error",
                                "error": f"{type(error).__name__}: {error}",
                            }
                            print(row["error"], file=sys.stderr, flush=True)
                        append_result(raw_path, row)
                finally:
                    indexed.unpersist(blocking=True)
        write_summary(raw_path, summary_path)
    finally:
        spark.stop()

    print(f"Raw timings: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
