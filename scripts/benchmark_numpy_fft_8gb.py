#!/usr/bin/env python3
"""Benchmark NumPy FFTs in isolated processes with a hard memory limit."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import statistics
from pathlib import Path
from typing import Any


DEFAULT_LENGTHS = "67108864,134217728,268435456,536870912"
LCG_MULTIPLIER = 48271
LCG_MODULUS = 2147483647


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return values


def limited_worker(
    connection: Any,
    series_length: int,
    memory_bytes: int,
    warmups: int,
    repetitions: int,
) -> None:
    import gc
    import resource
    import time

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    try:
        import numpy as np

        indices = np.arange(series_length, dtype=np.int64)
        np.multiply(indices, LCG_MULTIPLIER, out=indices)
        np.remainder(indices, LCG_MODULUS, out=indices)
        values = indices.astype(np.float64)
        values /= LCG_MODULUS
        values -= 0.5
        del indices
        gc.collect()

        def run_once() -> float:
            started = time.perf_counter()
            transformed = np.fft.fft(values)
            seconds = time.perf_counter() - started
            if transformed.size != series_length:
                raise RuntimeError("NumPy FFT returned an invalid output size")
            del transformed
            gc.collect()
            return seconds

        for _ in range(warmups):
            run_once()
        times = [run_once() for _ in range(repetitions)]
        peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        connection.send(
            {
                "status": "ok",
                "times": times,
                "peak_rss_gib": peak_rss_kib / (1024**2),
                "numpy_version": np.__version__,
            }
        )
    except MemoryError as error:
        connection.send({"status": "oom", "error": repr(error)})
    except BaseException as error:
        connection.send({"status": "error", "error": repr(error)})
    finally:
        connection.close()


def run_limited_fft(
    series_length: int,
    memory_gib: float,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=limited_worker,
        args=(
            child_connection,
            series_length,
            int(memory_gib * 1024**3),
            warmups,
            repetitions,
        ),
    )
    process.start()
    child_connection.close()
    process.join()

    if parent_connection.poll():
        result = parent_connection.recv()
    else:
        result = {
            "status": "killed",
            "error": f"worker exited without a result (exit code {process.exitcode})",
        }
    parent_connection.close()
    result["exit_code"] = process.exitcode
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=parse_int_list, default=parse_int_list(DEFAULT_LENGTHS))
    parser.add_argument("--memory-gib", type=float, default=8.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("results/numpy_fft_8gb"))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.memory_gib <= 0 or args.warmups < 0 or args.repetitions < 1:
        raise ValueError("invalid memory, warm-up, or repetition setting")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "numpy_fft_8gb_raw.csv"
    summary_path = args.output_dir / "numpy_fft_8gb_summary.csv"
    metadata_path = args.output_dir / "numpy_fft_8gb_metadata.json"
    paths = (raw_path, summary_path, metadata_path)
    if not args.overwrite and any(path.exists() for path in paths):
        raise FileExistsError(f"output exists in {args.output_dir}; pass --overwrite")

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    numpy_version = None
    for series_length in args.lengths:
        print(
            f"engine=numpy T={series_length} memory_gib={args.memory_gib:g}",
            flush=True,
        )
        result = run_limited_fft(
            series_length, args.memory_gib, args.warmups, args.repetitions
        )
        numpy_version = result.get("numpy_version", numpy_version)
        times = result.get("times", [])
        if result["status"] == "ok":
            for repetition, seconds in enumerate(times, start=1):
                raw_rows.append(
                    {
                        "series_length": series_length,
                        "repetition": repetition,
                        "status": "ok",
                        "seconds": seconds,
                        "peak_rss_gib": result["peak_rss_gib"],
                        "error": "",
                    }
                )
                print(
                    f"T={series_length} rep={repetition} seconds={seconds:.3f}",
                    flush=True,
                )
            summary_rows.append(
                {
                    "series_length": series_length,
                    "status": "ok",
                    "runs": len(times),
                    "median_seconds": statistics.median(times),
                    "min_seconds": min(times),
                    "max_seconds": max(times),
                    "peak_rss_gib": result["peak_rss_gib"],
                    "error": "",
                }
            )
        else:
            error = result.get("error", "unknown failure")
            raw_rows.append(
                {
                    "series_length": series_length,
                    "repetition": 0,
                    "status": result["status"],
                    "seconds": "",
                    "peak_rss_gib": "",
                    "error": error,
                }
            )
            summary_rows.append(
                {
                    "series_length": series_length,
                    "status": result["status"],
                    "runs": 0,
                    "median_seconds": "",
                    "min_seconds": "",
                    "max_seconds": "",
                    "peak_rss_gib": "",
                    "error": error,
                }
            )
            print(f"T={series_length} status={result['status']} error={error}", flush=True)

    raw_fields = [
        "series_length",
        "repetition",
        "status",
        "seconds",
        "peak_rss_gib",
        "error",
    ]
    with raw_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=raw_fields)
        writer.writeheader()
        writer.writerows(raw_rows)

    summary_fields = [
        "series_length",
        "status",
        "runs",
        "median_seconds",
        "min_seconds",
        "max_seconds",
        "peak_rss_gib",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "memory_limit_gib": args.memory_gib,
        "limit_type": "RLIMIT_AS",
        "numpy_version": numpy_version,
        "series_lengths": args.lengths,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "threads": 1,
        "timing_boundary": "np.fft.fft(values), with values resident inside the limit",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print(f"Raw timings: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
