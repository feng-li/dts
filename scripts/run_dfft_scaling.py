#!/usr/bin/env python3
"""Run the fixed-T, fixed-P DFFT task-slot scaling experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "scripts" / "benchmark_dfft_vs_numpy.py"
FFT_PARTITIONS = 256
CONCURRENT_TASK_SLOTS = (4, 8, 16, 32, 52, 104)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=2**28)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--spark-warmups", type=int, default=1)
    parser.add_argument("--driver-memory", default="128g")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "dfft_scaling_t28",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    design = {
        "series_length": args.length,
        "repetitions": args.repetitions,
        "spark_warmups": args.spark_warmups,
        "driver_memory": args.driver_memory,
        "fft_partitions": FFT_PARTITIONS,
        "concurrent_task_slots": list(CONCURRENT_TASK_SLOTS),
        "timing_scope": (
            "Spark FFT action after input construction and materialisation; "
            "Spark context startup is excluded"
        ),
    }
    (output_dir / "scaling_design.json").write_text(
        json.dumps(design, indent=2) + "\n", encoding="ascii"
    )

    environment = os.environ.copy()
    environment["SPARK_LOCAL_HOSTNAME"] = "localhost"
    environment["SPARK_LOCAL_IP"] = "127.0.0.1"
    environment["PYSPARK_PYTHON"] = args.python
    environment["PYSPARK_DRIVER_PYTHON"] = args.python
    environment["PYSPARK_SUBMIT_ARGS"] = (
        f"--driver-memory {args.driver_memory} pyspark-shell"
    )

    for task_slots in CONCURRENT_TASK_SLOTS:
        run_dir = output_dir / f"p{FFT_PARTITIONS:03d}_c{task_slots:03d}"
        command = [
            args.python,
            str(BENCHMARK),
            "--master",
            f"local[{task_slots}]",
            "--lengths",
            str(args.length),
            "--partitions",
            str(FFT_PARTITIONS),
            "--repetitions",
            str(args.repetitions),
            "--spark-warmups",
            str(args.spark_warmups),
            "--skip-numpy",
            "--output-dir",
            str(run_dir),
        ]
        if args.overwrite:
            command.append("--overwrite")

        print(
            f"Running T={args.length}, P={FFT_PARTITIONS}, "
            f"master=local[{task_slots}]",
            flush=True,
        )
        subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
