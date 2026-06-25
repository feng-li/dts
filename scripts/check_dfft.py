#!/usr/bin/env python3
"""Check the Spark DFFT implementation against numpy.fft."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from dts.dfft import spark_fft_rdd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=160)
    parser.add_argument("--partitions", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    from pyspark.sql import SparkSession

    args = parse_args()
    rng = np.random.default_rng(123)
    data = rng.normal(size=args.n).tolist()
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("dts-dfft-check")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    rdd = spark.sparkContext.parallelize(data, numSlices=args.partitions)
    got = np.asarray(spark_fft_rdd(rdd, args.n).map(lambda item: item[1]).collect())
    expected = np.fft.fft(data)
    print(f"match={np.allclose(got, expected)} max_abs_error={np.max(np.abs(got - expected)):.3e}")
    spark.stop()


if __name__ == "__main__":
    main()
