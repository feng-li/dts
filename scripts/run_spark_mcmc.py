#!/usr/bin/env python3
"""Run Spark-based frequency-domain MCMC on a CSV time series."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StructField, StructType

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

from dts.dfft import spark_periodogram_dataframe
from dts.mapper import mapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "SimARTFIMA11.csv")
    parser.add_argument("--column", default="y")
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--fft-partitions", type=int, default=16)
    parser.add_argument("--ar-order", type=int, default=1)
    parser.add_argument("--ma-order", type=int, default=1)
    parser.add_argument("--tfi-term", action="store_true")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--basinhopping", action="store_true")
    parser.add_argument("--basinhopping-iter", type=int, default=25)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "artifacts" / "spark_mcmc")
    return parser.parse_args()


def main() -> None:
    from pyspark.sql import SparkSession

    args = parse_args()
    spark = (
        SparkSession.builder.appName("dts-spectral-mcmc")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    raw_df = spark.read.csv(str(args.input), header=True, inferSchema=True)
    periodogram_df = spark_periodogram_dataframe(raw_df, args.column, args.fft_partitions, args.groups)

    conf_model = {
        "TFI_term": args.tfi_term,
        "partition_num": args.groups,
        "ar_order": args.ar_order,
        "ma_order": args.ma_order,
    }
    conf_mcmc = {
        "n_samples": args.samples,
        "Burn_in": args.burn_in,
        "seed": args.seed,
        "basinhopping": args.basinhopping,
        "basinhopping_iter": args.basinhopping_iter,
    }

    schema = StructType(
        [
            StructField("shard_id", IntegerType(), False),
            StructField("samples", ArrayType(ArrayType(DoubleType())), False),
            StructField("map_estimate", ArrayType(DoubleType()), False),
            StructField("log_p", DoubleType(), False),
            StructField("acceptance_rate", DoubleType(), False),
        ]
    )

    def shard_mcmc(pdf):
        return mapper(pdf, conf_model, conf_mcmc)

    result_df = periodogram_df.groupBy("shard_id").applyInPandas(shard_mcmc, schema=schema)
    args.output.mkdir(parents=True, exist_ok=True)
    result_df.orderBy("shard_id").write.mode("overwrite").parquet(str(args.output))
    spark.stop()


if __name__ == "__main__":
    main()
