#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Distributed Fast Fourier Transform (DFFT) implementation using Apache Spark.

This script implements a distributed Cooley–Tukey FFT algorithm using Spark RDDs.

The input time series is partitioned across multiple shards, local FFTs are

computed independently, followed by twiddle-factor corrections and a second-stage

FFT to recover the full discrete Fourier transform.

The implementation is intended for validating the correctness of the DFFT

algorithm by comparing its output against NumPy's FFT implementation.

Inputs:

    - One-dimensional time series data

    - Number of Spark partitions (must be a power of two)

Outputs:

    - Full FFT coefficients computed by DFFT

    - Verification against numpy.fft.fft()

Author: Zixuan Wang
Feb 2026

"""


from pyspark.sql import SparkSession
import numpy as np
import math
import cmath


def compute_subfft(kv, P, M):
    p, items = kv

    items = sorted(items, key=lambda x: (x[0] - p) // P)
    vals = [v for _, v in items]

    fft_vals = np.fft.fft(vals)

    return [(r + p * M, p, fft_vals[r]) for r in range(M)]


def DFFT(df, column: str, numShards: int):
    spark = SparkSession.builder.getOrCreate()

    rdd = df.rdd.map(lambda row: row[column])
    N = rdd.count()
    P = numShards

    if (P & (P - 1)) != 0:
        raise ValueError(f"numShards must be a power of two, but got {P}")

    rem = N % P
    if rem > 0:
        print(f"Warning: dropping last {rem} elements so N is divisible by P")
        rdd = (
            rdd.zipWithIndex()
               .filter(lambda vi: vi[1] < N - rem)
               .map(lambda vi: vi[0])
        )

    N_used = N - rem
    M = N_used // P

    subffts = (
        rdd.zipWithIndex()
           .map(lambda vi: (vi[1] % P, (vi[1], vi[0])))
           .groupByKey(numPartitions=P)
           .flatMap(lambda kv: compute_subfft(kv, P, M))
           .cache()
    )

    twiddled = subffts.map(lambda triple: (
        triple[0],
        triple[1],
        triple[2] * cmath.exp(
            -2j * math.pi * (triple[0] // M) * (triple[0] % M) / N_used
        )
    ))

    paired = twiddled.map(lambda triple: (
        triple[0] % M,
        (triple[1], triple[2])
    ))

    grouped = paired.groupByKey(numPartitions=P)

    def second_stage(grouped_pair):
        r, seq = grouped_pair

        seq_sorted = sorted(seq, key=lambda x: x[0])
        vals = [val for _, val in seq_sorted]

        fft_vals = np.fft.fft(vals)

        for q, fft_v in enumerate(fft_vals):
            yield (q * M + r, fft_v)

    final = (
        grouped
        .flatMap(second_stage)
        .sortByKey()
        .map(lambda kv: kv[1])
    )

    return final


if __name__ == "__main__":
    spark = (
        SparkSession.builder
        .appName("DFFT-Test")
        .getOrCreate()
    )

    sc = spark.sparkContext

    N = 160

    for P in (1, 2, 4, 8, 16):
        if N % P != 0:
            continue

        data = np.random.random(N).tolist()

        df = spark.createDataFrame(
            [(float(x),) for x in data],
            ["y"]
        )

        dfft_out = DFFT(df, "y", numShards=P).collect()
        numpy_out = np.fft.fft(data)

        print(f"P={P} -> match? {np.allclose(dfft_out, numpy_out)}")

    spark.stop()