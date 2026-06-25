"""Spark helpers for distributed FFT-based periodograms."""

from __future__ import annotations

import cmath
import math
from typing import Iterable

import numpy as np


def distributed_dft(data, sc, num_slices: int):
    """Reference distributed DFT used for small Spark sanity checks."""
    values = list(data)
    n_obs = len(values)
    broadcast = sc.broadcast(values)

    def compute_chunk(indices: Iterable[int]):
        x = broadcast.value
        out = []
        for k in indices:
            total = 0.0 + 0.0j
            for n, value in enumerate(x):
                total += value * cmath.exp(-2j * cmath.pi * k * n / n_obs)
            out.append((k, total))
        return out

    return (
        sc.parallelize(range(n_obs), numSlices=num_slices)
        .mapPartitions(lambda iterator: compute_chunk(list(iterator)))
        .sortByKey()
        .map(lambda item: item[1])
        .collect()
    )


def _compute_subfft(kv, partitions: int, block_size: int):
    shard, items = kv
    items = sorted(items, key=lambda item: (item[0] - shard) // partitions)
    vals = [value for _, value in items]
    fft_vals = np.fft.fft(vals)
    return [(r + shard * block_size, shard, fft_vals[r]) for r in range(block_size)]


def spark_fft_rdd(rdd, series_length: int | None = None):
    """Cooley-Tukey style FFT over an RDD.

    The RDD partition count must be a power of two and divide the series length.
    Returns an RDD of ``(frequency_index, fft_value)`` sorted by frequency index.
    """
    n_obs = int(series_length or rdd.count())
    partitions = rdd.getNumPartitions()
    if partitions < 1 or (partitions & (partitions - 1)) != 0:
        raise ValueError("RDD partition count must be a power of two")
    if n_obs % partitions:
        raise ValueError("series length must be divisible by the RDD partition count")

    block_size = n_obs // partitions
    subffts = (
        rdd.zipWithIndex()
        .map(lambda item: (item[1] % partitions, (item[1], item[0])))
        .groupByKey(numPartitions=partitions)
        .flatMap(lambda kv: _compute_subfft(kv, partitions, block_size))
        .cache()
    )
    twiddled = subffts.map(
        lambda item: (
            item[0],
            item[1],
            item[2]
            * cmath.exp(-2j * math.pi * (item[0] // block_size) * (item[0] % block_size) / n_obs),
        )
    )
    grouped = twiddled.map(lambda item: (item[0] % block_size, (item[1], item[2]))).groupByKey(
        numPartitions=block_size
    )

    def second_stage(grouped_pair):
        r, seq = grouped_pair
        vals = [value for _, value in sorted(seq, key=lambda item: item[0])]
        for q, fft_value in enumerate(np.fft.fft(vals)):
            yield (q * block_size + r, fft_value)

    return grouped.flatMap(second_stage).sortByKey()


def spark_fft_indexed_rdd(indexed_rdd, series_length: int, partitions: int):
    """FFT over an RDD of ``(original_index, value)`` pairs."""
    if partitions < 1 or (partitions & (partitions - 1)) != 0:
        raise ValueError("partitions must be a power of two")
    if series_length % partitions:
        raise ValueError("series length must be divisible by partitions")

    block_size = series_length // partitions
    subffts = (
        indexed_rdd.map(lambda item: (int(item[0]) % partitions, (int(item[0]), item[1])))
        .groupByKey(numPartitions=partitions)
        .flatMap(lambda kv: _compute_subfft(kv, partitions, block_size))
        .cache()
    )
    twiddled = subffts.map(
        lambda item: (
            item[0],
            item[1],
            item[2]
            * cmath.exp(-2j * math.pi * (item[0] // block_size) * (item[0] % block_size) / series_length),
        )
    )
    grouped = twiddled.map(lambda item: (item[0] % block_size, (item[1], item[2]))).groupByKey(
        numPartitions=block_size
    )

    def second_stage(grouped_pair):
        r, seq = grouped_pair
        vals = [value for _, value in sorted(seq, key=lambda item: item[0])]
        for q, fft_value in enumerate(np.fft.fft(vals)):
            yield (q * block_size + r, fft_value)

    return grouped.flatMap(second_stage).sortByKey()


def spark_periodogram_dataframe(df, column: str, fft_partitions: int, n_groups: int):
    """Compute a Spark DataFrame with positive-frequency periodogram shards."""
    spark = df.sparkSession
    rdd = df.select(column).rdd.map(lambda row: float(row[column]))
    n_obs = rdd.count()
    indexed = rdd.zipWithIndex().map(lambda item: (int(item[1]), float(item[0])))
    remainder = n_obs % fft_partitions
    if remainder:
        keep = n_obs - remainder
        indexed = indexed.filter(lambda item: item[0] < keep)
        n_obs = keep

    fft_rdd = spark_fft_indexed_rdd(indexed.repartition(fft_partitions), n_obs, fft_partitions)
    n_freq = int(math.floor((n_obs - 1) / 2))
    rows = (
        fft_rdd.filter(lambda item: 1 <= item[0] <= n_freq)
        .map(
            lambda item: (
                int((item[0] - 1) % n_groups),
                int(item[0]),
                float(2.0 * math.pi * item[0] / n_obs),
                float((item[1].real * item[1].real + item[1].imag * item[1].imag) / (2.0 * math.pi * n_obs)),
                float(item[1].real),
                float(item[1].imag),
            )
        )
    )
    return spark.createDataFrame(
        rows,
        schema=["shard_id", "frequency_index", "omega", "periodogram", "real", "imag"],
    )
