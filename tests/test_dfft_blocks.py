import os

import numpy as np
import pytest
from pyspark.sql import SparkSession

from dts.dfft_blocks import collect_block_spectrum, spark_fft_contiguous_blocks


os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[4]")
        .appName("dfft-block-accuracy")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("WARN")
    yield session
    session.stop()


@pytest.mark.parametrize(
    ("series_length", "partitions"),
    [(64, 1), (256, 4), (1024, 8), (1024, 128), (4096, 16), (360, 6)],
)
def test_block_fft_matches_numpy_for_real_input(spark, series_length, partitions):
    rng = np.random.default_rng(20260804 + series_length + partitions)
    values = rng.normal(size=series_length)
    block_size = series_length // partitions
    blocks = [
        (block_id, values[block_id * block_size : (block_id + 1) * block_size].copy())
        for block_id in range(partitions)
    ]

    result_rdd = spark_fft_contiguous_blocks(
        spark.sparkContext.parallelize(blocks, partitions),
        series_length,
        partitions,
    )
    actual = collect_block_spectrum(result_rdd, series_length, partitions)

    np.testing.assert_allclose(actual, np.fft.fft(values), rtol=1e-11, atol=1e-11)


def test_block_fft_matches_numpy_for_complex_input(spark):
    series_length = 2048
    partitions = 8
    rng = np.random.default_rng(271828)
    values = rng.normal(size=series_length) + 1j * rng.normal(size=series_length)
    block_size = series_length // partitions
    blocks = [
        (block_id, values[block_id * block_size : (block_id + 1) * block_size].copy())
        for block_id in range(partitions)
    ]

    result_rdd = spark_fft_contiguous_blocks(
        spark.sparkContext.parallelize(blocks, partitions),
        series_length,
        partitions,
    )
    actual = collect_block_spectrum(result_rdd, series_length, partitions)

    np.testing.assert_allclose(actual, np.fft.fft(values), rtol=1e-11, atol=1e-11)
