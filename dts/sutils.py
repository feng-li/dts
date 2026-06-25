"""Small Spark DataFrame utilities."""

from __future__ import annotations


def insert_group_id(sdf, n_groups: int, method: str = "time"):
    """Add a ``group_id`` column to a Spark DataFrame."""
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.getOrCreate()

    if method == "time":
        sample_size = sdf.count()
        ids = spark.range(sample_size)
        sample_size_per_partition = max(int(sample_size / n_groups), 1)
        return sdf.join(ids).withColumn("group_id", F.floor(F.col("id") / sample_size_per_partition))

    if method == "random":
        return sdf.withColumn("group_id", F.monotonically_increasing_id() % n_groups)

    raise ValueError(f"unknown grouping method: {method!r}")
