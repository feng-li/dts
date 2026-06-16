from pyspark.sql import functions as F
from pyspark.sql import SparkSession


def insert_group_id(sdf, n_groups, method):
    """
    Simple function that adds consecutive partition ids for a Spark DataFrame.

    """
    spark = SparkSession.builder.getOrCreate()

    if method == "ts":
        sample_size = sdf.count()
        id = spark.range(sample_size) # Spark DataFrame with an 'id' column 0,1,2,...
        sdf = sdf.join(id)
        sample_size_per_partition = int(sample_size/n_groups)
        sdf = sdf.withColumn("group_id", F.floor(sdf.id/sample_size_per_partition))

    elif method == "random":
        sdf = sdf.withColumn("group_id", F.monotonicall_increasing_id() % n_groups)

    return sdf
