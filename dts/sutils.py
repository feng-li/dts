from pyspark.sql import functions as F
from pyspark.sql.functions import udf, pandas_udf, PandasUDFType, monotonically_increasing_id



def insert_partition_id(sdf, partition_num):
    """
    Simple function that adds consecutive partition ids for a Spark DataFrame.

    """
    partition_num = 100
    sample_size = sdf.count()
    sample_size_per_partition = int(sample_size / partition_num)
    sdf = sdf.withColumn("id", F.monotonically_increasing_id()+1)
    sdf = sdf.withColumn("partition_id", F.ceil(sdf.id / partition_num))

    return(sdf)
