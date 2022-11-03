import os, sys, pathlib

# Only used for interactive mode
if hasattr(sys, 'ps1'):
    import findspark
    findspark.init() # Make sure you have $SPARK_HOME set
    libdir = pathlib.Path(os.getcwd()).parent
    sys.path.append(libdir)


## Pyspark setup
import pyspark
conf = pyspark.SparkConf().setAppName("Spark DTS App").setAll(
    [('spark.executor.memory', '8g'),
     ('spark.executor.cores', '1'),
     ('spark.cores.max', '16'),
     ('spark.driver.memory', '32g')])
spark = pyspark.sql.SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel("WARN")  # "DEBUG", "ERROR"

# Enable Arrow-based columnar data transfers. Ensure that PyArrow is installed and available on all cluster nodes with `pip install pyarrow`
# spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
# https://docs.azuredatabricks.net/spark/latest/spark-sql/udf-python-pandas.html#setting-arrow-batch-size
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", 10000) # default

# spark.conf.set("spark.sql.shuffle.partitions", 10)
# print(spark.conf.get("spark.sql.shuffle.partitions"))

from numpy.fft import fft

# dts functions
from dts import *


## Data source
project_path = "~/code/dts/projects"
data_path = os.path.expanduser(project_path) + '/../dts/data/SimARTFIMA11.txt'
sdf = spark.read.csv(data_path)


## Model and MCMC configurations
conf_model = {
    "q": 1,
    "p": 1,
    "TFI_term": False,
    "exact_L": True
}

conf_mcmc = {
    "n_groups": 10, # Number of groups
    "n_samples": 15000,
    "Burn_in": int(5000)
}

# Add partition id
sdf = insert_group_id(sdf=sdf, n_groups=conf_mcmc["n_groups"], method="ts")

# One dimensional FFT and periodogram FIXME: This should be done within Spark. The current
# method should only be used if the resulting NumPy ndarray is expected to be small.
def fft_periodogram(array):  # Construct Periodogram
    """Make an one-dimensional FFT and obtain the periodogram
    """
    fft_values = fft(array)
    id = int(np.floor((len(fft_values)-1)/2))
    out = np.square(np.abs(fft_values[0:(id)]))/(2 * np.pi * len(fft_values))
    return out

# I_pg_full = fft_periodogram(sdf.select("_c0").to_Pandas().to_numpy())
import numpy as np
I_pg_full = fft_periodogram(np.loadtxt(data_path))




import pandas as pd
df = spark.createDataFrame(
    [(1, 1.0), (1, 2.0), (2, 3.0), (2, 5.0), (2, 10.0)],
    ("id", "v"))

def subtract_mean(pdf: pd.DataFrame) -> pd.DataFrame:
    # pdf is a pandas.DataFrame
    v = pdf.v
    return pdf.assign(v=v - v.mean())

df.groupby("id").applyInPandas(subtract_mean, schema="id long, v double").show()
