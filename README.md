# `dts`

Distributed time series modeling with Apache Spark

# Introduction

`dts` is designed to facilitate forecasting ultra-long time series by utilizing the industry-standard MapReduce framework. The algorithm is developed on Spark platform with Python with interfaces.


# System requirements

- `Spark >= 3.2.0`
- `Python >= 3.7.0`
    - `pyspark >= 2.3.1`
    - `rpy2 >= 3.0.4`
    - `scikit-learn >= 0.21.2`
    - `numpy >= 1.16.3`
    - `pandas >= 0.23.4`
- `R >= 3.5.2`
    - `forecast >= 8.5`
    - `polynom = 1.3.9`
    - `dplyr >= 0.8.4`
    - `quantmod >= 0.4.13`
    - `magrittr >= 1.5`

# Usage

## `dts`
Run the [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) code to forecast the time series of the demo data.

```sh
  ./bash/run_dts.sh
```
or simply run
```py
  PYSPARK_PYTHON=/usr/local/bin/python3.7 ARROW_PRE_0_15_IPC_FORMAT=1 spark-submit ./run_darima.py
```
**Note**: `ARROW_PRE_0_15_IPC_FORMAT=1` is added to instruct `PyArrow >= 0.15.0` to use the legacy IPC format with the older Arrow Java that is in Spark 2.3.x and 2.4.x.

# References
