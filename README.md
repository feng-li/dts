# dts

`dts` provides reproducible implementations of frequency-domain divide-and-conquer Bayesian inference for long stationary time series. It combines Whittle-likelihood inference, distributed FFT, and shard-wise MCMC with subposterior aggregation.

## Purpose

- Enable scalable Bayesian inference for long univariate and multivariate time-series datasets.
- Evaluate spectral partitioning methods and provide practical experiments for ARMA, ARTFIMA, and dynamic AR(2) regression.
- Support both local and distributed execution (Ray/Spark) for replication and comparison of manuscript results.

## Main Features

- Whittle likelihood and frequency-domain likelihood approximation.
- Frequency partitioning into independent Whittle subposteriors.
- JAX-based MAP and Hessian-based proposal construction.
- Local, Ray, and Spark execution backends.
- Spark-based distributed FFT (exact, block-based implementation) to handle very long series.
- Posterior aggregation methods (simple averaging, consensus Monte Carlo, and additional methods).
- Scripts for experiment replication, Spark workflows, and artifact generation.

## Repository Layout

- `dts/`: reusable Python package.
- `scripts/`: entry points for replication and Spark workflows.
- `data/`: input data used by examples and experiments.
- `docs/`: manuscript sources and supporting replication notes.
- `results/`: generated outputs from benchmark runs.
- `legacy/`: archived earlier scripts for provenance.
- `requirements.txt` / `pyproject.toml`: environment specifications.

## Environment

- Python: >= 3.9
- Core dependencies: `numpy`, `scipy`, `pandas`, `jax`, `autograd`, `ray`, `statsmodels`, `tqdm`, `matplotlib`.
- Optional dependency for distributed FFT/MCMC: `pyspark` (Spark backend).
- The codebase is designed for Unix-like environments; Spark workflows require a working Spark installation and Java runtime for your Spark distribution.

Install the package in editable mode:

```sh
python -m pip install -e .
```

Add optional Spark support:

```sh
python -m pip install -e ".[spark]"
```

## Quick Start

Define model orders explicitly using conventional ARMA terminology:

```python
from dts import ModelSpec

# ARMA(1, 2): one autoregressive term and two moving-average terms.
model = ModelSpec(ar_order=1, ma_order=2)
```

Run a small sanity check:

```sh
python scripts/replicate_main_results.py --preset quick
python scripts/replicate_ar2_regression.py --preset quick
```

Run a local distributed Spark experiment:

```sh
spark-submit scripts/run_spark_mcmc.py \
  --input data/SimARTFIMA11.csv \
  --column y \
  --groups 10 \
  --fft-partitions 16 \
  --ar-order 1 \
  --ma-order 1 \
  --tfi-term \
  --output artifacts/spark_mcmc
```

Use `docs/REPLICATION.md` for full manuscript settings, including longer runs and output locations.
