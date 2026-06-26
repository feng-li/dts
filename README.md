# dts

Frequency-domain divide-and-conquer MCMC for stationary time series.

This repository contains the reproducible Python implementation for
`docs/Manuscript_2026_Zi.tex`. The active package is `dts/`; paper and
deployment entry points are in `scripts/`. Older source scripts migrated from
the student project are kept in `legacy/`, `figures/`, `section4_experiments/`,
and `section5_applications/` for traceability.

## What Is Implemented

- Whittle likelihood inference for ARMA and ARTFIMA models.
- Frequency-domain partitioning that preserves the global periodogram.
- Local, Ray, and Spark-based shard MCMC.
- Consensus and simple-average posterior aggregation.
- AR(2) regression diagnostics for the DC-BATS comparison.
- JAX-based gradients and Hessians for MAP estimation.

The code defaults JAX to CPU and enables 64-bit arithmetic through
`dts._jax.configure_jax()`. Regular NumPy is still used for random sampling,
mutable arrays, file I/O, and SciPy/statsmodels interop.

## Repository Layout

- `dts/`: reusable Python package.
- `scripts/`: runnable replication, Spark, and artifact utilities.
- `data/`: bundled manuscript inputs and benchmark data.
- `docs/Manuscript_2026_Zi.tex`: current paper source.
- `docs/REPLICATION.md`: detailed replication and development notes.
- `artifacts/`: generated outputs. This directory is ignored by git.
- `legacy/`: old migrated helpers retained only for reference.

Within `dts/`, `optimization.py` owns the shared JAX MAP/proposal-covariance
routine, and `runtime.py` owns command-line warning setup.

## Installation

Use the existing project environment when available:

```sh
python -m pip install -e .
```

For Spark runs, install the optional dependency:

```sh
python -m pip install -e ".[spark]"
```

The package dependencies are declared in `pyproject.toml`. `requirements.txt`
is kept for environment recreation.

## Quick Validation

Run the fast checks before starting manuscript-scale jobs:

```sh
python scripts/replicate_main_results.py \
  --preset quick \
  --experiments all \
  --output-dir artifacts/quick_main

python scripts/replicate_ar2_regression.py \
  --preset quick \
  --output-dir artifacts/quick_ar2
```

These commands show progress bars by default; add `--no-progress` for clean
batch logs. Inspect the CSV summaries and `manifest.json` files under the
selected artifact folders.

The shard backend defaults to `auto`: it uses Ray when the effective CPU count is
greater than 1, and local serial execution when it is 1. Omit `--num-cpus` to
use the detected CPU count, set `--num-cpus 1` for local debugging, or set a
larger value to cap the local Ray runtime:

```sh
python scripts/replicate_main_results.py \
  --preset quick \
  --experiments all \
  --num-cpus 4 \
  --output-dir artifacts/quick_main_ray

python scripts/replicate_ar2_regression.py \
  --preset quick \
  --num-cpus 4 \
  --output-dir artifacts/quick_ar2_ray
```

Omit `--ray-address` to let Ray start a local runtime, or pass a cluster address
such as `--ray-address auto`. Use `--backend local` or `--backend ray` only when
you need to override automatic selection.

## Paper Replication

Full manuscript-scale runs use 15,000 MCMC iterations with 5,000 burn-in and
can take a long time:

```sh
python scripts/replicate_main_results.py \
  --preset paper \
  --experiments all

python scripts/replicate_ar2_regression.py \
  --preset paper
```

The main script writes posterior summaries, figures, and a run manifest to
`artifacts/replication/`. The AR(2) script writes its diagnostic table to
`artifacts/ar2_regression/`.

## Spark Workflow

Run the distributed frequency-domain MCMC entry point with `spark-submit`:

```sh
spark-submit scripts/run_spark_mcmc.py \
  --input data/SimARTFIMA11.csv \
  --column y \
  --groups 10 \
  --fft-partitions 16 \
  --q 1 \
  --p 1 \
  --tfi-term \
  --output artifacts/spark_mcmc
```

Check the Spark FFT implementation independently:

```sh
python scripts/check_dfft.py \
  --n 160 \
  --partitions 8
```

Use `scripts/stack_shard_draws.py` for legacy shard files named
`shardXX_draws.npy` and `shardXX_logp.npy`.
