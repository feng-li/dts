# Plan

## Goal

Reorganize the current repository around `docs/Manuscript_2026_Zi.tex` so the main paper results can be reproduced from structured Python code.

## Source Status

- Target manuscript: `docs/Manuscript_2026_Zi.tex`.
- Requested source: `/data/student/Zixuan-UTS`.
- `/data/student/Zixuan-UTS/projects` is now readable.
- Migrated code from the source project scripts instead of preserving the monolithic files:
  `run_dts_python.py`, `run_dts_pyspark.py`, `run_dts_fixed_pyspark.py`, `try_true_parallel.py`,
  `DFFT.py`, `DTS_AR2.py`, `spectral_parallel_DLR.py`, and `stack_MCMC_draws.py`.

## Structure

- `dts/`: reusable Python package.
- `scripts/`: executable replication and Spark scripts.
- `data/`: bundled input data copied from `projects/data/`.
- `artifacts/`: generated results, ignored by git.

## Module Design

- `dts/mcmc.py`: model specification, ARMA/ARTFIMA spectral density, Whittle likelihood, priors, parameter transforms, and Metropolis sampler.
- `dts/partition.py`: periodogram construction and frequency/time partitioning.
- `dts/aggregation.py`: simple average, consensus Monte Carlo, parameter transforms, credible intervals, and Wasserstein diagnostics.
- `dts/experiments.py`: local full-data, frequency-shard, and time-shard experiment runners.
- `dts/dfft.py`: Spark DFFT and periodogram DataFrame helpers.
- `dts/mapper.py`: Spark grouped-map MCMC wrapper.
- `dts/regression.py`: AR(2) regression with ARMA/ARTFIMA errors for the DC-BATS comparison.
- `dts/artifacts.py`: stack per-shard MCMC draw/log-p artifacts.
- `dts/_jax.py`: central JAX runtime configuration for CPU and 64-bit autodiff.
- `dts/optimization.py`: shared MAP optimization and proposal covariance setup.
- `dts/runtime.py`: command-line warning setup for replication scripts.

## Replication Mapping

- Experiment 1: `scripts/replicate_main_results.py --experiments combination`.
- Experiment 2: `scripts/replicate_main_results.py --experiments group-size`.
- Experiment 3: `scripts/replicate_main_results.py --experiments partition`.
- Experiment 4: `scripts/replicate_main_results.py --experiments time-frequency`.
- AR(2) DC-BATS comparison diagnostics: `scripts/replicate_ar2_regression.py --preset paper`.
- Bromma/Stockholm application: `spark-submit scripts/run_spark_mcmc.py --input data/Bromma_AR2_TFI_MA2.csv --column y --groups 10 --fft-partitions 16 --ar-order 2 --ma-order 2 --tfi-term --samples 15000 --burn-in 5000 --basinhopping --basinhopping-iter 100`.
- All local experiments: `scripts/replicate_main_results.py --preset paper --experiments all`.
- Spark frequency-domain run: `scripts/run_spark_mcmc.py`.
- Stacking saved shard draws: `scripts/stack_shard_draws.py`.

## Notes

- `--preset quick` validates the full pipeline with short chains and truncated data.
- `--preset paper` uses the manuscript-scale MCMC length and can take a long time.
- Automatic differentiation now uses JAX. NumPy is still used for mutable arrays,
  random sampling, saved artifacts, and SciPy/statsmodels interop.
- Operational documentation is in `README.md` and `docs/REPLICATION.md`.
- Copied newly accessible source data into `data/`: Bromma, Maine demand, Vancouver CSV, DC-BATS AR(2), and LM/SM DC-BATS benchmark arrays.
- The full Bromma and paper-scale AR(2) runs are intentionally long-running; use quick presets for validation.
