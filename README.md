# dts

`dts` provides reproducible implementations of frequency-domain divide-and-conquer Bayesian inference for long stationary time series. It combines Whittle-likelihood inference, distributed FFT, and shard-wise MCMC with subposterior aggregation.

## Paper

This repository accompanies our paper [Spectral Divide-and-Conquer MCMC for Long Stationary Time Series](https://arxiv.org/abs/2609.23985) by Zixuan Wang, Matias Quiroz, Feng Li, Mattias Villani, and Robert Kohn (2026).

To cite the paper:

```bibtex
@misc{WangZ2026SpectralDivideandconquer,
  title = {Spectral Divide-and-Conquer MCMC for Long Stationary Time Series},
  author = {Wang, Zixuan and Quiroz, Matias and Li, Feng and Villani, Mattias and Kohn, Robert},
  year = {2026},
  month = sep,
  url = {https://arxiv.org/abs/2609.23985}
}
```

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
- `scripts/liquidity/`: CSI 300 liquidity construction, audits, forecasts, and plots; see the [script catalog](LIQUIDITY.md) and [workflow guide](scripts/liquidity/README.md).
- `projects/liquidity/`: shared liquidity models and order-book reconstruction.
- `data/`: input data used by examples and experiments.
- `docs/`: manuscript sources and supporting replication notes.
- `results/`: generated outputs from benchmark runs.
- `legacy/`: archived earlier scripts for provenance.
- `requirements.txt` / `pyproject.toml`: environment specifications.

## Generated Results

Store generated results under `~/running/dts/`, outside the source checkout.
This checkout's `artifacts`, `results`, `output`, and `tmp` entries are ignored
symbolic links to the corresponding directories there. Existing commands and
saved absolute paths continue to work through these links, including incremental
resume and references to previously generated stock panels. PDFs are stored in
`~/running/dts/output/pdf/`. The move is recorded in
`~/running/dts/migration.json`.

Remaining manuscript figures and older document outputs are stored under
`~/running/dts/output/docs/`; build files and previews are under
`~/running/dts/tmp/`, and Python/notebook caches are under
`~/running/dts/cache/`. Compatibility links preserve existing manuscript and
script paths. Additional moves are recorded in
`~/running/dts/migration_artifacts_*.json`, including file hashes. From `docs/`,
`latexmk -pdf ApplicationCSI300Liquidity.tex` writes its PDF to
`~/running/dts/output/pdf/` and auxiliary files to
`~/running/dts/tmp/latex/docs/` through the local `latexmkrc`.

## Environment

- Python: >= 3.9
- Core dependencies: `numpy`, `scipy`, `pandas`, `jax`, `autograd`, `ray`, `statsmodels`, `tqdm`, `matplotlib`, `pyspark`.
- The codebase is designed for Unix-like environments; Spark workflows require a working Spark installation and Java runtime for your Spark distribution.

Install the package in editable mode:

```sh
python -m pip install -e .
```

## Quick Start

Define model orders explicitly using conventional ARMA terminology:

```python
from dts import ModelSpec

# ARMA(1, 2): one autoregressive term and two moving-average terms.
model = ModelSpec(ar_order=1, ma_order=2)
```

Whittle MCMC uses a compiled JAX loop and cached Fourier quantities by default.
For numerical checks, call `dts.mcmc.sampler(..., backend="vanilla")` or set
`MCMCSettings(sampler_backend="vanilla")`. The compiled and vanilla versions
use the same priors, Metropolis rule and NumPy random stream; floating-point
likelihood rounding can differ. Exact time-domain sampling retains its Python
implementation. Spark commands accept `--sampler-backend vanilla`.

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
  --fft-partitions 256 \
  --ar-order 1 \
  --ma-order 1 \
  --tfi-term \
  --output artifacts/spark_mcmc
```
