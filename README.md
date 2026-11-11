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
- `scripts/liquidity/`: CSI 300 forecasting, timing comparisons and paper plots; see the [workflow guide](scripts/liquidity/README.md) and [data contract](LIQUIDITY.md).
- `HighFrequencyLiquidity/`: optional pinned submodule for order-book reconstruction, weights, liquidity construction and validation. Forecasting from supplied liquidity data runs without it.
- `data/`: input data used by examples and experiments.
- `docs/`: separate local manuscript checkout, with TeX sources and final paper figures.
- `~/running/dts/`: generated outputs from experiments, benchmarks and applications.
- `legacy/`: archived earlier scripts for provenance.
- `requirements.txt` / `pyproject.toml`: environment specifications.

## Generated Results

Store generated runs, prepared arrays, checkpoints and diagnostic reports under
`~/running/dts/`, outside the source checkout. This checkout's ignored
`artifacts`, `results`, `output` and `tmp` links retain compatibility with older
commands. Use `--out`, `--output-dir` or `--pdf`, as supported by each script, to
choose an external destination.

Manuscript TeX and final paper figures remain regular files in `docs/` and
`docs/figures/`. The manuscript is maintained in its own Git repository; DTS
tracks the inference, replication and plotting code. Level 2 archives and
duplicate local construction scripts are outside the DTS paper workflow.

## Paper outputs

The following Python scripts produce the main experiment, application and
computational results.

| Python script | Results produced |
| --- | --- |
| [replicate_main_results.py](scripts/replicate_main_results.py) | Posterior summaries and marginal-density comparisons for aggregation methods, group sizes, frequency partitioning and time-domain versus frequency-domain splitting |
| [replicate_ar2_regression.py](scripts/replicate_ar2_regression.py) | Full-data and spectral-DC AR(2) regression posterior summaries, credible intervals and Wasserstein-distance diagnostics |
| [run_section4_experiments.py](scripts/experiments/run_section4_experiments.py) | Spark ARTFIMA posterior draws and shard results, including the temperature-modelling application |
| [run_spark_mcmc.py](scripts/run_spark_mcmc.py) | Distributed FFT results and frequency-group MCMC samples |
| [benchmark_dfft_vs_numpy.py](scripts/benchmark_dfft_vs_numpy.py) | Single-process versus distributed FFT runtime and numerical-agreement measurements |
| [dfft_memory_8gb.py](scripts/figures/dfft_memory_8gb.py) | FFT memory-limit comparison figure and summary tables |
| [benchmark_dfft_scaling.py](scripts/benchmark_dfft_scaling.py) | Distributed FFT runtime measurements across concurrent-task settings |
| [dfft_scaling.py](scripts/figures/dfft_scaling.py) | FFT scaling figure, speed-ups and parallel-efficiency table |
| [compare_csi300_boundary_spark.py](scripts/liquidity/compare_csi300_boundary_spark.py) | Rolling Spectral DC ARTFIMA liquidity forecasts, matched benchmarks, forecast scores and posterior diagnostics |
| [compare_csi300_boundary_arima110.py](scripts/liquidity/compare_csi300_boundary_arima110.py) | ARIMA(1,1,0) benchmark forecasts and scores on the matched liquidity tasks |
| [compare_csi300_boundary_arima101.py](scripts/liquidity/compare_csi300_boundary_arima101.py) | ARIMA(1,0,1) benchmark forecasts and scores on the matched liquidity tasks |
| [plot_csi300_liquidity_session_profile.py](scripts/liquidity/plot_csi300_liquidity_session_profile.py) | Intraday liquidity profiles and one-second training/test spread plots |
| [plot_csi300_paper_rmse.py](scripts/liquidity/plot_csi300_paper_rmse.py) | G=20 RMSE-reduction figure by forecast time and horizon, using audited paired-bootstrap summaries |
| [plot_csi300_liquidity_forecast_errors.py](scripts/liquidity/plot_csi300_liquidity_forecast_errors.py) | Matched-forecast RMSE reductions and paired-bootstrap confidence intervals; supports `--ray-actors` |
| [run_csi300_timing_comparison.py](scripts/liquidity/run_csi300_timing_comparison.py) | Matched full-data, sequential-DC and parallel-DC timing measurements |
| [report_csi300_timing_comparison.py](scripts/liquidity/report_csi300_timing_comparison.py) | Timing-component tables, speed-up summaries and runtime comparison figures |
| [investigate_csi300_timing.py](scripts/liquidity/investigate_csi300_timing.py) | CPU-frequency, memory, critical-path and socket-placement diagnostics |

Use the saved experiment configurations for paper comparisons. The `quick`
presets provide small smoke runs; plotting scripts use saved results without
refitting the models.

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
