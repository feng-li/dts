# dts

Frequency-domain divide-and-conquer MCMC for stationary time series.

This repository is organized around the manuscript in
`docs/Manuscript_2026_Zi.tex`. The reusable Python code lives in `dts/`, while
paper replication entry points live in `scripts/`.

## Structure

- `dts/mcmc.py`: ARMA/ARTFIMA spectral density, Whittle likelihood, priors, and Metropolis sampler.
- `dts/partition.py`: periodogram construction and systematic/sequential frequency partitioning.
- `dts/aggregation.py`: average and consensus Monte Carlo aggregation plus posterior summaries.
- `dts/experiments.py`: local experiment runners used by the replication script.
- `dts/dfft.py` and `dts/mapper.py`: Spark DFFT and grouped shard MCMC helpers.
- `dts/regression.py`: AR(2) regression diagnostics used for the DC-BATS comparison.
- `dts/artifacts.py`: helpers for stacking shard-level MCMC artifacts.
- `scripts/replicate_main_results.py`: local replication for the main paper experiments.
- `scripts/replicate_ar2_regression.py`: AR(2) regression CI and Wasserstein diagnostics.
- `scripts/run_spark_mcmc.py`: Spark replication entry point for distributed frequency-domain MCMC.
- `scripts/stack_shard_draws.py`: stack `shardXX_draws.npy` and `shardXX_logp.npy` outputs.
- `data/`: bundled SimARTFIMA, Vancouver, Bromma, Maine, and AR(2) benchmark data.

## Install

```sh
pip install -e .
```

Install Spark support only when needed:

```sh
pip install -e ".[spark]"
```

## Replicate Main Results

Quick validation:

```sh
python scripts/replicate_main_results.py --preset quick --experiments combination
```

Manuscript-scale run:

```sh
python scripts/replicate_main_results.py --preset paper --experiments all
```

Outputs are written to `artifacts/replication/`, including posterior summaries,
diagnostics, and marginal posterior figures. The paper preset uses the paper's
15,000 MCMC iterations with 5,000 burn-in and is expected to be slow.

AR(2) regression comparison:

```sh
python scripts/replicate_ar2_regression.py --preset paper
```

## Spark Run

```sh
spark-submit scripts/run_spark_mcmc.py \
  --input data/SimARTFIMA11.csv \
  --column y \
  --groups 10 \
  --fft-partitions 16 \
  --q 1 --p 1 --tfi-term
```

Bromma/Stockholm application:

```sh
spark-submit scripts/run_spark_mcmc.py \
  --input data/Bromma_AR2_TFI_MA2.csv \
  --column y \
  --groups 10 \
  --fft-partitions 16 \
  --q 2 --p 2 --tfi-term \
  --samples 15000 \
  --burn-in 5000 \
  --basinhopping \
  --basinhopping-iter 100 \
  --output artifacts/bromma_spark
```

Stack per-shard numpy artifacts:

```sh
python scripts/stack_shard_draws.py artifacts/Bromma_ARTFIMA22_Whittle_G10 --groups 10
```

Check the Spark FFT implementation:

```sh
python scripts/check_dfft.py --n 160 --partitions 8
```
