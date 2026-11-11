# Liquidity forecasting and timing experiments

Run commands from the DTS repository root. Generated data, checkpoints, reports
and figures belong under `~/running/dts/`. Forecasting from supplied liquidity
data requires only DTS; construction and validation of Level 2 data belong in
the optional root `HighFrequencyLiquidity/` submodule.

See [LIQUIDITY.md](../../LIQUIDITY.md) for the prepared-data contract, forecasting
scripts, paper figures and construction interface. The existing builder/export
filenames are compatibility entry points into the submodule. Local copies of
older construction scripts are not part of the DTS replication workflow.

## Forecasts from supplied data

The active forecasting entry point is `compare_csi300_boundary_spark.py`.
Resume the saved G=20 experiment with its own settings:

```sh
python -u scripts/liquidity/compare_csi300_boundary_spark.py \
  --config ~/running/dts/artifacts/liquidity/csi300_boundary_spark_g20_full1000_onechain_v5/run_config.json
```

`compare_csi300_boundary_random_days.py` retains the original Ray experiment.
`compare_csi300_boundary_arima110.py` and `compare_csi300_boundary_arima101.py`
provide the corresponding ARIMA benchmarks. Numerical helpers, source guards,
reports and figure scripts are catalogued in `LIQUIDITY.md`.

Saved configurations determine the actual iteration budget. A separate timing
experiment does not change the existing 1,000-date forecasts. Different model
or sampling settings require a separate output directory.

## Paper figures

`plot_csi300_liquidity_session_profile.py` generates the training/test spread
panels. `plot_csi300_paper_rmse.py` regenerates the current G=20 RMSE-reduction
figure from the audited `review_20261016` summaries, checking their forecast
and comparison hashes. Its drawing code is shared with
`plot_csi300_liquidity_forecast_errors.py`, which retains the paired bootstrap
implementation and `--ray-actors` option for the original matched archive.

See the root [paper-output index](../../README.md#paper-outputs) for links to
the manuscript figures, forecast score table and supporting comparison values.

## Computation-time comparison

The benchmark requires the saved training responses, reference manifests and
frozen numerical source archive. It uses every assigned positive frequency,
independent shard-level MAP fits, and the same compiled fixed-mixture sampler
for the full posterior and subposteriors. DC has no prerequisite full-data MAP.

| Script | Role |
| --- | --- |
| `benchmark_csi300_computation.py` | Fits and times full-data, sequential-DC, parallel-DC and Spark G=1 configurations; saves hashes and monitored results |
| `_computation_timing.py` | Separates compilation from synchronous MCMC execution without changing the kernel or random stream |
| `run_csi300_timing_comparison.py` | Runs and resumes the matched timing protocol from a saved configuration |
| `profile_csi300_timing.py` | Replays fitted proposals after compilation, synchronises workers and checks full-length draws against their references |
| `report_csi300_timing_comparison.py` | Reports components and keeps full-data, Spark G=1 and sequential-DC speed-ups distinct |
| `observe_csi300_timing_resources.py` | Records affinities, RSS, swap counters and sampled CPU frequencies |
| `investigate_csi300_timing.py` | Analyses the critical path and resources; supports short-chain and socket-placement controls |
| `run_csi300_timing_comparison_pipeline.py` | Resumes a prepared archive through timing, diagnostics, audit and guarded TeX publication |
| `update_csi300_timing_tex.py` | Updates the comparison after audit and protects concurrent TeX edits |

The current timing specification uses the longest history, 24,300,012 training
observations before 24 July 2026, with 65,000 iterations and 5,000 burn-in.
Three timing repetitions compare full-data MCMC on one core with G=20 on twenty
physical cores. Sequential G=20 and Spark G=1 have one repetition each.
The MCMC-only and hardware diagnostics use the same selected history.

```sh
python -u scripts/liquidity/run_csi300_timing_comparison.py \
  --dates 20260724 --iterations 65000 --burn 5000 \
  --quiet-seconds 0 --interference-policy record \
  --out ~/running/dts/artifacts/liquidity/csi300_timing_65000_g20_longest_20261109_v1
```

Resume the timing stages with the saved configuration:

```sh
python -u scripts/liquidity/run_csi300_timing_comparison.py \
  --config ~/running/dts/artifacts/liquidity/csi300_timing_65000_g20_longest_20261109_v1/config.json
```

`record` retains background-load observations without restarting for brief CPU
activity; report these as shared-host measurements. The optional strict policy
uses `--quiet-seconds 60 --interference-policy exclude`. Neither policy stops
other users' work. Completed cases are verified and skipped on resume.

Outputs include configs, frozen scripts, hashes, raw timing and monitoring
records, CSV summaries, `timing_table.tex` and `timing_breakdown.pdf`. FFT stage
times include scheduling, shuffles, grouping, checkpoint writing and validation;
they are not pure FFT-kernel timings. The native baseline uses SciPy's real FFT;
the distributed FFT supports NumPy kernels or `--fft-backend jax`. MCMC-loop
compilation is measured separately from synchronous execution.

The complete publication pipeline additionally requires its prepared
`pipeline_config.json`, environment and audit snapshots, and `paper_update/`
manifest/template in the experiment archive. To resume that prepared workflow:

```sh
python -u scripts/liquidity/run_csi300_timing_comparison_pipeline.py \
  --input ~/running/dts/artifacts/liquidity/csi300_timing_65000_g20_longest_20261109_v1
```

Source/configuration guards preserve historical experiments, including the
35,000-iteration study. The publisher keeps earlier quality scores separate
from new timing measurements. PDF publication requires LaTeX, Poppler and
`pdfplumber`; the numerical timing stages do not require those tools.

## Posterior approximation quality

`evaluate_csi300_stein.py` and `_stein_discrepancy.py` evaluate the
Gorham--Mackey (2015) graph Stein discrepancy against the common full-data
Whittle posterior using saved draws and frozen numerical sources. This is an
approximation-quality diagnostic, separate from ESS and inference time.

```sh
python -u scripts/liquidity/evaluate_csi300_stein.py \
  --benchmark ~/running/dts/artifacts/liquidity/csi300_computation_independent_maps_20261102_v1 \
  --out ~/running/dts/artifacts/liquidity/csi300_stein_independent_maps_20261102_v1 \
  --draws 1500 --workers 20 --lp-workers 4
```

`report_csi300_independent_timing.py` reproduces the earlier short-chain note
from that benchmark and its matching quality output. Those older quality scores
must not be relabelled as results of the new 65,000-iteration study.

Compatibility modules `_source_compat.py`, `_source_relocation.json` and
`scripts/__init__.py` preserve earlier forecast imports and source guards.
