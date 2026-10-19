# CSI 300 liquidity application

## Repository layout

| Location | Responsibility |
| --- | --- |
| `HighFrequencyLiquidity/` | Optional pinned Git submodule: weights and source inputs, order-book replay, liquidity construction and validation |
| `scripts/liquidity/` | DTS modelling, rolling forecasts, benchmarks, reports and paper figures |
| `tests/` | DTS inference, forecasting and submodule-interface checks |
| `~/running/dts/` | Generated indices, prepared arrays, fit checkpoints and analysis outputs |

Construction has one maintained implementation in the submodule. The former
`projects/liquidity/` folder has been removed. Historical local prototypes were
preserved in `~/running/dts/archives/dts-liquidity-before-submodule-20261019.tar.gz`;
they are outside the current paper workflow.

## Forecast from supplied data

The forecasting code runs entirely in DTS. It does not import construction
code, read stock weights or access Level 2 records. You can leave the
`HighFrequencyLiquidity/` submodule uninitialised.

Use a normal clone and install DTS:

```bash
git clone git@github.com:feng-li/dts.git
cd dts
python -m pip install -e .
```

The existing CSI 300 experiment expects a prepared input directory:

| File | Contents |
| --- | --- |
| `spread.npy` | Floating-point array with one row per date and 14,220 one-second positions per row; missing values are NaN |
| `input_manifest.json` | Ordered dates, array shape, target `liq_available_vw_bps` and size `notional_1e+06`, plus input provenance |
| `contiguous_runs.csv` | Observed within-session runs, including `day_index`, `date`, `session`, `start`, `stop` and `n` |

Each row uses 09:30:01--11:30:00 followed by 13:00:01--14:57:00. Supplied
data must follow this experiment's grid and schema; missing intervals remain
explicit. The construction weights and raw stock records are not forecast inputs.

For example, create a rolling plan from already supplied data:

```bash
python scripts/liquidity/plan_csi300_boundary_forecasts.py \\
  --input /path/to/prepared_liquidity \\
  --out ~/running/dts/artifacts/liquidity/my_rolling_plan \\
  --evaluation-days 1000 --date-policy usable
```

The paper experiment retains its requirement of at least ten million earlier
training changes. The daily Ray runner accepts this input and plan through
`--input` and `--plan`. The current Spark runner reuses a saved DTS reference
experiment through `--reference`; its reference fits, run configuration and
rolling tasks are modelling artifacts, independent of the construction submodule.
Existing saved forecast configurations can be resumed without the submodule.

## Optional construction submodule

This step is needed only to construct liquidity from Level 2 records, audit
construction inputs, or export a daily index archive into the prepared format.

From the DTS repository root:

```bash
git submodule update --init --recursive
```

A fresh checkout can use:

```bash
git clone --recurse-submodules git@github.com:feng-li/dts.git
```

The parent repository pins an exact construction commit. `git submodule update`
uses that revision. Advancing construction requires reviewing and staging a new
submodule revision. An explicit `HFL_REPO` or launcher `--hfl-repo PATH` can
select a development checkout; the default is the root submodule.

The submodule includes its [weight/source inputs](HighFrequencyLiquidity/data/csi300/README.md),
[source manifest](HighFrequencyLiquidity/data/csi300/SOURCES.json),
[input contracts](HighFrequencyLiquidity/docs/DATA.md) and
[construction scripts](HighFrequencyLiquidity/docs/SCRIPTS.md).
Raw Level 2 archives remain external, normally at `/data/data/level2data`.
The actual daily-weight adapter and timing reconciliation are recorded in the
[submodule TODO](HighFrequencyLiquidity/TODO.md); adding the submodule does not
switch the existing monthly-weight index to actual daily weights.

## Construct and export

Run from the DTS root using an interpreter with the DTS and construction
dependencies. The launcher starts a fresh Python process to keep the two
repositories' `scripts` packages separate:

```bash
python scripts/liquidity/run_construction.py build_all_csi300_liquidity \
  --start 20230103 --end 20230103 --ray-actors 80 --dry-run
```

Remove `--dry-run` to construct the selected date; omit the date limits to visit
the archive. Ray actors process dates concurrently and stocks serially inside
each date. Construction retains its per-stock/day cache checks, coverage gates
and explicit failure records.

Use `run_construction.py --help` to list other construction, weight and audit
commands. Its optional `--hfl-repo PATH` precedes the command name. Options after
the command are passed to the submodule unchanged; relative paths retain the
calling directory. Interrupts and exit status reach the construction process.

Input defaults come from `HighFrequencyLiquidity/data/csi300`. Output defaults
remain under `~/running/dts/artifacts/liquidity`. `HFL_DATA_ROOT`,
`HFL_OUTPUT_ROOT` and explicit script options override these paths.
The existing batch/daily builder filenames remain thin CLI compatibility
wrappers. Their former construction functions are owned by the submodule.

Export an existing daily archive for DTS:

```bash
python scripts/liquidity/prepare_csi300_dc.py \
  --root ~/running/dts/artifacts/liquidity/csi300_monthly_imputed_1000ms \
  --out ~/running/dts/artifacts/liquidity/csi300_dc_1m
```

The exporter also runs in the submodule. It retains the established DTS output
name and contract: `spread.npy`, `input_manifest.json` and
`contiguous_runs.csv`. `1m` means **RMB 1 million**, at one-second resolution.
Missing observations and AM/PM boundaries remain explicit.

Keep the prepared array and manifest unchanged when resuming saved forecasts.
A changed index or weight convention needs a separate export and matching
rolling plan; construction migration does not bypass scientific source guards.

## Forecasting scripts

All filenames below are under `scripts/liquidity/`.

| Script | Purpose |
| --- | --- |
| `plan_csi300_boundary_forecasts.py` | Plans daily expanding-history forecasts around opening, lunch, reopening and the sampled closing window; requires at least ten million earlier training changes |
| `compare_csi300_boundary_spark.py` | Current Spark Spectral DC fit, covariance consensus, posterior checks and forecasting; resumable histories, FFT groups and sampling blocks |
| `compare_csi300_boundary_random_days.py` | Original Ray daily experiment and matched forecast conditioning/scoring functions; saved G=2 reference experiment |
| `compare_csi300_boundary_arima110.py` | ARIMA(1,1,0) benchmark on the saved forecast tasks |
| `compare_csi300_boundary_arima101.py` | ARIMA(1,0,1) benchmark on the saved forecast tasks |
| `compare_csi300_forecasts.py` | Earlier quarterly comparison; shared forecast kernels, target checks and predictive scores |
| `probe_csi300_fractional_changes.py` | Native fractional-change target, MAP search, clock calibration and forecasting helpers |
| `run_csi300_fractional_changes_dc.py` | Development exact-target DC fit and compiled likelihood/sampling helpers |
| `fit_csi300_observed_dc.py` | Earlier observed-time Whittle fit and shared sampling diagnostics |
| `report_csi300_boundary_random_days.py` | Original daily experiment's score and posterior-diagnostic report |
| `report_csi300_forecast_comparison.py` | Quarterly comparison's model-fit and forecast report |
| `summarize_csi300_fractional_framework.py` | Development comparison, paired uncertainty summaries and TeX macros |

### Current G=20 experiment

Resume the completed production specification with its saved configuration:

```bash
JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
PYSPARK_SUBMIT_ARGS='--driver-memory 64g pyspark-shell' \
/home/fli/.virtualenvs/py3.12-spark4/bin/python -u \
  scripts/liquidity/compare_csi300_boundary_spark.py \
  --config ~/running/dts/artifacts/liquidity/csi300_boundary_spark_g20_full1000_onechain_v5/run_config.json
```

The saved configuration determines G, chain count and numerical settings.
Production uses G=20 and one chain per shard; independent-chain diagnostics
are requested separately. Every assigned frequency and 1/G of the induced
prior enter the shard target. The fixed proposal and checked covariance
regularisation are implemented by `_spectral_coordinates.py`,
`_spectral_sampling.py` and `_spectral_covariance.py`.

Completed histories are reused. Interrupted sampling resumes from its last
atomic block, retaining RNG state and fixed proposal parameters. A missing
metadata sidecar is recovered only after the saved arrays pass their checks.
Source/configuration guards continue to reject incompatible model changes.
No likelihood or forecasting code was changed by the submodule consolidation.

`--history-workers 0` chooses concurrent histories from available Spark slots
and G. Fresh full-data reference dates must pass their checks before the
remaining histories run. One-chain ESS checks do not establish independent-chain
convergence. See the script's `--help` and saved configuration for all settings.

## Paper figures and descriptive reports

| Script | Purpose |
| --- | --- |
| `plot_csi300_liquidity_session_profile.py` | Figure 1: one-second train/test spread scatter with the saved sample split |
| `plot_csi300_liquidity_forecast_errors.py` | Figure 2: RMSE reduction relative to matched ARIMA, by origin/horizon, with paired block-bootstrap intervals; supports `--ray-actors` |
| `report_csi300_session_boundaries.py` | Descriptive trading-clock profiles used by the application plots |
| `plot_csi300_liquidity_application.py` | Earlier application figure from saved daily forecasts |

Use each plotting script's arguments to select the saved forecast run; plotting
does not refit the model. Manuscript TeX and figure PDFs remain real files in
`docs/` and `docs/figures/`. Other reports, score tables and plot protocols belong
under `~/running/dts/`.

## Tests

DTS tests are collected from `tests/` and run without the construction submodule.
The forecast-independence test uses a source checkout containing only DTS code
and supplied data. Construction integration tests skip their real-submodule
cases when it is uninitialised.

Construction tests are optional and live in the submodule. Run the two suites
in separate interpreters so their common package names stay isolated:

```bash
python -m pytest tests
(cd HighFrequencyLiquidity && python -m pytest -m 'not ray')
```

The interface tests exercise the real submodule exporter, preserve missing
seconds/session boundaries, and feed its output to the DTS rolling planner.
The generic dispatcher and three compatibility entry points test path overrides,
namespace isolation, argument forwarding and exit status.

`_source_compat.py`, `_source_relocation.json` and `scripts/__init__.py` retain
compatibility with earlier forecasting module names and their recorded hashes.
They do not accept arbitrary numerical changes or repurpose old posteriors.
