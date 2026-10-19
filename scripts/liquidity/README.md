# Liquidity forecasting experiments

This folder contains the DTS CSI 300 modelling, rolling forecasts, benchmarks,
reports and paper figures. Construction, weights, replay and construction tests
are maintained in the root [`HighFrequencyLiquidity/`](../../HighFrequencyLiquidity/README.md)
Git submodule, which is optional when forecasting from supplied data.

## Forecast from given data

Use the supplied `spread.npy`, `input_manifest.json` and `contiguous_runs.csv`.
The forecasting scripts have no dependency on HighFrequencyLiquidity:

1. Create the rolling plan with `plan_csi300_boundary_forecasts.py --input PATH`.
2. Run the daily forecasting experiment or resume the saved Spark configuration.
3. Add ARIMA benchmarks and regenerate application figures from saved forecasts.

The daily runner accepts `--input` and `--plan`. The Spark runner uses the saved
DTS reference experiment selected by `--reference`. Its modelling checkpoints
are separate from liquidity construction. A normal Git clone is sufficient.

## Optional construction and export

To build liquidity from Level 2 records or export a daily index archive,
initialise the pinned construction source from the DTS root:

```bash
git submodule update --init --recursive
```

Use `run_construction.py COMMAND [OPTIONS]` for construction/audits, then
`prepare_csi300_dc.py` to export the forecasting input directory.

The three construction compatibility entry points are thin wrappers:
`build_all_csi300_liquidity.py`, `build_csi300_hf_liquidity.py` and
`prepare_csi300_dc.py`. Their implementation runs inside the submodule in a
fresh Python process. Inputs default to the submodule's `data/csi300`; outputs
remain under `~/running/dts/artifacts/liquidity`. `HFL_REPO` selects an explicit
development checkout when needed.

Forecasting code, settings and input/checkpoint checks retain their existing
paths. Preserve saved arrays and manifests when resuming a fit. The former
`projects/liquidity/` prototypes were archived before removal.

See [LIQUIDITY.md](../../LIQUIDITY.md) for the script catalog, construction/export
interface, current G=20 resume command, source compatibility and test commands.
