# Changes

## Jun 26, 2026: Main improvements since migration

- Moved the active implementation into the `dts/` package with runnable entry points in `scripts/`, while retaining legacy sources for traceability.
- Accelerated the replication path with JAX-backed Whittle posterior evaluation, MAP/proposal optimization, cached full fits, and reduced redundant regression shard work.
- Added progress bars for long MCMC, optimization, shard, and replication loops, with `--no-progress` for batch logs.
- Added automatic shard execution selection: `--backend auto` uses Ray when `--num-cpus > 1` and local execution when `--num-cpus 1`; Spark remains available only for Spark-specific workflows.
- Documented quick and paper replication commands, plus Ray/local backend controls, in the README.
