# Changes

## Jul 14, 2026: Indexing corrections and script cleanup

- Aligned the implementation with the revised manuscript indexing: Whittle
  likelihood calculations now use the positive Fourier-frequency indices
  `1, ..., floor((T - 1) / 2)` and exclude the zero frequency.
- Corrected systematic frequency partitioning so the paper's one-based rule
  `{g, g + G, g + 2G, ...}` corresponds to zero-based code positions
  `g - 1, g - 1 + G, g - 1 + 2G, ...`; this was checked across the active
  Python path, Spark/DFFT helpers, AR(2) regression comparison, and legacy
  example scripts.
- Revised the distributed FFT manuscript description to keep notation
  consistent: `P` denotes DFFT partitions, `G` denotes Whittle groups, and
  one-based paper indices are mapped explicitly to zero-based implementation
  indices.
- Moved standalone legacy plotting and prototype scripts under `scripts/`
  or `legacy/` so the active package remains centered on `dts/`, while older
  experimental material is still retained for traceability.

## Jun 26, 2026: Main improvements since migration

- Moved the active implementation into the `dts/` package with runnable entry points in `scripts/`, while retaining legacy sources for traceability.
- Accelerated the replication path with JAX-backed Whittle posterior evaluation, MAP/proposal optimization, cached full fits, and reduced redundant regression shard work.
- Added progress bars for long MCMC, optimization, shard, and replication loops, with `--no-progress` for batch logs.
- Added automatic shard execution selection: `--backend auto` uses Ray when `--num-cpus > 1` and local execution when `--num-cpus 1`; Spark remains available only for Spark-specific workflows.
- Documented quick and paper replication commands, plus Ray/local backend controls, in the README.
