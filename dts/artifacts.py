"""Helpers for reading and stacking shard-level MCMC artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def stack_shard_artifacts(
    artifact_dir: str | Path,
    n_groups: int | None = None,
    save: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack ``shardXX_draws.npy`` and ``shardXX_logp.npy`` files.

    Returns arrays with shapes ``(G, n_samples, n_params)`` and
    ``(G, n_samples)``. When ``save`` is true, also writes ``draws_all.npy`` and
    ``logp_all.npy`` in the artifact directory.
    """
    artifact_dir = Path(artifact_dir)
    if n_groups is None:
        shard_files = sorted(artifact_dir.glob("shard*_draws.npy"))
        n_groups = len(shard_files)
    if n_groups < 1:
        raise ValueError("no shard artifacts found")

    draws_list = []
    logp_list = []
    for shard_id in range(n_groups):
        draws_path = artifact_dir / f"shard{shard_id:02d}_draws.npy"
        logp_path = artifact_dir / f"shard{shard_id:02d}_logp.npy"
        if not draws_path.exists() or not logp_path.exists():
            raise FileNotFoundError(f"missing shard files for shard {shard_id:02d} in {artifact_dir}")
        draws_list.append(np.load(draws_path))
        logp_list.append(np.load(logp_path))

    draws_all = np.stack(draws_list, axis=0)
    logp_all = np.stack(logp_list, axis=0)
    if save:
        np.save(artifact_dir / "draws_all.npy", draws_all)
        np.save(artifact_dir / "logp_all.npy", logp_all)
    return draws_all, logp_all
