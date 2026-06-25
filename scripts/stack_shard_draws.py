#!/usr/bin/env python3
"""Stack shard-level ``*.npy`` MCMC artifacts into ``draws_all.npy`` files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dts.artifacts import stack_shard_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--groups", type=int, default=None)
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draws_all, logp_all = stack_shard_artifacts(args.artifact_dir, n_groups=args.groups, save=not args.no_save)
    print(f"draws_all: {draws_all.shape}")
    print(f"logp_all : {logp_all.shape}")


if __name__ == "__main__":
    main()
