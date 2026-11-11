#!/usr/bin/env python3
"""Compatibility entry point for construction in the HighFrequencyLiquidity submodule."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.liquidity._construction import exec_construction


def main():
    exec_construction('build_all_csi300_liquidity')


if __name__ == '__main__':
    main()
