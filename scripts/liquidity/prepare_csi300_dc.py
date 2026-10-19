#!/usr/bin/env python3
"""Export forecasting inputs using the maintained HighFrequencyLiquidity code.

The established --root, --out and --workers options are passed through. DTS
defaults still select csi300_dc_1m under ~/running/dts/artifacts/liquidity.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.liquidity._construction import exec_construction


def main():
    exec_construction('prepare_csi300_dc')


if __name__ == '__main__':
    main()
