#!/usr/bin/env python3
"""Run a construction or validation command from HighFrequencyLiquidity."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.liquidity._construction import main


if __name__ == '__main__':
    main()
