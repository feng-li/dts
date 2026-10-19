"""Command-line workflows, including compatibility imports for moved scripts."""
from pathlib import Path

# Cached Ray/Numba objects can still refer to scripts.<old liquidity module>.
# Resolve those imports to the same source files without duplicate entry points.
__path__.append(str(Path(__file__).resolve().parent / 'liquidity'))
