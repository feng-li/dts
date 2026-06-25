"""Runtime helpers shared by command-line scripts."""

from __future__ import annotations

import warnings

import numpy as np


def suppress_numeric_warnings() -> None:
    """Suppress known NumPy/SciPy warning noise in replication scripts."""
    complex_warning = getattr(np, "ComplexWarning", None)
    if complex_warning is None:
        complex_warning = getattr(getattr(np, "exceptions", object), "ComplexWarning", Warning)
    warnings.filterwarnings("ignore", category=complex_warning)
    warnings.filterwarnings("ignore", message=".*where.*without.*out.*", category=UserWarning)
