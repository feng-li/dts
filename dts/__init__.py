"""Distributed time-series inference utilities."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from dts.mcmc import ModelSpec
from dts.experiments import MCMCSettings
from dts.regression import RegressionSettings, RegressionSpec

__all__ = ["MCMCSettings", "ModelSpec", "RegressionSettings", "RegressionSpec", "__version__"]
