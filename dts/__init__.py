"""Public API for frequency-domain divide-and-conquer time-series inference."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from dts.aggregation import (
    consensus,
    credible_interval,
    parameter_names,
    simple_average,
    transform_partial_draws,
)
from dts.experiments import (
    DistributedResult,
    MCMCSettings,
    fit_frequency_divide_and_conquer,
    fit_full_whittle,
    load_series,
)
from dts.mcmc import ModelSpec
from dts.partition import frequency_domain, shard_frequency_domain
from dts.regression import (
    RegressionSettings,
    RegressionSpec,
    fit_regression_frequency_divide_and_conquer,
)

__all__ = [
    "DistributedResult",
    "MCMCSettings",
    "ModelSpec",
    "RegressionSettings",
    "RegressionSpec",
    "__version__",
    "consensus",
    "credible_interval",
    "fit_frequency_divide_and_conquer",
    "fit_full_whittle",
    "fit_regression_frequency_divide_and_conquer",
    "frequency_domain",
    "load_series",
    "parameter_names",
    "shard_frequency_domain",
    "simple_average",
    "transform_partial_draws",
]
