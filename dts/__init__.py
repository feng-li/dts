"""Public API for frequency-domain divide-and-conquer time-series inference."""

from importlib import import_module

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

_LAZY_EXPORTS = {
    "DistributedResult": ("dts.experiments", "DistributedResult"),
    "MCMCSettings": ("dts.experiments", "MCMCSettings"),
    "ModelSpec": ("dts.mcmc", "ModelSpec"),
    "RegressionSettings": ("dts.regression", "RegressionSettings"),
    "RegressionSpec": ("dts.regression", "RegressionSpec"),
    "consensus": ("dts.aggregation", "consensus"),
    "credible_interval": ("dts.aggregation", "credible_interval"),
    "fit_frequency_divide_and_conquer": ("dts.experiments", "fit_frequency_divide_and_conquer"),
    "fit_full_whittle": ("dts.experiments", "fit_full_whittle"),
    "fit_regression_frequency_divide_and_conquer": (
        "dts.regression",
        "fit_regression_frequency_divide_and_conquer",
    ),
    "frequency_domain": ("dts.partition", "frequency_domain"),
    "load_series": ("dts.experiments", "load_series"),
    "parameter_names": ("dts.aggregation", "parameter_names"),
    "shard_frequency_domain": ("dts.partition", "shard_frequency_domain"),
    "simple_average": ("dts.aggregation", "simple_average"),
    "transform_partial_draws": ("dts.aggregation", "transform_partial_draws"),
}


def __getattr__(name: str):
    """Load public exports lazily so lightweight submodules stay lightweight."""
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'dts' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value

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
