"""Subposterior aggregation and result transformations."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from dts.mcmc import reparam


def _as_draw_list(draws: Iterable[np.ndarray]) -> list[np.ndarray]:
    out = [np.asarray(item, dtype=float) for item in draws]
    if not out:
        raise ValueError("at least one draw array is required")
    n_cols = out[0].shape[1]
    min_rows = min(item.shape[0] for item in out)
    if any(item.ndim != 2 or item.shape[1] != n_cols for item in out):
        raise ValueError("all draw arrays must be two-dimensional with equal columns")
    return [item[:min_rows] for item in out]


def regularized_cov(draws: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Sample covariance with a small ridge for stable inversion."""
    cov = np.cov(np.asarray(draws, dtype=float), rowvar=False)
    cov = np.atleast_2d(cov)
    return cov + ridge * np.eye(cov.shape[0])


def simple_average(draws: Iterable[np.ndarray]) -> np.ndarray:
    """Average corresponding draws across shards."""
    draw_list = _as_draw_list(draws)
    return np.mean(np.stack(draw_list, axis=0), axis=0)


def consensus(draws: Iterable[np.ndarray], ridge: float = 1e-8) -> np.ndarray:
    """Precision-weighted consensus Monte Carlo aggregation."""
    draw_list = _as_draw_list(draws)
    weights = [np.linalg.inv(regularized_cov(item, ridge=ridge)) for item in draw_list]
    precision = np.sum(weights, axis=0)
    precision_inv = np.linalg.inv(precision)
    weighted_sum = np.sum([item @ weight.T for item, weight in zip(draw_list, weights)], axis=0)
    return weighted_sum @ precision_inv.T


def transform_partial_draws(
    draws: np.ndarray,
    ar_order: int,
    ma_order: int,
    tfi_term: bool,
) -> np.ndarray:
    """Transform partial AR/MA parameters and exponentiated scale parameters."""
    arr = np.asarray(draws, dtype=float)
    transformed = arr.copy()

    if ar_order > 0:
        transformed[:, :ar_order] = np.vstack(
            [reparam(row[:ar_order], MA=False) for row in arr]
        )
    if ma_order > 0:
        ma_start = ar_order
        ma_end = ar_order + ma_order
        transformed[:, ma_start:ma_end] = np.vstack(
            [reparam(row[ma_start:ma_end], MA=True) for row in arr]
        )

    if tfi_term:
        transformed[:, -3] = np.exp(arr[:, -3])
        transformed[:, -2] = np.exp(arr[:, -2])
    else:
        transformed[:, -1] = np.exp(arr[:, -1])

    return transformed


def parameter_names(
    ar_order: int,
    ma_order: int,
    tfi_term: bool,
    transformed: bool = True,
) -> list[str]:
    """Column names for parameter draws."""
    names = [f"phi{i + 1}" for i in range(ar_order)] + [
        f"theta{i + 1}" for i in range(ma_order)
    ]
    if tfi_term:
        names.extend(["lambda" if transformed else "log_lambda", "sigma2" if transformed else "log_sigma2", "d"])
    else:
        names.append("sigma2" if transformed else "log_sigma2")
    return names


def credible_interval(draws: np.ndarray, probs: Sequence[float] = (0.025, 0.5, 0.975)) -> np.ndarray:
    """Column-wise posterior quantiles."""
    return np.quantile(np.asarray(draws, dtype=float), probs, axis=0).T


def wasserstein_quantile_distance(reference: np.ndarray, candidate: np.ndarray, grid_size: int = 200) -> np.ndarray:
    """Normalized one-dimensional Wasserstein distance by marginal quantiles."""
    probs = np.linspace(0.0, 1.0, grid_size)
    ref_q = np.quantile(reference, probs, axis=0)
    cand_q = np.quantile(candidate, probs, axis=0)
    scale = np.maximum(np.ptp(ref_q, axis=0), 1e-12)
    return np.mean(np.abs(ref_q - cand_q), axis=0) / scale


def wasserstein_barycenter(draws: Iterable[np.ndarray], nsamp: int = 5000) -> np.ndarray:
    """Merge chains by averaging marginal quantile functions.

    This is the Wasserstein-barycenter helper from the AR(2) comparison script,
    made reusable for diagnostics and DC-BATS comparisons.
    """
    draw_list = _as_draw_list(draws)
    n_params = draw_list[0].shape[1]
    probs = np.linspace(0.0, 1.0, nsamp)
    merged = np.zeros((nsamp, n_params))
    for param in range(n_params):
        quantiles = np.asarray([np.quantile(chain[:, param], probs) for chain in draw_list])
        merged[:, param] = quantiles.mean(axis=0)
    return merged
