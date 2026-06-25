"""Shared MAP optimization helpers for DTS inference routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dts._jax import configure_jax

configure_jax()

from jax import grad, hessian
import numpy as np
from scipy.optimize import Bounds, basinhopping, minimize

from dts.mcmc import make_positive_definite


class OptimizationSettings(Protocol):
    optimize: bool
    max_iter_optim: int
    gtol: float
    proposal_scale: float | None
    basinhopping: bool
    basinhopping_iter: int
    seed: int


@dataclass(frozen=True)
class MapEstimate:
    """MAP estimate and random-walk proposal covariance."""

    theta: np.ndarray
    proposal_cov: np.ndarray


def fit_map_and_proposal(
    objective,
    theta0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    settings: OptimizationSettings,
    group_id: int = 0,
    fallback_cov_scale: float = 0.05,
) -> MapEstimate:
    """Fit a MAP estimate and derive a positive-definite proposal covariance."""
    theta0 = np.asarray(theta0, dtype=float)
    n_params = len(theta0)
    theta_map = theta0
    proposal_cov = np.eye(n_params) * 0.02

    if not settings.optimize:
        return MapEstimate(theta=theta_map, proposal_cov=proposal_cov)

    hess_objective = hessian(objective)
    minimizer_kwargs = {
        "jac": grad(objective),
        "hess": hess_objective,
        "bounds": Bounds(lower, upper, keep_feasible=True),
        "method": "trust-constr",
        "options": {"gtol": settings.gtol, "maxiter": settings.max_iter_optim},
    }
    if settings.basinhopping:
        result = basinhopping(
            objective,
            x0=theta0,
            niter=settings.basinhopping_iter,
            stepsize=1.0,
            minimizer_kwargs=minimizer_kwargs,
            seed=settings.seed + group_id,
        )
    else:
        result = minimize(objective, x0=theta0, **minimizer_kwargs)

    if not result.success:
        result = minimize(
            objective,
            x0=theta0,
            jac=grad(objective),
            bounds=Bounds(lower, upper),
            method="L-BFGS-B",
            options={"maxiter": settings.max_iter_optim},
        )

    theta_map = np.asarray(result.x, dtype=float)
    try:
        target_cov = np.linalg.inv(make_positive_definite(np.asarray(hess_objective(theta_map))))
    except np.linalg.LinAlgError:
        target_cov = np.eye(n_params) * fallback_cov_scale
    scale = settings.proposal_scale or (2.38 / np.sqrt(n_params))
    proposal_cov = make_positive_definite(scale * target_cov)
    return MapEstimate(theta=theta_map, proposal_cov=proposal_cov)
