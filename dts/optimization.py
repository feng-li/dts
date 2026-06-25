"""Shared MAP optimization helpers for DTS inference routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dts._jax import configure_jax

configure_jax()

from jax import grad, hessian, jit
import numpy as np
from scipy.optimize import Bounds, basinhopping, minimize

from dts.mcmc import make_positive_definite
from dts.progress import progress_bar


class OptimizationSettings(Protocol):
    optimize: bool
    max_iter_optim: int
    gtol: float
    proposal_scale: float | None
    basinhopping: bool
    basinhopping_iter: int
    seed: int
    progress: bool


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

    objective_jit = jit(objective)
    grad_objective = jit(grad(objective))
    hess_objective = jit(hessian(objective))

    def scipy_objective(theta):
        return float(objective_jit(theta))

    def scipy_grad(theta):
        return np.asarray(grad_objective(theta), dtype=float)

    def scipy_hess(theta):
        return np.asarray(hess_objective(theta), dtype=float)

    progress_enabled = bool(getattr(settings, "progress", False))

    def make_optimizer_callback(bar):
        last_iter = 0

        def callback(xk, state=None):
            nonlocal last_iter
            current_iter = getattr(state, "nit", None)
            if current_iter is None:
                current_iter = last_iter + 1
            current_iter = min(int(current_iter), settings.max_iter_optim)
            if current_iter > last_iter:
                bar.update(current_iter - last_iter)
                last_iter = current_iter
            return False

        return callback

    minimizer_kwargs = {
        "jac": scipy_grad,
        "hess": scipy_hess,
        "bounds": Bounds(lower, upper, keep_feasible=True),
        "method": "trust-constr",
        "options": {"gtol": settings.gtol, "maxiter": settings.max_iter_optim},
    }
    if settings.basinhopping:
        with progress_bar(
            total=settings.basinhopping_iter,
            desc=f"MAP basinhopping group {group_id}",
            unit="step",
            leave=False,
            disable=not progress_enabled,
        ) as bar:
            result = basinhopping(
                scipy_objective,
                x0=theta0,
                niter=settings.basinhopping_iter,
                stepsize=1.0,
                minimizer_kwargs=minimizer_kwargs,
                seed=settings.seed + group_id,
                callback=lambda x, f, accept: bar.update(1),
            )
    else:
        with progress_bar(
            total=settings.max_iter_optim,
            desc=f"MAP group {group_id}",
            unit="iter",
            leave=False,
            disable=not progress_enabled,
        ) as bar:
            result = minimize(
                scipy_objective,
                x0=theta0,
                callback=make_optimizer_callback(bar),
                **minimizer_kwargs,
            )

    if not result.success:
        with progress_bar(
            total=settings.max_iter_optim,
            desc=f"MAP fallback group {group_id}",
            unit="iter",
            leave=False,
            disable=not progress_enabled,
        ) as bar:
            result = minimize(
                scipy_objective,
                x0=theta0,
                jac=scipy_grad,
                bounds=Bounds(lower, upper),
                method="L-BFGS-B",
                callback=make_optimizer_callback(bar),
                options={"maxiter": settings.max_iter_optim},
            )

    theta_map = np.asarray(result.x, dtype=float)
    try:
        target_cov = np.linalg.inv(make_positive_definite(scipy_hess(theta_map)))
    except np.linalg.LinAlgError:
        target_cov = np.eye(n_params) * fallback_cov_scale
    scale = settings.proposal_scale or (2.38 / np.sqrt(n_params))
    proposal_cov = make_positive_definite(scale * target_cov)
    return MapEstimate(theta=theta_map, proposal_cov=proposal_cov)
