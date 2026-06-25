"""High-level runners for the manuscript replication scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from autograd import grad, hessian
from scipy.optimize import Bounds, basinhopping, minimize

from dts.aggregation import consensus, simple_average
from dts.mcmc import (
    ModelSpec,
    log_prior,
    make_positive_definite,
    parameter_bounds,
    sampler,
    whittle_log_likelihood,
)
from dts.partition import frequency_domain, shard_frequency_domain, time_partition_indices


@dataclass(frozen=True)
class MCMCSettings:
    n_samples: int = 15000
    burn_in: int = 5000
    seed: int = 123
    optimize: bool = True
    max_iter_optim: int = 500
    gtol: float = 1e-4
    proposal_scale: Optional[float] = None
    basinhopping: bool = False
    basinhopping_iter: int = 25


@dataclass
class ShardResult:
    draws: np.ndarray
    log_p: np.ndarray
    acceptance: np.ndarray
    map_estimate: np.ndarray
    proposal_cov: np.ndarray
    group_id: int = 0

    @property
    def acceptance_rate(self) -> float:
        return float(np.mean(self.acceptance)) if len(self.acceptance) else 0.0


@dataclass
class DistributedResult:
    full: Optional[ShardResult]
    shards: list[ShardResult]
    average_draws: np.ndarray
    consensus_draws: np.ndarray
    partition: str
    n_groups: int


def load_series(path: str | Path) -> np.ndarray:
    """Load a one-dimensional time series from txt, csv, or npy."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=float).reshape(-1)
    if path.suffix == ".csv":
        data = np.genfromtxt(path, delimiter=",", names=True)
        if data.dtype.names:
            return np.asarray(data[data.dtype.names[-1]], dtype=float).reshape(-1)
        return np.asarray(data, dtype=float).reshape(-1)
    return np.asarray(np.loadtxt(path), dtype=float).reshape(-1)


def initial_params(model: ModelSpec, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 0.01 * np.ones(model.n_params) + rng.normal(0.0, 0.1, size=model.n_params)


def whittle_log_posterior(params, model: ModelSpec, periodogram, omega, n_groups: int = 1):
    return log_prior(
        params,
        0.0,
        1.0,
        model.last_arma,
        model.tfi_term,
        n_groups=n_groups,
        check_bounds=False,
    ) + np.sum(whittle_log_likelihood(params, model.q, model.p, periodogram, model.tfi_term, omega))


def fit_whittle_shard(
    periodogram: np.ndarray,
    omega: np.ndarray,
    model: ModelSpec,
    settings: MCMCSettings,
    n_groups: int = 1,
    group_id: int = 0,
) -> ShardResult:
    """Fit one Whittle subposterior."""
    theta0 = initial_params(model, seed=settings.seed + group_id)

    def logp(theta):
        return whittle_log_posterior(theta, model, periodogram, omega, n_groups=n_groups)

    lower, upper = parameter_bounds(model.n_params, model.tfi_term)
    proposal_cov = np.eye(model.n_params) * 0.02
    theta_map = theta0

    if settings.optimize:
        def objective(theta):
            return -logp(theta)

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
            target_cov = np.linalg.inv(make_positive_definite(hess_objective(theta_map)))
        except np.linalg.LinAlgError:
            target_cov = np.eye(model.n_params) * 0.05
        scale = settings.proposal_scale or (2.38 / np.sqrt(model.n_params))
        proposal_cov = make_positive_definite(scale * target_cov)

    draws, log_p, acceptance = sampler(
        model.q,
        model.p,
        data=None,
        I_pg=periodogram,
        TFI_term=model.tfi_term,
        omega_shard=omega,
        n_samples=settings.n_samples,
        paramsStar=theta_map,
        proposal_width=proposal_cov,
        Burn_in=settings.burn_in,
        exact=False,
        n_groups=n_groups,
        random_state=settings.seed + 10_000 + group_id,
    )
    return ShardResult(draws, log_p, acceptance, theta_map, proposal_cov, group_id=group_id)


def fit_full_whittle(data: np.ndarray, model: ModelSpec, settings: MCMCSettings) -> ShardResult:
    values, omega = frequency_domain(data)
    return fit_whittle_shard(values, omega, model, settings, n_groups=1, group_id=0)


def fit_frequency_divide_and_conquer(
    data: np.ndarray,
    model: ModelSpec,
    settings: MCMCSettings,
    n_groups: int = 10,
    partition: str = "systematic",
    include_full: bool = True,
) -> DistributedResult:
    """Run Whittle MCMC on frequency shards and aggregate draws."""
    shards = []
    for group_id, (_, shard_periodogram, shard_omega) in enumerate(
        shard_frequency_domain(data, n_groups=n_groups, method=partition)
    ):
        shards.append(
            fit_whittle_shard(
                shard_periodogram,
                shard_omega,
                model,
                settings,
                n_groups=n_groups,
                group_id=group_id,
            )
        )

    full = fit_full_whittle(data, model, settings) if include_full else None
    shard_draws = [item.draws for item in shards]
    return DistributedResult(
        full=full,
        shards=shards,
        average_draws=simple_average(shard_draws),
        consensus_draws=consensus(shard_draws),
        partition=partition,
        n_groups=n_groups,
    )


def fit_time_domain_as_frequency_shards(
    data: np.ndarray,
    model: ModelSpec,
    settings: MCMCSettings,
    n_groups: int = 10,
    include_full: bool = True,
) -> DistributedResult:
    """Naive time-domain splitting followed by local Whittle fits.

    This reproduces the comparison logic in the paper: each shorter segment is
    analyzed independently, which loses global low-frequency information.
    """
    shards = []
    for group_id, indices in enumerate(time_partition_indices(len(data), n_groups)):
        values, omega = frequency_domain(data[indices])
        shards.append(
            fit_whittle_shard(
                values,
                omega,
                model,
                settings,
                n_groups=n_groups,
                group_id=group_id,
            )
        )

    full = fit_full_whittle(data, model, settings) if include_full else None
    shard_draws = [item.draws for item in shards]
    return DistributedResult(
        full=full,
        shards=shards,
        average_draws=simple_average(shard_draws),
        consensus_draws=consensus(shard_draws),
        partition="time",
        n_groups=n_groups,
    )
