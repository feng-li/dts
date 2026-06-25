"""Frequency-domain regression models with ARMA/ARTFIMA errors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from jax import grad, hessian
import jax.numpy as np
import jax.scipy.stats as sps_jax
import numpy as onp
from scipy.optimize import Bounds, basinhopping, minimize

from dts.aggregation import consensus, simple_average
from dts.mcmc import f_ARTFIMA, make_positive_definite, reparam
from dts.partition import frequency_partition_indices, time_partition_indices


@dataclass(frozen=True)
class RegressionSpec:
    """Regression model with ARMA/ARTFIMA residuals.

    Parameter order is ``[ar_partial..., ma_partial..., beta..., tail...]``.
    The tail is ``[log_sigma2]`` for ARMA errors and
    ``[log_lambda, log_sigma2, d]`` for ARTFIMA errors.
    """

    q: int = 2
    p: int = 0
    n_exog: int = 1
    tfi_term: bool = False

    @property
    def n_params(self) -> int:
        return self.q + self.p + self.n_exog + (3 if self.tfi_term else 1)

    @property
    def beta_slice(self) -> slice:
        start = self.q + self.p
        return slice(start, start + self.n_exog)

    @property
    def process_slice(self) -> slice:
        return slice(0, self.q + self.p)


@dataclass(frozen=True)
class RegressionSettings:
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
class RegressionShardResult:
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
class RegressionDistributedResult:
    full: RegressionShardResult | None
    shards: list[RegressionShardResult]
    average_draws: np.ndarray
    consensus_draws: np.ndarray
    n_groups: int


def load_ar2_regression(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``[x, y, residual]`` style AR(2) regression data."""
    arr = onp.load(path, allow_pickle=True)
    if arr.ndim == 2 and arr.shape[0] >= 2:
        x = onp.asarray(arr[0], dtype=float)
        y = onp.asarray(arr[1], dtype=float)
    else:
        raise ValueError(f"unsupported AR(2) regression data shape: {arr.shape}")
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x, y.reshape(-1)


def simulate_ar2_regression(
    n: int,
    beta: float = 2.0,
    phi1: float = 1.942,
    phi2: float = -0.943,
    sigma2: float = 1.0,
    burnin: int = 100_000,
    seed: int | None = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the persistent AR(2) regression benchmark from the paper."""
    rng = onp.random.default_rng(seed)
    total_n = n + burnin
    residual = onp.zeros(total_n)
    noise = rng.normal(scale=onp.sqrt(sigma2), size=total_n)
    for t in range(total_n):
        if t == 0:
            residual[t] = noise[t]
        elif t == 1:
            residual[t] = phi1 * residual[t - 1] + noise[t]
        else:
            residual[t] = phi1 * residual[t - 1] + phi2 * residual[t - 2] + noise[t]
    residual = residual[burnin:]
    x = rng.normal(size=(n, 1))
    y = beta * x[:, 0] + residual
    return x, y, residual


def regression_parameter_names(spec: RegressionSpec, transformed: bool = True) -> list[str]:
    names = [f"phi{i + 1}" for i in range(spec.q)] + [f"theta{i + 1}" for i in range(spec.p)]
    names.extend([f"beta{i + 1}" if spec.n_exog > 1 else "beta" for i in range(spec.n_exog)])
    if spec.tfi_term:
        names.extend(["lambda" if transformed else "log_lambda", "sigma2" if transformed else "log_sigma2", "d"])
    else:
        names.append("sigma2" if transformed else "log_sigma2")
    return names


def transform_regression_draws(draws: np.ndarray, spec: RegressionSpec) -> np.ndarray:
    arr = onp.asarray(draws, dtype=float)
    out = arr.copy()
    if spec.q:
        out[:, : spec.q] = onp.vstack([reparam(row[: spec.q], MA=False) for row in arr])
    if spec.p:
        start = spec.q
        end = spec.q + spec.p
        out[:, start:end] = onp.vstack([reparam(row[start:end], MA=True) for row in arr])
    if spec.tfi_term:
        out[:, -3] = onp.exp(arr[:, -3])
        out[:, -2] = onp.exp(arr[:, -2])
    else:
        out[:, -1] = onp.exp(arr[:, -1])
    return out


def regression_frequency_domain(x: np.ndarray, y: np.ndarray):
    x = onp.asarray(x, dtype=float)
    y = onp.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return onp.fft.fft(x, axis=0), onp.fft.fft(y)


def residual_periodogram(y_hat, x_hat, beta):
    residual_hat = y_hat - np.dot(x_hat, beta)
    return np.square(np.abs(residual_hat)) / (2.0 * np.pi * len(y_hat))


def regression_log_prior(
    params,
    spec: RegressionSpec,
    n_groups: int = 1,
    beta_sd: float = 100.0,
    tail_sd: float = 100.0,
    lambda_mu: float = -2.3,
    lambda_sd: float = 0.4,
):
    process = params[spec.process_slice]
    prior_process = np.where(
        np.all(np.abs(process) < 1.0),
        -len(process) * np.log(2.0),
        -np.inf,
    )

    beta = params[spec.beta_slice]
    prior_beta = np.sum(sps_jax.norm.logpdf(beta, loc=0.0, scale=beta_sd))
    if spec.tfi_term:
        prior_tail = (
            sps_jax.norm.logpdf(params[-1], loc=0.0, scale=tail_sd)
            + sps_jax.norm.logpdf(params[-3], loc=lambda_mu, scale=lambda_sd)
            + sps_jax.norm.logpdf(params[-2], loc=0.0, scale=tail_sd)
        )
    else:
        prior_tail = sps_jax.norm.logpdf(params[-1], loc=0.0, scale=tail_sd)
    return (prior_process + prior_beta + prior_tail) / n_groups


def regression_whittle_log_likelihood(params, y_hat, x_hat, spec: RegressionSpec, indices):
    beta = params[spec.beta_slice]
    if spec.tfi_term:
        d = params[-1]
        lambda_ = np.exp(params[-3])
        var = np.exp(params[-2])
    else:
        d = 0.0
        lambda_ = 0.0
        var = np.exp(params[-1])

    phi = reparam(params[: spec.q], MA=False) if spec.q else np.array([])
    theta = reparam(params[spec.q : spec.q + spec.p], MA=True) if spec.p else np.array([])
    omega = 2.0 * np.pi * np.asarray(indices) / len(y_hat)
    density = f_ARTFIMA(omega, phi, theta, var, d, lambda_)
    periodogram = residual_periodogram(y_hat, x_hat, beta)
    return -(np.log(density) + periodogram[indices] / density)


def regression_log_posterior(params, y_hat, x_hat, spec: RegressionSpec, indices, n_groups: int = 1):
    return np.sum(regression_whittle_log_likelihood(params, y_hat, x_hat, spec, indices)) + regression_log_prior(
        params, spec, n_groups=n_groups
    )


def regression_parameter_bounds(spec: RegressionSpec):
    lower = -onp.ones(spec.n_params) * 5.0
    upper = onp.ones(spec.n_params) * 5.0
    lower[spec.process_slice] = -1.0
    upper[spec.process_slice] = 1.0
    return lower, upper


def initial_regression_params(spec: RegressionSpec, seed: int = 123):
    rng = onp.random.default_rng(seed)
    return 0.1 * onp.ones(spec.n_params) + rng.normal(0.0, 0.05, size=spec.n_params)


def _valid_regression_process(theta, spec: RegressionSpec) -> bool:
    return bool(onp.all(onp.abs(onp.asarray(theta[spec.process_slice], dtype=float)) < 1.0))


def regression_sampler(
    theta_init,
    proposal_cov,
    logp_fn,
    spec: RegressionSpec,
    settings: RegressionSettings,
    group_id: int = 0,
):
    rng = onp.random.default_rng(settings.seed + 20_000 + group_id)
    theta_current = onp.asarray(theta_init, dtype=float)
    proposal_cov = make_positive_definite(proposal_cov)
    draws = onp.zeros((settings.n_samples, spec.n_params))
    log_p = onp.zeros(settings.n_samples)
    acceptance = onp.zeros(settings.n_samples, dtype=bool)
    logp_current = float(logp_fn(theta_current))
    for i in range(settings.n_samples):
        theta_proposal = rng.multivariate_normal(theta_current, proposal_cov)
        if _valid_regression_process(theta_proposal, spec):
            logp_proposal = float(logp_fn(theta_proposal))
        else:
            logp_proposal = -onp.inf
        if onp.log(rng.random()) < min(0.0, logp_proposal - logp_current):
            theta_current = theta_proposal
            logp_current = logp_proposal
            acceptance[i] = True
        draws[i] = theta_current
        log_p[i] = logp_current
    return draws[settings.burn_in :], log_p[settings.burn_in :], acceptance[settings.burn_in :]


def fit_regression_whittle_shard(
    y_hat,
    x_hat,
    indices: np.ndarray,
    spec: RegressionSpec,
    settings: RegressionSettings,
    n_groups: int = 1,
    group_id: int = 0,
) -> RegressionShardResult:
    theta0 = initial_regression_params(spec, seed=settings.seed + group_id)

    def logp(theta):
        return regression_log_posterior(theta, y_hat, x_hat, spec, indices, n_groups=n_groups)

    theta_map = theta0
    proposal_cov = onp.eye(spec.n_params) * 0.02
    lower, upper = regression_parameter_bounds(spec)

    if settings.optimize:
        def objective(theta):
            return -logp(theta)

        hess_objective = hessian(objective)
        minimizer_kwargs = {
            "method": "trust-constr",
            "jac": grad(objective),
            "hess": hess_objective,
            "bounds": Bounds(lower, upper, keep_feasible=True),
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
        theta_map = onp.asarray(result.x, dtype=float)
        try:
            cov = onp.linalg.inv(make_positive_definite(onp.asarray(hess_objective(theta_map))))
        except onp.linalg.LinAlgError:
            cov = onp.eye(spec.n_params) * 0.05
        scale = settings.proposal_scale or (2.38 / onp.sqrt(spec.n_params))
        proposal_cov = make_positive_definite(scale * cov)

    draws, log_p, acceptance = regression_sampler(theta_map, proposal_cov, logp, spec, settings, group_id=group_id)
    return RegressionShardResult(draws, log_p, acceptance, theta_map, proposal_cov, group_id=group_id)


def fit_regression_frequency_divide_and_conquer(
    x: np.ndarray,
    y: np.ndarray,
    spec: RegressionSpec,
    settings: RegressionSettings,
    n_groups: int = 16,
    include_full: bool = True,
) -> RegressionDistributedResult:
    x_hat, y_hat = regression_frequency_domain(x, y)
    n_freq = int(onp.floor((len(y) - 1) / 2))
    all_indices = onp.arange(n_freq)
    groups = frequency_partition_indices(n_freq, n_groups, method="systematic")
    shards = [
        fit_regression_whittle_shard(y_hat, x_hat, indices, spec, settings, n_groups=n_groups, group_id=gid)
        for gid, indices in enumerate(groups)
    ]
    full = None
    if include_full:
        full = fit_regression_whittle_shard(y_hat, x_hat, all_indices, spec, settings, n_groups=1, group_id=0)
    shard_draws = [item.draws for item in shards]
    return RegressionDistributedResult(
        full=full,
        shards=shards,
        average_draws=simple_average(shard_draws),
        consensus_draws=consensus(shard_draws),
        n_groups=n_groups,
    )


def time_domain_ar2_log_likelihood(theta, x, y):
    """Exact conditional Gaussian likelihood for regression with AR(2) errors."""
    beta = theta[0]
    phi = reparam(theta[1:3], MA=False)
    sigma2 = np.exp(theta[-1])
    residual = y - beta * np.asarray(x).reshape(-1)
    innovations = residual[2:] - phi[0] * residual[1:-1] - phi[1] * residual[:-2]
    return -0.5 * len(innovations) * np.log(2.0 * np.pi * sigma2) - 0.5 * np.sum(innovations**2 / sigma2)
