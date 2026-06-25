"""Core likelihoods and MCMC routines for distributed time-series inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import autograd.numpy as np
import autograd.scipy.stats as sps_autograd
import numpy as onp
import statsmodels.api as sm


@dataclass(frozen=True)
class ModelSpec:
    """ARMA/ARTFIMA model specification.

    Parameters are represented as
    ``[ar_partial..., ma_partial..., log_lambda, log_sigma2, d]`` for ARTFIMA
    and ``[ar_partial..., ma_partial..., log_sigma2]`` for ARMA.
    """

    q: int
    p: int
    tfi_term: bool = False
    exact: bool = False

    @property
    def n_params(self) -> int:
        return self.q + self.p + (3 if self.tfi_term else 1)

    @property
    def last_arma(self) -> int:
        return 3 if self.tfi_term else 1


def reparam(params, MA: bool = False):
    """Convert partial autocorrelation parameters to AR or MA coefficients."""
    out = np.array(params)
    sign = 1.0 if MA else -1.0
    for j in range(1, len(params)):
        updated = out[:j] + sign * out[j] * out[j - 1 :: -1]
        out = np.concatenate([updated, out[j:]])
    return out


def f_ARTFIMA(omega, phi, theta, var, d, lambda_):
    """Spectral density for ARTFIMA and its ARMA special case."""
    omega = np.asarray(omega)
    phi = np.asarray(phi)
    theta = np.asarray(theta)

    tfi = np.abs(1.0 - np.exp(-(lambda_ + 1j * omega))) ** (-2.0 * d)

    if len(phi):
        ar_basis = np.exp(-1j * np.outer(omega, np.arange(1, len(phi) + 1)))
        ar_part = 1.0 / (1.0 - np.dot(ar_basis, phi))
    else:
        ar_part = 1.0

    if len(theta):
        ma_basis = np.exp(-1j * np.outer(omega, np.arange(1, len(theta) + 1)))
        ma_part = 1.0 + np.dot(ma_basis, theta)
    else:
        ma_part = 1.0

    return (
        tfi
        * (var / (2.0 * np.pi))
        * (np.real(ar_part) ** 2 + np.imag(ar_part) ** 2)
        * (np.real(ma_part) ** 2 + np.imag(ma_part) ** 2)
    )


def whittle_log_likelihood(params, q, p, I_pg, TFI_term, omega_shard):
    """Whittle log-likelihood contributions for positive Fourier frequencies."""
    params = np.asarray(params)
    if TFI_term:
        d = params[-1]
        lambda_ = np.exp(params[-3])
        var = np.exp(params[-2])
    else:
        d = 0.0
        lambda_ = 0.0
        var = np.exp(params[-1])

    phi = reparam(params[:q], MA=False) if q > 0 else np.array([])
    theta = reparam(params[q : q + p], MA=True) if p > 0 else np.array([])
    density = f_ARTFIMA(omega_shard, phi, theta, var, d, lambda_)
    return -(np.log(density) + I_pg / density)


def exact_log_likelihood_arma(data, params, q, p):
    """Exact Gaussian ARMA log-likelihood using statsmodels innovations."""
    if hasattr(params, "_value"):
        params = params._value
    params_np = onp.asarray(params, dtype=float)

    phi = onp.asarray(reparam(params_np[:q], MA=False), dtype=float) if q > 0 else onp.array([])
    theta = onp.asarray(reparam(params_np[q : q + p], MA=True), dtype=float) if p > 0 else onp.array([])
    var = float(onp.exp(params_np[-1]))
    return sm.tsa.innovations.arma_loglike(onp.asarray(data, dtype=float), phi, theta, sigma2=var)


def _valid_process_params(params, last_arma: int) -> bool:
    process_params = onp.asarray(params[:-last_arma], dtype=float)
    return bool(onp.all(onp.abs(process_params) < 1.0))


def log_prior(
    params,
    mu: float,
    sd: float,
    Last_ARMA: int,
    TFI_term: bool,
    n_groups: int = 1,
    check_bounds: bool = True,
):
    """Fractionated prior used by subposteriors."""
    if check_bounds and not _valid_process_params(params, Last_ARMA):
        return -np.inf

    n_process = len(params) - Last_ARMA
    prior_process = -n_process * np.log(2.0)

    if TFI_term:
        prior_tail = (
            sps_autograd.norm.logpdf(params[-1], loc=mu, scale=sd)
            + sps_autograd.norm.logpdf(params[-3], loc=mu, scale=sd)
            + sps_autograd.norm.logpdf(params[-2], loc=mu, scale=sd)
        )
    else:
        prior_tail = sps_autograd.norm.logpdf(params[-1], loc=mu, scale=sd)

    return (prior_process + prior_tail) / n_groups


def make_positive_definite(matrix, min_eigenvalue: float = 1e-8):
    """Symmetrize a covariance matrix and clip small eigenvalues."""
    mat = onp.asarray(matrix, dtype=float)
    mat = (mat + mat.T) / 2.0
    values, vectors = onp.linalg.eigh(mat)
    values = onp.clip(values, min_eigenvalue, None)
    return (vectors * values) @ vectors.T


def parameter_bounds(n_params: int, tfi_term: bool) -> Tuple[onp.ndarray, onp.ndarray]:
    """Bounds for partial autocorrelations and transformed scale parameters."""
    lower = -onp.ones(n_params)
    upper = onp.ones(n_params)
    if tfi_term:
        lower[-3:] = -30.0
        upper[-3:] = 30.0
    else:
        lower[-1:] = -30.0
        upper[-1:] = 30.0
    return lower, upper


def sampler(
    q,
    p,
    data,
    I_pg,
    TFI_term,
    omega_shard,
    n_samples,
    paramsStar,
    proposal_width,
    Burn_in,
    params_prior_mu=0,
    params_prior_sd=1.0,
    exact=False,
    n_groups: int = 1,
    random_state: Optional[int] = None,
):
    """Random-walk Metropolis sampler.

    This keeps the original project function name and argument order, with
    optional ``n_groups`` and ``random_state`` arguments for reproducibility.
    """
    rng = onp.random.default_rng(random_state)
    params_current = onp.asarray(paramsStar, dtype=float)
    proposal_cov = make_positive_definite(proposal_width)
    n_params = len(params_current)
    last_arma = 3 if TFI_term else 1

    posterior_samples = onp.zeros((n_samples, n_params))
    log_p = onp.zeros(n_samples)
    acceptance = onp.zeros(n_samples, dtype=bool)

    def log_posterior(theta):
        if exact:
            log_likelihood = exact_log_likelihood_arma(data, theta, q, p)
        else:
            log_likelihood = whittle_log_likelihood(theta, q, p, I_pg, TFI_term, omega_shard)
        prior = log_prior(
            theta,
            params_prior_mu,
            params_prior_sd,
            last_arma,
            TFI_term,
            n_groups=n_groups,
            check_bounds=False,
        )
        return float(onp.sum(log_likelihood) + prior)

    log_p_current = log_posterior(params_current)

    for i in range(n_samples):
        params_proposal = rng.multivariate_normal(params_current, proposal_cov)
        if _valid_process_params(params_proposal, last_arma):
            log_p_proposal = log_posterior(params_proposal)
        else:
            log_p_proposal = -onp.inf

        log_alpha = min(0.0, log_p_proposal - log_p_current)
        if onp.log(rng.random()) < log_alpha:
            params_current = params_proposal
            log_p_current = log_p_proposal
            acceptance[i] = True

        posterior_samples[i, :] = params_current
        log_p[i] = log_p_current

    return posterior_samples[Burn_in:], log_p[Burn_in:], acceptance[Burn_in:]


def sampler_exact(
    q,
    p,
    data,
    n_samples,
    paramsStar,
    proposal_width,
    Burn_in,
    params_prior_mu=0,
    params_prior_sd=1.0,
    random_state: Optional[int] = None,
):
    return sampler(
        q=q,
        p=p,
        data=data,
        I_pg=None,
        TFI_term=False,
        omega_shard=None,
        n_samples=n_samples,
        paramsStar=paramsStar,
        proposal_width=proposal_width,
        Burn_in=Burn_in,
        params_prior_mu=params_prior_mu,
        params_prior_sd=params_prior_sd,
        exact=True,
        random_state=random_state,
    )


def sampler_whittle(
    q,
    p,
    I_pg,
    TFI_term,
    omega_shard,
    n_samples,
    paramsStar,
    proposal_width,
    Burn_in,
    params_prior_mu=0,
    params_prior_sd=1.0,
    n_groups: int = 1,
    random_state: Optional[int] = None,
):
    return sampler(
        q=q,
        p=p,
        data=None,
        I_pg=I_pg,
        TFI_term=TFI_term,
        omega_shard=omega_shard,
        n_samples=n_samples,
        paramsStar=paramsStar,
        proposal_width=proposal_width,
        Burn_in=Burn_in,
        params_prior_mu=params_prior_mu,
        params_prior_sd=params_prior_sd,
        exact=False,
        n_groups=n_groups,
        random_state=random_state,
    )
