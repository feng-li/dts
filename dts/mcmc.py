"""Core likelihoods and MCMC routines for distributed time-series inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

from dts._jax import configure_jax

configure_jax()

from jax import jit, lax
import jax.numpy as np
import jax.scipy.stats as sps_jax
import numpy as onp
import statsmodels.api as sm

from dts.progress import progress_bar


@dataclass(frozen=True)
class ModelSpec:
    """ARMA/ARTFIMA model specification.

    ``ar_order`` and ``ma_order`` follow the conventional ARMA(p, q)
    terminology: ``ar_order`` is the autoregressive order and ``ma_order`` is
    the moving-average order.

    Parameters are represented as
    ``[ar_partial..., ma_partial..., log_lambda, log_sigma2, d]`` for ARTFIMA
    and ``[ar_partial..., ma_partial..., log_sigma2]`` for ARMA.  With
    ``mean_term`` the location parameter ``mu`` is appended last, giving
    ``[..., d, mu]``; it enters only through the marginal mean density, since
    the Whittle likelihood is evaluated at positive frequencies only.
    """

    ar_order: int
    ma_order: int
    tfi_term: bool = False
    exact: bool = False
    mean_term: bool = False

    @property
    def n_params(self) -> int:
        return self.ar_order + self.ma_order + (3 if self.tfi_term else 1) + int(self.mean_term)

    @property
    def last_arma(self) -> int:
        """Size of the trailing transformed-scale block (log lambda, log sigma2, d)."""
        return 3 if self.tfi_term else 1

    @property
    def n_guard(self) -> int:
        """Trailing elements the ``|.| < 1`` process-parameter check must skip."""
        return self.last_arma + (1 if self.mean_term else 0)


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


def whittle_log_likelihood(params, ar_order, ma_order, I_pg, TFI_term, omega_shard,
                           mean_term=False):
    """Positive-frequency contributions; an optional trailing mean is excluded."""
    phi, theta, var, d, lambda_, _ = split_parameters(
        params, ar_order, ma_order, TFI_term, mean_term=mean_term
    )
    density = f_ARTFIMA(omega_shard, phi, theta, var, d, lambda_)
    return -(np.log(density) + I_pg / density)


def split_parameters(params, ar_order, ma_order, TFI_term, mean_term=False):
    """Unpack ``[ar_partial..., ma_partial..., log_lambda, log_sigma2, d, mu]``.

    ``mu`` is returned as 0.0 when ``mean_term`` is false, so callers that do
    not carry a location parameter keep working unchanged.
    """
    if TFI_term:
        d = params[-2] if mean_term else params[-1]
        log_lambda = params[-4] if mean_term else params[-3]
        var = np.exp(params[-3] if mean_term else params[-2])
        lambda_ = np.exp(log_lambda)
    else:
        d = 0.0
        lambda_ = 0.0
        var = np.exp(params[-2] if mean_term else params[-1])

    phi = reparam(params[:ar_order], MA=False) if ar_order > 0 else np.array([])
    theta = (
        reparam(params[ar_order : ar_order + ma_order], MA=True)
        if ma_order > 0
        else np.array([])
    )
    mu = params[-1] if mean_term else 0.0
    return phi, theta, var, d, lambda_, mu


def n_mean_grid(series_length: int, points_per_observation: int = 8) -> int:
    """Quadrature size for the marginal-mean variance; power of two, >= 1024."""
    target = max(1024, points_per_observation * int(series_length))
    return int(2 ** onp.ceil(onp.log2(target)))


def mean_variance(params, ar_order, ma_order, TFI_term, series_length, n_grid=None,
                  mean_term=False):
    """Variance of the sample mean under the spectral model.

    ``v_T = T^-2 * int f(w) F_T(w) dw`` with the Fejer kernel
    ``F_T(w) = (sin(T w / 2) / sin(w / 2))^2``.  The integral is evaluated by
    midpoint quadrature on ``(0, pi)``; the integrand is a trigonometric
    polynomial of degree ``T`` times a smooth density, and the grid defaults to
    eight points per observation, which reproduces closed-form AR(1) and white
    noise variances to machine precision.
    """
    phi, theta, var, d, lambda_, _ = split_parameters(
        params, ar_order, ma_order, TFI_term, mean_term=mean_term
    )
    T = series_length
    if n_grid is None:
        n_grid = n_mean_grid(T)
    grid = np.pi * (np.arange(n_grid) + 0.5) / n_grid
    fejer = np.sin(T * grid / 2.0) / np.sin(grid / 2.0)
    density = f_ARTFIMA(grid, phi, theta, var, d, lambda_)
    integral = 2.0 * np.sum(density * fejer * fejer) * (np.pi / n_grid)
    return integral / (T * T)


def log_mean_density(params, ybar, series_length, ar_order, ma_order, TFI_term,
                     n_groups: int = 1, n_grid=None, mean_term=True):
    """Marginal log density of the realized sample mean, ``log N(ybar; mu, v_T)``.

    Added to the positive-frequency Whittle likelihood when a location
    parameter is carried; divided by ``n_groups`` so that the product of the
    shard targets equals the declared mean-augmented target.
    """
    _, _, _, _, _, mu = split_parameters(
        params, ar_order, ma_order, TFI_term, mean_term=mean_term
    )
    if not mean_term:
        mu = 0.0
    variance = mean_variance(
        params, ar_order, ma_order, TFI_term, series_length, n_grid=n_grid,
        mean_term=mean_term,
    )
    log_density = -0.5 * (np.log(2.0 * np.pi * variance) + (ybar - mu) ** 2 / variance)
    return log_density / n_groups


@partial(jit, static_argnames=("ar_order", "ma_order", "TFI_term", "last_arma",
                               "mean_term", "mean_series_length"))
def whittle_log_posterior_jit(
    theta,
    ar_order: int,
    ma_order: int,
    I_pg,
    TFI_term: bool,
    omega_shard,
    n_groups: int,
    params_prior_mu: float,
    params_prior_sd: float,
    last_arma: int,
    mean_term: bool = False,
    mean_ybar: float | None = None,
    mean_series_length: int | None = None,
    mean_prior_mu: float = 0.0,
    mean_prior_sd: float = 1.0,
):
    if mean_term and (mean_ybar is None or mean_series_length is None):
        raise ValueError("mean_term=True requires the sample mean and series length")
    if mean_term and (mean_series_length < 1 or int(mean_series_length) != mean_series_length):
        raise ValueError("mean series length must be a positive integer")
    log_likelihood = np.sum(whittle_log_likelihood(
        theta,
        ar_order,
        ma_order,
        I_pg,
        TFI_term,
        omega_shard,
        mean_term=mean_term,
    ))
    if mean_term:
        log_likelihood = log_likelihood + log_mean_density(
            theta,
            mean_ybar,
            int(mean_series_length),
            ar_order,
            ma_order,
            TFI_term,
            n_groups=n_groups,
            mean_term=True,
        )
    prior = log_prior(
        theta,
        params_prior_mu,
        params_prior_sd,
        last_arma,
        TFI_term,
        n_groups=n_groups,
        check_bounds=True,
        mean_term=mean_term,
        mean_prior_mu=mean_prior_mu,
        mean_prior_sd=mean_prior_sd,
    )
    return log_likelihood + prior


def exact_log_likelihood_arma(data, params, ar_order, ma_order, mean_term=False):
    """Exact Gaussian ARMA log-likelihood using statsmodels innovations.

    With ``mean_term`` the location parameter is the last element and the
    series is demeaned before the zero-mean innovations likelihood is
    evaluated; the joint Gaussian likelihood already carries the sample-mean
    information, so no marginal mean density is added here.
    """
    if hasattr(params, "_value"):
        params = params._value
    params_np = onp.array(params, dtype=float, copy=True)

    # reparam returns immutable jax buffers; statsmodels needs writable arrays
    phi = (
        onp.array(reparam(params_np[:ar_order], MA=False), dtype=float, copy=True)
        if ar_order > 0
        else onp.array([])
    )
    theta = (
        onp.array(reparam(params_np[ar_order : ar_order + ma_order], MA=True), dtype=float, copy=True)
        if ma_order > 0
        else onp.array([])
    )
    var = float(onp.exp(params_np[-2] if mean_term else params_np[-1]))
    values = onp.array(data, dtype=float, copy=True)
    if mean_term:
        values = values - float(params_np[-1])
    return sm.tsa.innovations.arma_loglike(values, phi, theta, sigma2=var)


def _valid_process_params(params, last_arma: int, mean_term: bool = False) -> bool:
    guard = last_arma + (1 if mean_term else 0)
    process_params = onp.asarray(params[:-guard], dtype=float)
    return bool(onp.all(onp.abs(process_params) < 1.0))


def log_prior(
    params,
    mu: float,
    sd: float,
    Last_ARMA: int,
    TFI_term: bool,
    n_groups: int = 1,
    check_bounds: bool = True,
    mean_term: bool = False,
    mean_prior_mu: float = 0.0,
    mean_prior_sd: float = 1.0,
):
    """Fractionated prior used by subposteriors."""
    offset = 1 if mean_term else 0
    n_process = len(params) - Last_ARMA - offset
    prior_process = -n_process * np.log(2.0)

    tail = params[-(Last_ARMA + offset): len(params) - offset] if offset else params[-Last_ARMA:]
    if TFI_term:
        prior_tail = (
            sps_jax.norm.logpdf(tail[0], loc=mu, scale=sd)
            + sps_jax.norm.logpdf(tail[1], loc=mu, scale=sd)
            + sps_jax.norm.logpdf(tail[2], loc=mu, scale=sd)
        )
    else:
        prior_tail = sps_jax.norm.logpdf(tail[0], loc=mu, scale=sd)

    prior = prior_process + prior_tail
    if mean_term:
        prior = prior + sps_jax.norm.logpdf(params[-1], loc=mean_prior_mu, scale=mean_prior_sd)
    prior = prior / n_groups
    if check_bounds:
        valid = np.all(np.abs(params[:-Last_ARMA - offset] if offset else params[:-Last_ARMA]) < 1.0)
        prior = np.where(valid, prior, -np.inf)
    return prior


def make_positive_definite(matrix, min_eigenvalue: float = 1e-8):
    """Symmetrize a covariance matrix and clip small eigenvalues."""
    mat = onp.asarray(matrix, dtype=float)
    mat = (mat + mat.T) / 2.0
    values, vectors = onp.linalg.eigh(mat)
    values = onp.clip(values, min_eigenvalue, None)
    return (vectors * values) @ vectors.T


def parameter_bounds(n_params: int, tfi_term: bool, mean_term: bool = False) -> Tuple[onp.ndarray, onp.ndarray]:
    """Bounds for partial autocorrelations and transformed scale parameters."""
    process_limit = 1.0 - 1e-8
    lower = onp.full(n_params, -process_limit)
    upper = onp.full(n_params, process_limit)
    offset = 1 if mean_term else 0
    tail = slice(n_params - (3 if tfi_term else 1) - offset, n_params - offset)
    lower[tail] = -30.0
    upper[tail] = 30.0
    if mean_term:
        lower[-1] = -30.0
        upper[-1] = 30.0
    return lower, upper


def validate_mean_data(mean_data):
    """Validate observed mean metadata before optimization or sampling."""
    if mean_data is None:
        raise ValueError("mean_term=True requires mean_data=(sample_mean, series_length)")
    try:
        ybar, length = mean_data
        ybar = float(ybar)
        valid_length = float(length)
        length = int(length)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("mean_data must contain a finite mean and a positive integer length") from exc
    if not onp.isfinite(ybar) or length < 1 or valid_length != length:
        raise ValueError("mean_data must contain a finite mean and a positive integer length")
    return ybar, length


@partial(jit, static_argnames=('ar_order', 'ma_order'))
def _frequency_basis(omega, ar_order, ma_order):
    """Parameter-independent real Fourier quantities for repeated evaluations."""
    omega = np.asarray(omega, dtype=np.float64)
    q = 4.0 * np.sin(omega / 2.0) ** 2
    qc = 4.0 * np.cos(omega / 2.0) ** 2
    ar_phase = np.outer(omega, np.arange(1, ar_order + 1)) if ar_order > 1 else np.empty((len(omega), 0))
    ma_phase = np.outer(omega, np.arange(1, ma_order + 1)) if ma_order > 1 else np.empty((len(omega), 0))
    return q, qc, np.cos(ar_phase), np.sin(ar_phase), np.cos(ma_phase), np.sin(ma_phase)


@partial(jit, static_argnames=('ar_order', 'ma_order', 'TFI_term', 'last_arma',
                               'mean_term', 'mean_series_length'))
def _cached_whittle_log_posterior(
    params, ar_order, ma_order, I_pg, basis, TFI_term, n_groups,
    params_prior_mu, params_prior_sd, last_arma, mean_term=False,
    mean_ybar=None, mean_series_length=None, mean_prior_mu=0.0, mean_prior_sd=1.0,
):
    """The native posterior evaluated through its algebraic log spectrum.

    No transformed-parameter cutoffs are added. The first-order identities
    use sin/cos half-angle forms to avoid cancellation at either spectral end.
    Higher-order models use the same partial-coefficient transformation.
    """
    phi, theta, _, d, lam, _ = split_parameters(
        params, ar_order, ma_order, TFI_term, mean_term=mean_term
    )
    q, qc, ar_cos, ar_sin, ma_cos, ma_sin = basis
    scale_index = -(2 + int(mean_term)) if TFI_term else -(1 + int(mean_term))
    logf = np.full_like(q, params[scale_index] - np.log(2.0 * np.pi))
    if ar_order == 1:
        ar_squared = np.where(phi[0] >= 0.0,
                              (1.0 - phi[0]) ** 2 + phi[0] * q,
                              (1.0 + phi[0]) ** 2 - phi[0] * qc)
        logf = logf - np.log(ar_squared)
    elif ar_order:
        logf = logf - np.log((1.0 - ar_cos @ phi) ** 2 + (ar_sin @ phi) ** 2)
    if ma_order == 1:
        ma_squared = np.where(theta[0] >= 0.0,
                              (1.0 - theta[0]) ** 2 + theta[0] * qc,
                              (1.0 + theta[0]) ** 2 - theta[0] * q)
        logf = logf + np.log(ma_squared)
    elif ma_order:
        logf = logf + np.log((1.0 + ma_cos @ theta) ** 2 + (ma_sin @ theta) ** 2)
    if TFI_term:
        logf = logf - d * np.log((-np.expm1(-lam)) ** 2 + np.exp(-lam) * q)
    value = -np.sum(logf + I_pg * np.exp(-logf))
    if mean_term:
        value = value + log_mean_density(
            params, mean_ybar, int(mean_series_length), ar_order, ma_order,
            TFI_term, n_groups=n_groups, mean_term=True,
        )
    return value + log_prior(
        params, params_prior_mu, params_prior_sd, last_arma, TFI_term,
        n_groups=n_groups, check_bounds=True, mean_term=mean_term,
        mean_prior_mu=mean_prior_mu, mean_prior_sd=mean_prior_sd,
    )


@partial(jit, static_argnames=('ar_order', 'ma_order', 'TFI_term', 'last_arma',
                               'mean_term', 'mean_series_length'))
def _compiled_whittle_steps(
    current, current_logp, increments, log_uniforms, ar_order, ma_order, I_pg,
    basis, TFI_term, n_groups, params_prior_mu, params_prior_sd, last_arma,
    mean_term=False, mean_ybar=None, mean_series_length=None,
    mean_prior_mu=0.0, mean_prior_sd=1.0,
):
    """Compile the complete Metropolis transition loop, with supplied randomness."""
    n_process = len(current) - last_arma - int(mean_term)

    def posterior(point):
        return _cached_whittle_log_posterior(
            point, ar_order, ma_order, I_pg, basis, TFI_term, n_groups,
            params_prior_mu, params_prior_sd, last_arma, mean_term,
            mean_ybar, mean_series_length, mean_prior_mu, mean_prior_sd,
        )

    def transition(state, random_inputs):
        point, value = state
        increment, log_uniform = random_inputs
        proposal = point + increment
        candidate = lax.cond(
            np.all(np.abs(proposal[:n_process]) < 1.0), posterior,
            lambda unused: np.asarray(-np.inf), proposal,
        )
        accept = log_uniform < np.minimum(0.0, candidate - value)
        point = np.where(accept, proposal, point)
        value = np.where(accept, candidate, value)
        return (point, value), (point, value, accept)

    return lax.scan(transition, (current, current_logp), (increments, log_uniforms))


def _sample_compiled_whittle(
    current, covariance, rng, ar_order, ma_order, I_pg, omega, TFI_term,
    n_samples, burn, n_groups, params_prior_mu, params_prior_sd, last_arma,
    progress, description, mean_term, mean_ybar, mean_series_length,
    mean_prior_mu, mean_prior_sd,
):
    """Use NumPy's existing RNG stream with the compiled posterior/transition."""
    basis = _frequency_basis(omega, ar_order, ma_order)
    arguments = (ar_order, ma_order, I_pg, basis, TFI_term, n_groups,
                 params_prior_mu, params_prior_sd, last_arma, mean_term,
                 mean_ybar, mean_series_length, mean_prior_mu, mean_prior_sd)
    value = _cached_whittle_log_posterior(current, *arguments)
    samples = onp.zeros((n_samples, len(current)))
    logp = onp.zeros(n_samples)
    accepted = onp.zeros(n_samples, dtype=bool)
    zero = onp.zeros(len(current))
    # Chunked execution keeps progress visible; random draws retain the same
    # normal/uniform interleaving as the vanilla loop, including rejected moves.
    chunk_size = 1024 if progress else max(n_samples, 1)
    with progress_bar(total=n_samples, desc=description or 'MCMC samples',
                      unit='sample', leave=False, disable=not progress) as bar:
        for start in range(0, n_samples, chunk_size):
            stop = min(n_samples, start + chunk_size)
            increments = onp.empty((stop-start, len(current)))
            uniforms = onp.empty(stop-start)
            for index in range(stop-start):
                increments[index] = rng.multivariate_normal(zero, covariance)
                uniforms[index] = onp.log(rng.random())
            (current, value), result = _compiled_whittle_steps(
                current, value, increments, uniforms, *arguments,
            )
            samples[start:stop] = onp.asarray(result[0])
            logp[start:stop] = onp.asarray(result[1])
            accepted[start:stop] = onp.asarray(result[2])
            bar.update(stop-start)
    return samples[burn:], logp[burn:], accepted[burn:]


def sampler(
    ar_order,
    ma_order,
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
    progress: bool = False,
    progress_desc: str | None = None,
    mean_term: bool = False,
    mean_data: Optional[Tuple[float, int]] = None,
    mean_prior_mu: float = 0.0,
    mean_prior_sd: float = 1.0,
    backend: str = 'compiled',
):
    """Random-walk Metropolis sampler.

    This keeps the original project function name and argument order, with
    optional ``n_groups`` and ``random_state`` arguments for reproducibility.
    ``mean_data=(ybar, T)`` switches on the mean-augmented target: the location
    parameter ``mu`` is the last element of ``paramsStar`` and the marginal mean
    density of the realized sample mean is divided by ``n_groups`` on each
    shard.

    ``backend='compiled'`` caches Fourier quantities and compiles the full
    Whittle Metropolis loop in JAX. ``backend='vanilla'`` retains the original
    Python loop and likelihood for numerical checks. Both use the same NumPy
    RNG stream and proposal-covariance handling; likelihood rounding can differ.
    Exact time-domain inference uses the Python loop with either backend.
    """
    if backend not in {'compiled', 'vanilla'}:
        raise ValueError("backend must be 'compiled' or 'vanilla'")
    if mean_data is not None:
        mean_term = True
    ybar, mean_series_length = (
        validate_mean_data(mean_data) if mean_term and not exact else (None, None)
    )
    rng = onp.random.default_rng(random_state)
    params_current = onp.asarray(paramsStar, dtype=float)
    proposal_cov = make_positive_definite(proposal_width)
    n_params = len(params_current)
    last_arma = 3 if TFI_term else 1
    I_pg_jax = None if exact else np.asarray(I_pg)
    omega_shard_jax = None if exact else np.asarray(omega_shard)

    if not exact and backend == 'compiled':
        return _sample_compiled_whittle(
            params_current, proposal_cov, rng, ar_order, ma_order, I_pg_jax,
            omega_shard_jax, TFI_term, n_samples, Burn_in, n_groups,
            params_prior_mu, params_prior_sd, last_arma, progress, progress_desc,
            mean_term, ybar, mean_series_length, mean_prior_mu, mean_prior_sd,
        )

    posterior_samples = onp.zeros((n_samples, n_params))
    log_p = onp.zeros(n_samples)
    acceptance = onp.zeros(n_samples, dtype=bool)

    def log_posterior(theta):
        if exact:
            log_likelihood = exact_log_likelihood_arma(
                data,
                theta,
                ar_order,
                ma_order,
                mean_term=mean_term,
            )
            prior = log_prior(
                theta,
                params_prior_mu,
                params_prior_sd,
                last_arma,
                TFI_term,
                n_groups=n_groups,
                check_bounds=True,
                mean_term=mean_term,
                mean_prior_mu=mean_prior_mu,
                mean_prior_sd=mean_prior_sd,
            )
            return float(onp.sum(log_likelihood) + prior)
        return float(
            whittle_log_posterior_jit(
                theta,
                ar_order,
                ma_order,
                I_pg_jax,
                TFI_term,
                omega_shard_jax,
                n_groups,
                params_prior_mu,
                params_prior_sd,
                last_arma,
                mean_term,
                ybar,
                mean_series_length,
                mean_prior_mu,
                mean_prior_sd,
            )
        )

    log_p_current = log_posterior(params_current)

    sample_iter = progress_bar(
        range(n_samples),
        desc=progress_desc or "MCMC samples",
        unit="sample",
        leave=False,
        disable=not progress,
    )
    for i in sample_iter:
        params_proposal = rng.multivariate_normal(params_current, proposal_cov)
        if _valid_process_params(params_proposal, last_arma, mean_term=mean_term):
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
    ar_order,
    ma_order,
    data,
    n_samples,
    paramsStar,
    proposal_width,
    Burn_in,
    params_prior_mu=0,
    params_prior_sd=1.0,
    random_state: Optional[int] = None,
    progress: bool = False,
    progress_desc: str | None = None,
    mean_term: bool = False,
    mean_prior_mu: float = 0.0,
    mean_prior_sd: float = 1.0,
):
    return sampler(
        ar_order=ar_order,
        ma_order=ma_order,
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
        progress=progress,
        progress_desc=progress_desc,
        mean_term=mean_term,
        mean_data=(float(onp.mean(onp.asarray(data, dtype=float))), len(data)) if mean_term else None,
        mean_prior_mu=mean_prior_mu,
        mean_prior_sd=mean_prior_sd,
    )


def sampler_whittle(
    ar_order,
    ma_order,
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
    progress: bool = False,
    progress_desc: str | None = None,
    mean_term: bool = False,
    mean_data: Optional[Tuple[float, int]] = None,
    mean_prior_mu: float = 0.0,
    mean_prior_sd: float = 1.0,
):
    return sampler(
        ar_order=ar_order,
        ma_order=ma_order,
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
        progress=progress,
        progress_desc=progress_desc,
        mean_term=mean_term,
        mean_data=mean_data,
        mean_prior_mu=mean_prior_mu,
        mean_prior_sd=mean_prior_sd,
    )
