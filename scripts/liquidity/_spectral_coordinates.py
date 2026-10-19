"""Exact-target coordinates for the boundary-sensitive liquidity ARTFIMA fit.

Sampling and consensus use (atanh(MA), sqrt(lambda), log(sigma2), d).
Every shard receives 1/G of the induced prior's Jacobian. Their product
therefore remains the full-data Whittle posterior, expressed in these
coordinates. Samples are mapped back before forecasting and validation.
"""
from functools import partial

import numpy as np

from dts._jax import configure_jax

configure_jax()
import jax.numpy as jnp
from jax import jit, lax

from dts.mcmc import _cached_whittle_log_posterior, _frequency_basis


def to_coordinates(native, lambda_power=.5):
    values = np.array(native, dtype=float, copy=True)
    if not np.isfinite(values).all() or np.any(np.abs(values[..., 0]) >= 1):
        raise ValueError('Finite parameters and an MA coefficient inside (-1,1) required')
    values[..., 0] = np.arctanh(values[..., 0])
    if lambda_power:
        values[..., 1] = np.exp(lambda_power * values[..., 1])
    return values


def from_coordinates(coordinates, lambda_power=.5):
    values = np.array(coordinates, dtype=float, copy=True)
    if lambda_power:
        if np.any(values[..., 1] <= 0):
            raise ValueError('The positive tempering coordinate must exceed zero')
        values[..., 1] = np.log(values[..., 1]) / lambda_power
    values[..., 0] = np.tanh(values[..., 0])
    return values


def log_jacobian(coordinates, lambda_power=.5):
    values = np.asarray(coordinates)
    absolute = np.abs(values[..., 0])
    result = 2 * (np.log(2.) - absolute - np.log1p(np.exp(-2 * absolute)))
    if lambda_power:
        result = result - np.log(abs(lambda_power)) - np.log(values[..., 1])
    return result


def coordinate_objective(point, q, pg, groups, lambda_power=.5):
    """Unchanged negative Whittle subposterior, with its exact induced prior."""
    from scripts.liquidity.probe_csi300_fractional_changes import target, spectrum_hessian
    point = np.asarray(point)
    if not np.isfinite(point).all() or (lambda_power and point[1] <= 0):
        return np.inf, np.zeros(4), np.zeros((4, 4))
    native = from_coordinates(point, lambda_power)
    value, gradient = target(native, q, pg, groups)
    curvature = spectrum_hessian(native, q, pg, groups)
    ma = native[0]
    j0 = (1 - ma) * (1 + ma)
    scale = np.array([j0, 1., 1., 1.])
    second = np.array([-2*ma*j0, 0., 0., 0.])
    log_gradient = np.array([-2*ma, 0., 0., 0.])
    log_hessian = np.array([-2*j0, 0., 0., 0.])
    if lambda_power:
        scale[1] = 1 / (lambda_power * point[1])
        second[1] = -1 / (lambda_power * point[1]**2)
        log_gradient[1] = -1 / point[1]
        log_hessian[1] = 1 / point[1]**2
    transformed_hessian = curvature * scale[:, None] * scale[None, :]
    transformed_hessian += np.diag(gradient * second - log_hessian/groups)
    transformed_gradient = gradient * scale - log_gradient/groups
    return (float(value - log_jacobian(point, lambda_power)/groups),
            transformed_gradient, transformed_hessian)


def coordinate_map(native_map, q, pg, groups, lambda_power=.5):
    """Polish the resolved native mode for the induced fractional prior."""
    point = to_coordinates(native_map, lambda_power)
    for iteration in range(21):
        value, gradient, curvature = coordinate_objective(point, q, pg, groups, lambda_power)
        eigenvalues = np.linalg.eigvalsh(curvature)
        if not np.isfinite(value) or not np.isfinite(curvature).all() or eigenvalues.min() <= 0:
            raise ValueError('Unresolved coordinate MAP curvature')
        covariance = np.linalg.inv(curvature)
        distance = float(np.sqrt(max(0., gradient @ covariance @ gradient)))
        if distance <= .001:
            return dict(theta=point.tolist(), native_theta=from_coordinates(point, lambda_power).tolist(),
                        covariance=covariance.tolist(), newton_distance=distance,
                        negative_log_posterior=value, iterations=iteration)
        step = covariance @ gradient
        for fraction in 2. ** -np.arange(25):
            candidate = point - fraction * step
            proposal_value = coordinate_objective(candidate, q, pg, groups, lambda_power)[0]
            if np.isfinite(proposal_value) and proposal_value <= value + 1e-8:
                point = candidate
                break
        else:
            raise ValueError('Coordinate MAP line search did not resolve')
    raise ValueError('Coordinate MAP stationarity did not resolve')


def coordinate_log_posterior(point, pg, basis, groups, lambda_power=.5):
    native = point.at[0].set(jnp.tanh(point[0]))
    absolute = jnp.abs(point[0])
    logj = 2 * (jnp.log(2.) - absolute - jnp.log1p(jnp.exp(-2 * absolute)))
    valid = jnp.isfinite(point).all()
    if lambda_power:
        valid = valid & (point[1] > 0)
        positive = jnp.where(point[1] > 0, point[1], 1.)
        native = native.at[1].set(jnp.log(positive)/lambda_power)
        logj = logj - jnp.log(abs(lambda_power)) - jnp.log(positive)
    value = _cached_whittle_log_posterior(
        native, 0, 1, pg, basis, True, groups, 0., 1., 3, False)
    return jnp.where(valid, value + logj/groups, -jnp.inf)


@partial(jit, static_argnames=('lambda_power',))
def _coordinate_steps(current, value, increments, log_uniforms, independent, center,
                      independence_precision, pg, basis, groups, lambda_power=.5):
    def transition(state, random_inputs):
        point, logp = state
        increment, log_uniform, jump = random_inputs
        proposal = jnp.where(jump, center + increment, point + increment)
        candidate = coordinate_log_posterior(proposal, pg, basis, groups, lambda_power)
        old_delta, new_delta = point - center, proposal - center
        correction = jnp.where(jump, .5 * (new_delta @ independence_precision @ new_delta
                                          - old_delta @ independence_precision @ old_delta), 0.)
        accept = jnp.isfinite(candidate) & (log_uniform < jnp.minimum(0., candidate - logp + correction))
        point = jnp.where(accept, proposal, point)
        logp = jnp.where(accept, candidate, logp)
        return (point, logp), (point, logp, accept)
    return lax.scan(transition, (current, value), (increments, log_uniforms, independent))


def coordinate_sampler(initial, covariance, pg, omega, groups, iterations, burn,
                       random_state, backend='compiled', lambda_power=.5,
                       kernel='random-walk', center=None):
    """Fixed-proposal MH using the repository's complete native likelihood.

    Compiled and vanilla loops consume the same NumPy random stream. Draws
    stay in sampling coordinates for covariance-weighted consensus.
    The mixture selects a fixed independence-MH Gaussian on 80% of steps and
    symmetric RWMH otherwise. Each kernel uses its own full Hastings ratio;
    their constant-weight mixture preserves the same target. The fixed center
    is required explicitly, including when continuing a saved chain.
    lambda_power=0 retains log(lambda) for diagnostic comparisons only.
    """
    if kernel not in {'random-walk', 'mixture'}:
        raise ValueError('Unknown MH kernel')
    if kernel == 'mixture' and center is None:
        raise ValueError('A fixed MAP center is required for mixture-MH recovery')
    rng = np.random.default_rng(random_state)
    center = np.zeros(4) if center is None else np.asarray(center)
    # covariance is the existing 2.38^2/p Hessian random-walk proposal.
    independence_covariance = covariance * (1.2**2 / (2.38**2/4))
    independence_precision = np.linalg.inv(independence_covariance)
    increments = np.empty((iterations, 4))
    uniforms = np.empty(iterations)
    independent = np.zeros(iterations, dtype=bool)
    for index in range(iterations):
        independent[index] = kernel == 'mixture' and rng.random() < .8
        proposal_covariance = independence_covariance if independent[index] else covariance
        increments[index] = rng.multivariate_normal(np.zeros(4), proposal_covariance)
        uniforms[index] = np.log(rng.random())
    basis = _frequency_basis(omega, 0, 1)
    point = jnp.asarray(initial)
    pg = jnp.asarray(pg)
    value = coordinate_log_posterior(point, pg, basis, groups, lambda_power)
    if backend == 'compiled':
        _, output = _coordinate_steps(point, value, jnp.asarray(increments),
                                     jnp.asarray(uniforms), jnp.asarray(independent),
                                     jnp.asarray(center), jnp.asarray(independence_precision),
                                     pg, basis, groups, lambda_power)
        draws, logp, accepted = map(np.asarray, output)
    elif backend == 'vanilla':
        draws, logp, accepted = np.empty((iterations, 4)), np.empty(iterations), np.zeros(iterations, bool)
        posterior = jit(coordinate_log_posterior, static_argnames=('lambda_power',))
        for index in range(iterations):
            proposal = (center if independent[index] else point) + increments[index]
            candidate = posterior(proposal, pg, basis, groups, lambda_power)
            old_delta, new_delta = np.asarray(point) - center, np.asarray(proposal) - center
            correction = (.5 * (new_delta @ independence_precision @ new_delta
                                - old_delta @ independence_precision @ old_delta)
                          if independent[index] else 0.)
            accept = np.isfinite(candidate) and uniforms[index] < min(0., float(candidate - value) + correction)
            if accept:
                point, value = proposal, candidate
            draws[index], logp[index], accepted[index] = point, value, accept
    else:
        raise ValueError('Unknown sampler backend')
    return draws[burn:], logp[burn:], accepted[burn:]


def coordinate_sampling_quality(draws):
    """Check physical parameters, sampling coordinates and covariance moments.

    Consensus weights depend on second moments, whose mixing need not match
    parameter mixing. Whitening makes their diagnostic independent of units.
    The existing ESS/Rhat thresholds remain unchanged.
    """
    from scripts.liquidity.compare_csi300_boundary_spark import sampling_quality
    draws = np.asarray(draws)
    native = sampling_quality(from_coordinates(draws))
    coordinates = sampling_quality(draws)
    flat = draws.reshape(-1, 4)
    covariance = np.cov(flat, rowvar=False)
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return {**native, 'passed': False, 'covariance_ess_bulk_min': 0.,
                'covariance_ess_tail_min': 0., 'ess_bulk_min': 0., 'ess_tail_min': 0.}
    white = np.linalg.solve(chol, (flat-flat.mean(axis=0)).T).T.reshape(draws.shape)
    products = np.stack([white[..., i]*white[..., j] for i in range(4) for j in range(i, 4)], axis=-1)
    moments = sampling_quality(products)
    result = dict(native)
    result.update(native_ess_bulk_min=native['ess_bulk_min'],
                  native_ess_tail_min=native['ess_tail_min'],
                  coordinate_ess_bulk_min=coordinates['ess_bulk_min'],
                  coordinate_ess_tail_min=coordinates['ess_tail_min'],
                  covariance_ess_bulk_min=moments['ess_bulk_min'],
                  covariance_ess_tail_min=moments['ess_tail_min'],
                  ess_bulk_min=min(x['ess_bulk_min'] for x in [native, coordinates, moments]),
                  ess_tail_min=min(x['ess_tail_min'] for x in [native, coordinates, moments]),
                  passed=all(x['passed'] for x in [native, coordinates, moments]))
    if result['rhat_max'] is not None:
        result['rhat_max'] = max(x['rhat_max'] for x in [native, coordinates, moments])
    return result
