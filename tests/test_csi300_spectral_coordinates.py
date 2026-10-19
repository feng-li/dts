import numpy as np
import pytest

from scripts.liquidity._spectral_coordinates import (
    to_coordinates, from_coordinates, log_jacobian, coordinate_objective,
    coordinate_log_posterior, coordinate_sampler)


@pytest.mark.parametrize('power', [0., .5])
def test_fractional_jacobian_preserves_full_posterior_product(power):
    from dts.mcmc import _frequency_basis, whittle_log_posterior_jit
    import jax.numpy as jnp
    n, groups = 240, 20
    omega = 2*np.pi*np.arange(1, n//2)/n
    pg = np.random.default_rng(1).exponential(size=len(omega))
    for ma in [-.9976, -.4, .7]:
        native = np.array([ma, -4.6, -.01, .94])
        point = to_coordinates(native, power)
        actual = sum(float(coordinate_log_posterior(jnp.asarray(point), pg[g::groups],
                     _frequency_basis(omega[g::groups], 0, 1), groups, power)) for g in range(groups))
        expected = float(whittle_log_posterior_jit(native, 0, 1, pg, True, omega,
                                                  1, 0., 1., 3, False)) + log_jacobian(point, power)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(from_coordinates(point, power), native, atol=1e-15)


@pytest.mark.parametrize('power', [0., .5])
def test_transformed_derivatives_match_native_autodiff(power):
    import jax
    import jax.numpy as jnp
    from dts.mcmc import _frequency_basis
    omega = np.linspace(.0001, np.pi-.001, 300)
    q = 4*np.sin(omega/2)**2
    pg = np.random.default_rng(2).exponential(size=len(omega))
    basis = _frequency_basis(omega, 0, 1)
    point = to_coordinates([-.9976, -4.6, -.01, .94], power)
    objective = lambda x: -coordinate_log_posterior(x, pg, basis, 20, power)
    value, gradient, curvature = coordinate_objective(point, q, pg, 20, power)
    np.testing.assert_allclose(value, objective(jnp.asarray(point)), rtol=1e-11)
    np.testing.assert_allclose(gradient, jax.grad(objective)(jnp.asarray(point)), rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(curvature, jax.hessian(objective)(jnp.asarray(point)), rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize('kernel', ['random-walk', 'mixture'])
def test_compiled_sampler_matches_vanilla_and_resumes_rng(kernel):
    omega = np.linspace(.001, np.pi-.001, 60)
    pg = np.random.default_rng(3).exponential(size=len(omega))
    initial = to_coordinates([-.9, -4., -.1, .7])
    covariance = np.diag([.01, 1e-5, .0001, .0001])
    args = (initial, covariance, pg, omega, 20, 200, 0)
    settings = dict(kernel=kernel, center=initial)
    compiled = coordinate_sampler(*args, random_state=42, **settings)
    vanilla = coordinate_sampler(*args, random_state=42, backend='vanilla', **settings)
    for left, right in zip(compiled, vanilla):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-11)
    rng = np.random.default_rng(42)
    first = coordinate_sampler(initial, covariance, pg, omega, 20, 80, 0, rng, **settings)
    second = coordinate_sampler(first[0][-1], covariance, pg, omega, 20, 120, 0, rng, **settings)
    for full, a, b in zip(compiled, first, second):
        np.testing.assert_allclose(full, np.concatenate([a, b]), rtol=1e-12, atol=1e-11)


def test_negative_tempering_proposal_has_zero_density():
    import jax.numpy as jnp
    from dts.mcmc import _frequency_basis
    omega = np.linspace(.001, np.pi-.001, 60)
    point = to_coordinates([-.9, -4., -.1, .7])
    point[1] = -.1
    value = coordinate_log_posterior(jnp.asarray(point), np.ones(60), _frequency_basis(omega, 0, 1), 20)
    assert np.isneginf(value)
    with pytest.raises(ValueError):
        from_coordinates(point)


def test_independence_move_uses_the_full_hastings_correction():
    import jax.numpy as jnp
    from scipy.stats import multivariate_normal
    from dts.mcmc import _frequency_basis
    from scripts.liquidity._spectral_coordinates import _coordinate_steps
    from scripts.liquidity.run_csi300_fractional_changes_dc import target_value
    omega = np.linspace(.01, np.pi-.01, 50)
    pg = np.random.default_rng(18).exponential(size=len(omega))
    q = 4*np.sin(omega/2)**2
    center = to_coordinates([-.9, -4., -.1, .7])
    covariance = np.diag([.01, 1e-5, .0001, .0001])
    old, proposal = center.copy(), center + np.array([.05, .001, .01, -.02])
    posterior = lambda z: -target_value(from_coordinates(z), q, pg, 20) + log_jacobian(z)/20
    old_value, new_value = posterior(old), posterior(proposal)
    hastings = multivariate_normal.logpdf(old, center, covariance) - multivariate_normal.logpdf(proposal, center, covariance)
    if new_value-old_value+hastings > 0:
        old, proposal = proposal, old
        old_value, new_value = new_value, old_value
        hastings = -hastings
    threshold = min(0., new_value-old_value+hastings)
    assert threshold < -1e-5 and abs(hastings) > .01
    for offset, expected in [(-1e-5, True), (1e-5, False)]:
        _, output = _coordinate_steps(jnp.asarray(old), jnp.asarray(old_value),
            jnp.asarray((proposal-center)[None]), jnp.asarray([threshold+offset]),
            jnp.asarray([True]), jnp.asarray(center), jnp.asarray(np.linalg.inv(covariance)),
            jnp.asarray(pg), _frequency_basis(omega, 0, 1), 20)
        assert bool(np.asarray(output[2])[0]) == expected


def test_covariance_moment_diagnostics_are_reported():
    from scripts.liquidity._spectral_coordinates import coordinate_sampling_quality
    z = np.random.default_rng(15).normal(size=(1, 8192, 4))
    z[..., 0] = -3 + .02*z[..., 0]
    z[..., 1] = .1 + .001*z[..., 1]
    quality = coordinate_sampling_quality(z)
    assert quality['passed']
    assert quality['covariance_ess_bulk_min'] > 400
    assert quality['ess_bulk_min'] <= quality['coordinate_ess_bulk_min']
    assert quality['rhat_max'] is None


def test_coordinate_block_checkpoint_recovers_rng_and_rejects_other_coordinates(tmp_path, monkeypatch):
    from scripts.liquidity import _spectral_sampling as recovery
    omega = np.linspace(.01, np.pi-.01, 50)
    pg = np.random.default_rng(16).exponential(size=len(omega))
    center = to_coordinates([-.9, -4., -.1, .7])
    def quality(draws):
        passed = draws.shape[1] >= 112
        return dict(passed=passed, ess_bulk_min=500 if passed else 0,
                    ess_tail_min=500 if passed else 0)
    settings = dict(initial=[center], covariance=np.diag([.01, 1e-5, .0001, .0001]),
                    pg=pg, omega=omega, groups=20, chains=1, iterations=64, burn=16,
                    maximum=128, extension=64, ess_target=400, seed=17,
                    quality_function=quality, parameterization='ma-sqrtlambda',
                    kernel='mixture', proposal_center=center)
    full = recovery.sampling_blocks(tmp_path/'full.npz', **settings)
    original = recovery.atomic_npz
    def interrupted(path, **values):
        original(path, **values)
        raise KeyboardInterrupt
    monkeypatch.setattr(recovery, 'atomic_npz', interrupted)
    with pytest.raises(KeyboardInterrupt):
        recovery.sampling_blocks(tmp_path/'resume.npz', **settings)
    monkeypatch.setattr(recovery, 'atomic_npz', original)
    resumed = recovery.sampling_blocks(tmp_path/'resume.npz', **settings)
    np.testing.assert_array_equal(full[0], resumed[0])
    assert full[1:] == resumed[1:]
    with pytest.raises(ValueError, match='Checkpoint parameterization'):
        recovery.sampling_blocks(tmp_path/'resume.npz', **{
            **settings, 'parameterization': 'native', 'kernel': 'random-walk'})
