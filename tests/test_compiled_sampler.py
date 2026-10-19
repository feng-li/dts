"""The compiled sampler keeps the native posterior and Metropolis rule."""
import numpy as np
import pytest

from dts.mcmc import (
    _cached_whittle_log_posterior, _frequency_basis, sampler,
    whittle_log_posterior_jit,
)


CASES = [
    (0, 0, False, False, [0.1]),
    (2, 1, False, False, [0.2, -0.1, 0.3, -0.2]),
    (0, 1, True, False, [0.2, -8.0, -0.2, 0.3]),
    (2, 2, True, False, [0.1, -0.1, 0.2, -0.2, -5.0, -0.2, 0.3]),
    (1, 0, False, True, [0.2, -0.2, 0.4]),
    (1, 1, True, True, [0.2, 0.1, -5.0, -0.2, 0.3, 0.4]),
]


def spectrum(n=512):
    omega = 2*np.pi*np.arange(1, n//2)/n
    return np.random.default_rng(8).exponential(size=len(omega)), omega


@pytest.mark.parametrize('ar,ma,tfi,mean,params', CASES)
def test_cached_target_matches_native_prior_and_mean_assembly(ar, ma, tfi, mean, params):
    pg, omega = spectrum()
    point = np.array(params)
    last = 3 if tfi else 1
    kwargs = dict(mean_term=mean, mean_ybar=0.5 if mean else None,
                  mean_series_length=512 if mean else None,
                  mean_prior_mu=-0.1, mean_prior_sd=0.8)
    expected = whittle_log_posterior_jit(point, ar, ma, pg, tfi, omega,
                                       200, 0.3, 1.7, last, **kwargs)
    actual = _cached_whittle_log_posterior(point, ar, ma, pg,
                  _frequency_basis(omega, ar, ma), tfi, 200, 0.3, 1.7, last, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize('params', [
    [-0.2, -60., 0., 0.1], [-0.2, -50., 0., -0.1],
    [-0.2, 5., 0., 21.], [-0.2, -5., -45., 0.1],
])
def test_compiled_target_does_not_truncate_gaussian_prior_tails(params):
    pg, omega = spectrum()
    point = np.array(params)
    actual = _cached_whittle_log_posterior(point, 0, 1, pg,
                         _frequency_basis(omega, 0, 1), True, 200, 0., 1., 3)
    expected = whittle_log_posterior_jit(point, 0, 1, pg, True, omega, 200, 0., 1., 3)
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)


@pytest.mark.parametrize('phi,theta', [(1-1e-8, 1-1e-8),
                                      (-1+1e-8, -1+1e-8)])
def test_cached_first_order_spectrum_resolves_both_frequency_endpoints(phi, theta):
    omega = np.array([2*np.pi/24_300_012, .02, .2, np.pi-.02,
                      np.pi-2*np.pi/24_300_012])
    pg = np.ones(len(omega))
    point = np.array([phi, theta, -50., -.2, .1])
    actual = _cached_whittle_log_posterior(point, 1, 1, pg,
                         _frequency_basis(omega, 1, 1), True, 200, 0., 1., 3)
    expected = whittle_log_posterior_jit(point, 1, 1, pg, True, omega, 200, 0., 1., 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)


@pytest.mark.parametrize('ar,ma,tfi,mean,params', CASES)
def test_compiled_and_vanilla_use_matching_random_streams_and_transitions(ar, ma, tfi, mean, params):
    pg, omega = spectrum(256)
    args = dict(ar_order=ar, ma_order=ma, data=None, I_pg=pg, TFI_term=tfi,
                omega_shard=omega, n_samples=128, Burn_in=20,
                paramsStar=np.array(params), proposal_width=np.eye(len(params))*.003,
                n_groups=7, random_state=15, mean_term=mean,
                mean_data=(.5, 256) if mean else None)
    compiled = sampler(**args)
    vanilla = sampler(**args, backend='vanilla')
    np.testing.assert_allclose(compiled[0], vanilla[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(compiled[1], vanilla[1], rtol=1e-12, atol=1e-8)
    np.testing.assert_array_equal(compiled[2], vanilla[2])


def test_progress_chunks_keep_seeded_draws_and_burn_in():
    pg, omega = spectrum(64)
    args = dict(ar_order=0, ma_order=0, data=None, I_pg=pg, TFI_term=False,
                omega_shard=omega, n_samples=1100, Burn_in=100,
                paramsStar=np.array([.1]), proposal_width=np.array([[.01]]),
                random_state=19)
    full = sampler(**args)
    chunks = sampler(**args, progress=True)
    for expected, actual in zip(full, chunks):
        np.testing.assert_array_equal(actual, expected)


def test_exact_likelihood_retains_vanilla_path_and_backend_validation():
    args = dict(ar_order=1, ma_order=0, data=np.random.default_rng(4).normal(size=64),
                I_pg=None, TFI_term=False, omega_shard=None, exact=True,
                n_samples=12, Burn_in=2, paramsStar=np.array([.2, 0.]),
                proposal_width=np.eye(2)*.003, random_state=15)
    for actual, expected in zip(sampler(**args), sampler(**args, backend='vanilla')):
        np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match='backend'):
        sampler(**args, backend='typo')


def test_sampler_backend_is_separate_from_execution_backend():
    from dts.experiments import MCMCSettings
    settings = MCMCSettings(backend='local', sampler_backend='vanilla')
    assert settings.backend == 'local' and settings.sampler_backend == 'vanilla'
    assert MCMCSettings().sampler_backend == 'compiled'
