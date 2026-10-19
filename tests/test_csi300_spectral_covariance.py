import json

import numpy as np
import pytest

from scripts.liquidity import _spectral_covariance as covariance


def test_quadrature_density_equals_native_whittle_and_induced_prior():
    from scripts.liquidity._spectral_coordinates import to_coordinates, log_jacobian
    from dts.mcmc import whittle_log_posterior_jit
    omega = np.linspace(.001, np.pi-.01, 200)
    q = 4*np.sin(omega/2)**2
    pg = np.random.default_rng(40).exponential(size=len(omega))
    native = np.array([-.9976, -4.6, -.01, .94])
    point = to_coordinates(native)
    actual = covariance.coordinate_densities(point[None], q, pg, 20)[0]
    expected = float(whittle_log_posterior_jit(native, 0, 1, pg, True, omega,
                     20, 0., 1., 3, False)) + log_jacobian(point)/20
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-12)


def gaussian_density(center, cov):
    precision = np.linalg.inv(cov)
    def density(points, q, pg, groups):
        delta = points-center
        return -.5*np.einsum('ni,ij,nj->n', delta, precision, delta)
    return density


def test_quadrature_recovers_known_gaussian_covariance(monkeypatch):
    center = np.array([-3., .1, -.01, .94])
    cov = np.diag([.01, 1e-5, .001, .0001])
    monkeypatch.setattr(covariance, 'coordinate_densities', gaussian_density(center, cov))
    result = covariance.quadrature_moments(np.ones(2), np.ones(2), 20, center, cov)
    assert result['passed'] and result['order'] == 7
    np.testing.assert_allclose(result['mean'], center, atol=1e-13)
    np.testing.assert_allclose(result['covariance'], cov, atol=1e-13)


def test_regularization_reduces_sampling_noise_and_rejects_bad_anchor():
    anchor = np.diag([.01, 1e-5, .001, .0001])
    draws = np.random.default_rng(41).multivariate_normal(np.zeros(4), anchor, 8192)
    regularized, quality = covariance.regularize_covariance(draws, anchor)
    sample = np.cov(draws.T)
    assert 0 < quality['shrinkage_strength'] <= 1
    assert np.linalg.norm(regularized-anchor) <= np.linalg.norm(sample-anchor)
    np.linalg.cholesky(regularized)
    with pytest.raises(ValueError, match='covariance disagreement'):
        covariance.regularize_covariance(draws, anchor*2)


def test_covariance_checkpoint_repairs_sidecar_and_rejects_changed_samples(tmp_path, monkeypatch):
    center = np.array([-3., .1, -.01, .94])
    cov = np.diag([.01, 1e-5, .001, .0001])
    draws = np.random.default_rng(42).multivariate_normal(center, cov, (1, 8192))
    for name in ['shards', 'frequencies']:
        (tmp_path/name).mkdir()
    sample = tmp_path/'shards/G20_shard000.npz'
    np.savez(sample, coordinate_draws=draws, quality_json=json.dumps(dict(
        coordinate_map=dict(theta=center.tolist(), covariance=cov.tolist()))))
    np.savez(tmp_path/'frequencies/shard000.npz', q=np.ones(2), periodogram=np.ones(2))
    monkeypatch.setattr(covariance, 'coordinate_densities', gaussian_density(center, cov))
    result = covariance.ensure_covariance(0, tmp_path, dict(groups=20))
    sidecar = tmp_path/'covariances/shard000.json'
    sidecar.unlink()
    monkeypatch.setattr(covariance, 'quadrature_moments', lambda *a, **k:pytest.fail('Recomputed cached integration'))
    assert covariance.ensure_covariance(0, tmp_path, dict(groups=20)) == result
    assert json.loads(sidecar.read_text()) == result
    np.savez(sample, coordinate_draws=draws*2)
    with pytest.raises(ValueError, match='cache inputs changed'):
        covariance.ensure_covariance(0, tmp_path, dict(groups=20))


def test_consensus_accepts_covariances_without_changing_the_gaussian_product():
    from dts.aggregation import consensus
    rng = np.random.default_rng(43)
    left = rng.normal(size=(20000, 1))
    right = rng.normal(2., 2., size=(20000, 1))
    combined = consensus([left, right], ridge=0., covariances=[np.array([[1.]]), np.array([[4.]])])
    np.testing.assert_allclose(combined, .8*left+.2*right, atol=1e-14)
    assert abs(combined.mean()-.4) < .03
    assert abs(combined.var()-.8) < .03
    with pytest.raises(ValueError, match='One covariance'):
        consensus([left, right], covariances=[np.eye(1)])


def test_sampling_reuse_requires_the_identical_target_and_sampling_contract():
    from scripts.liquidity.compare_csi300_boundary_spark import (
        validate_sampling_reuse, sampling_contract_hash, SAMPLING_CONTRACT_SHA256,
        COVARIANCE_PREVIOUS_SHA256)
    assert sampling_contract_hash() == SAMPLING_CONTRACT_SHA256
    runner='scripts/liquidity/compare_csi300_boundary_spark.py'
    previous=dict(groups=20, seed=10, parameterization='ma-sqrtlambda',
                  source_sha256={runner:COVARIANCE_PREVIOUS_SHA256,'dts/mcmc.py':'same'})
    current={**previous,'source_sha256':{runner:'new','dts/mcmc.py':'same'}}
    validate_sampling_reuse(previous,current)
    with pytest.raises(ValueError, match='identical inputs'):
        validate_sampling_reuse(previous,{**current,'seed':11})
    with pytest.raises(ValueError, match='sources changed'):
        validate_sampling_reuse(previous,{**current,'source_sha256':{runner:'new','dts/mcmc.py':'changed'}})
