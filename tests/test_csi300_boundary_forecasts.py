import numpy as np

from scripts.liquidity.compare_csi300_boundary_random_days import (
    clock_calibration, standardized_training_changes, increment_kernel,
    weighted_variance_diagonal, prediction_record, chain_quality)
from scripts.liquidity.probe_csi300_fractional_changes import target
from scripts.liquidity.run_csi300_fractional_changes_dc import target_value, regular_target_value, compiled_rwm


def test_null_score_and_curvature_preserve_original_posterior():
    from scipy.optimize._numdiff import approx_derivative
    from scripts.liquidity.probe_csi300_fractional_changes import null_objective, null_hessian
    q = np.linspace(.001, 3.999, 101)
    pg = np.random.default_rng(16).exponential(size=len(q))
    z = np.array([-.2, -.1])
    value, gradient = null_objective(z, q, pg)
    native, native_gradient = target(np.array([z[0], 0., z[1], 0.]), q, pg)
    np.testing.assert_allclose(value*len(q), native-np.log(2*np.pi), rtol=1e-12)
    np.testing.assert_allclose(gradient*len(q), native_gradient[[0, 2]], rtol=1e-12)
    numeric = approx_derivative(lambda x: null_objective(x, q, pg)[1]*len(q), z)
    np.testing.assert_allclose(null_hessian(z, q, pg), numeric, rtol=1e-8, atol=1e-8)


def test_null_gate_accepts_stationary_fit_despite_optimizer_failure(monkeypatch):
    from scipy.optimize import minimize, OptimizeResult
    from scripts.liquidity import probe_csi300_fractional_changes as probe
    q = np.linspace(.001, 3.999, 501)
    pg = np.exp(-.2)*((1-.3)**2+.3*q)/(2*np.pi)
    actual = minimize(lambda z: probe.null_objective(z, q, pg), [-.2, 0.],
                      jac=True, method='BFGS', options={'gtol': 1e-10})
    monkeypatch.setattr(probe, 'minimize', lambda *args, **kwargs:
                        OptimizeResult(x=actual.x, success=False, message='ABNORMAL', fun=actual.fun))
    fit = probe.fit_null_spectrum(q, pg, [-.2, 0.])
    assert fit['null_success'] and not fit['null_optimizer_success']
    assert fit['null_newton_distance'] < .01
    np.testing.assert_array_equal(fit['null_theta'], actual.x)
    assert len(fit['null_attempts']) == 1


def test_null_gate_rejects_nonstationary_fit_despite_optimizer_success(monkeypatch):
    from scipy.optimize import OptimizeResult
    from scripts.liquidity import probe_csi300_fractional_changes as probe
    q = np.linspace(.001, 3.999, 501)
    pg = np.ones(len(q))*.5
    monkeypatch.setattr(probe, 'minimize', lambda *args, **kwargs:
                        OptimizeResult(x=np.array([0., 1.]), success=True, message='success', fun=1.))
    fit = probe.fit_null_spectrum(q, pg, [0., 1.])
    assert not fit['null_success'] and fit['null_optimizer_success']
    assert len(fit['null_attempts']) == 2


def test_saved_configuration_roundtrips_and_cli_overrides(tmp_path):
    import json
    from scripts.liquidity.compare_csi300_boundary_random_days import parse_arguments, run_configuration
    args = parse_arguments(['--out', str(tmp_path), '--days', '1000', '--ray-actors', '100'])
    path = tmp_path/'run_config.json'
    settings = run_configuration(args)
    path.write_text(json.dumps(settings))
    assert run_configuration(parse_arguments(['--config', str(path)])) == settings
    overridden = parse_arguments(['--config', str(path), '--ray-actors', '50'])
    assert overridden.ray_actors == 50 and overridden.days == 1000


def test_configuration_rejects_invalid_sampling_arguments(tmp_path):
    import json, pytest
    from scripts.liquidity.compare_csi300_boundary_random_days import parse_arguments
    path = tmp_path/'run_config.json'
    path.write_text(json.dumps(dict(chains=1)))
    with pytest.raises(SystemExit):
        parse_arguments(['--config', str(path)])
    with pytest.raises(SystemExit):
        parse_arguments(['--days', '1001'])


def test_resume_rejects_new_map_search_over_old_sampling_results(tmp_path):
    import hashlib, json, pytest
    from pathlib import Path
    from scripts.liquidity.compare_csi300_boundary_random_days import LEGACY_SOURCE_SHA256, register_protocol
    source = Path(__file__).resolve().parents[1]/'scripts/liquidity'
    current = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
               for name in LEGACY_SOURCE_SHA256}
    path = tmp_path/'protocol.json'
    folder = tmp_path/'days/20211126'
    folder.mkdir(parents=True)
    (folder/'fit.json').write_text('{}')
    path.write_text(json.dumps(dict(days=1000, source_sha256=LEGACY_SOURCE_SHA256)))
    config = dict(days=1000, source_sha256=current)
    with pytest.raises(ValueError,match='incompatible'):
        register_protocol(path, config)
    saved = json.loads(path.read_text())
    assert saved['source_sha256'] == LEGACY_SOURCE_SHA256
    assert 'source_history' not in saved


def test_resume_rejects_changed_inference_configuration(tmp_path):
    import json, pytest
    from scripts.liquidity.compare_csi300_boundary_random_days import register_protocol
    path = tmp_path/'protocol.json'
    folder = tmp_path/'days/20211126'
    folder.mkdir(parents=True)
    (folder/'fit.json').write_text('{}')
    previous = dict(days=1000, groups=2, source_sha256={})
    path.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match='incompatible'):
        register_protocol(path, {**previous, 'groups': 4})
    assert json.loads(path.read_text()) == previous


def test_value_only_mh_kernel_preserves_native_target():
    rng = np.random.default_rng(4)
    q = np.linspace(.00001, 3.999, 231)
    pg = rng.exponential(size=len(q))
    for groups in [1, 2, 4]:
        for theta in [np.array([-.3, -6., .1, .2]), np.array([.6, -2., -.8, -.2])]:
            np.testing.assert_allclose(target_value(theta, q, pg, groups),
                                       target(theta, q, pg, groups)[0], rtol=1e-13)


def test_exact_regular_frequency_sum_matches_explicit_native_target():
    rng = np.random.default_rng(9)
    for n in [30, 32, 512, 10000]:
        q = 4*np.sin(np.pi*np.arange(1, n//2)/n)**2
        pg = rng.exponential(size=len(q))
        for groups in [1, 2]:
            for group in range(groups):
                for ma in [-.7, .3]:
                    for ll in [-20., -3., 2.]:
                        theta = np.array([ma, ll, -.3, .2])
                        explicit = target_value(theta, q[group::groups], pg[group::groups], groups)
                        exact = regular_target_value(theta, q[group::groups], pg[group::groups], groups, n, group)
                        np.testing.assert_allclose(exact, explicit, rtol=2e-12, atol=1e-9)


def test_regular_sampler_agrees_with_explicit_sampler():
    n = 1000
    q = 4*np.sin(np.pi*np.arange(1, n//2)/n)**2
    pg = np.random.default_rng(2).exponential(size=len(q))
    theta = np.array([-.2, -3., 0., .1])
    for group in [0, 1]:
        a, _ = compiled_rwm(q[group::2], pg[group::2], 2, theta, np.eye(4)*.01, 200, 50, 4)
        b, _ = compiled_rwm(q[group::2], pg[group::2], 2, theta, np.eye(4)*.01, 200, 50, 4, n, group)
        np.testing.assert_array_equal(a, b)


def test_clock_profile_recovers_edge_changes_without_future_information():
    clock = np.linspace(0, .2, 14220)
    train = np.exp(np.tile(clock, (3, 1)))
    profile, mean, scale, audit = clock_calibration(train)
    combined = np.r_[train, train[:1]*100]
    np.testing.assert_array_equal(clock_calibration(combined[:3])[0], profile)
    np.testing.assert_allclose(profile, clock, atol=1e-12)
    np.testing.assert_allclose(standardized_training_changes(train, mean, scale), 0., atol=1e-10)


def test_clock_normalization_removes_mean_and_variance_by_valid_pair():
    rng = np.random.default_rng(20)
    increments = rng.normal(size=(100, 14220))*np.linspace(.001, .01, 14220)
    increments += np.linspace(-.01, .01, 14220)
    train = np.exp(np.cumsum(increments, axis=1))
    train[0, 500] = np.nan
    _, mean, scale, _ = clock_calibration(train)
    delta = np.diff(np.log(train), axis=1)
    y = (delta-mean[1:])/scale[1:]
    y[:, 7199] = np.nan  # Excluded lunch transition.
    good = np.r_[np.arange(7199), np.arange(7200, 14219)]
    np.testing.assert_allclose(np.nanmean(y[:, good], axis=0), 0., atol=1e-12)
    np.testing.assert_allclose(np.nanstd(y[:, good], axis=0), 1., atol=1e-12)


def test_weighted_reconstruction_preserves_joint_path_variance():
    from scripts.liquidity.probe_csi300_fractional_changes import increment_predictor
    theta = np.array([-.2, -4., -.3, -.1])
    weights, covariance, chol = increment_kernel(theta, 30, 12)
    cumulative, full = increment_predictor(theta, 30, 12)
    np.testing.assert_allclose(np.cumsum(weights, axis=0), cumulative, atol=1e-12)
    constant = np.full(12, .2)
    np.testing.assert_allclose(weighted_variance_diagonal(covariance, constant), .04*np.diag(full), atol=1e-12)
    varying = np.linspace(.1, .5, 12)
    matrix = np.tril(np.ones((12, 12)))*varying
    np.testing.assert_allclose(weighted_variance_diagonal(covariance, varying),
                               np.diag(matrix@covariance@matrix.T), atol=1e-12)


def test_missing_target_preserves_forecast_but_not_score():
    record = prediction_record(np.nan, 2., np.array([1., 2., 3., 4.]))
    assert record['prediction'] == 2.
    assert np.isfinite(record['lower95'])
    assert np.isnan(record['ae']) and np.isnan(record['cover95'])


def test_fixed_sampler_is_seeded_and_retains_declared_draws():
    q = np.array([.1, .3, .8])
    pg = np.array([.2, .3, .1])
    theta = np.array([-.2, -3., 0., .1])
    a, rate = compiled_rwm(q, pg, 2, theta, np.eye(4)*.01, 40, 10, 6)
    b, _ = compiled_rwm(q, pg, 2, theta, np.eye(4)*.01, 40, 10, 6)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (30, 4) and 0 <= rate <= 1


def test_sampling_gate_rejects_stuck_chains():
    draws = np.zeros((4, 100, 4))
    assert not chain_quality(draws)['passed']
