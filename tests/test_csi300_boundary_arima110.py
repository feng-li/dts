import numpy as np
from scipy.optimize._numdiff import approx_derivative

from scripts.liquidity.compare_csi300_boundary_arima110 import ar1_objective, fit_ar1


def test_exact_ar1_sums_match_explicit_whittle_likelihood():
    rng = np.random.default_rng(39)
    for n in [30, 32, 512]:
        q = 4*np.sin(np.pi*np.arange(1, n//2)/n)**2
        pg = rng.exponential(size=len(q))
        for phi in [-.9, -.3, 0., .7, .99]:
            parameters = np.array([phi, -.2])
            value, gradient, hessian = ar1_objective(parameters, n, pg.sum(), q@pg)

            def explicit(z):
                coefficient, log_variance = z
                denominator = (1-coefficient)**2+coefficient*q
                logf = log_variance-np.log(2*np.pi)-np.log(denominator)
                ratio = pg*np.exp(-logf)
                value = (np.sum(logf+ratio)+.5*log_variance**2
                         + np.log(2)+.5*np.log(2*np.pi))
                derivative = (2*(coefficient-1)+q)/denominator
                score = np.array([np.sum((ratio-1)*derivative),
                                  np.sum(1-ratio)+log_variance])
                return value, score

            expected, score = explicit(parameters)
            np.testing.assert_allclose(value, expected, rtol=1e-12)
            np.testing.assert_allclose(gradient, score, rtol=1e-10, atol=1e-9)
            numeric_hessian = approx_derivative(lambda z: explicit(z)[1], parameters)
            np.testing.assert_allclose(hessian, numeric_hessian, rtol=3e-6, atol=1e-5)


def test_ar1_map_recovers_known_spectrum():
    n = 4096
    q = 4*np.sin(np.pi*np.arange(1, n//2)/n)**2
    for phi in [-.6, .35]:
        pg = 1/(2*np.pi*((1-phi)**2+phi*q))
        fit = fit_ar1(q, pg, n)
        np.testing.assert_allclose(fit['phi'], phi, atol=1e-7)
        np.testing.assert_allclose(fit['sigma2'], 1., atol=1e-7)
        assert fit['newton_distance'] < .01


def test_ar1_reconstruction_matches_direct_conditional_forecast():
    from scripts.liquidity.compare_csi300_forecasts import kernel
    from scripts.liquidity.compare_csi300_boundary_random_days import weighted_variance_diagonal
    phi, sigma2, horizon = -.2, .7, 12
    weights, covariance, _ = kernel('AR1', dict(phi=phi, sigma2=sigma2), 1, horizon)
    np.testing.assert_allclose(weights[:, 0], phi**np.arange(1, horizon+1))
    rng = np.random.default_rng(42)
    normal = rng.normal(size=(50_000, horizon))
    innovations = np.sqrt(sigma2)*normal
    for h in range(1, horizon):
        innovations[:, h] += phi*innovations[:, h-1]
    scale = np.linspace(.01, .05, horizon)
    integrated = np.cumsum(scale*innovations, axis=1)
    expected = weighted_variance_diagonal(covariance, scale)
    np.testing.assert_allclose(integrated.var(axis=0), expected, rtol=.03)
