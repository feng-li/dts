import json
import numpy as np
from scipy.optimize._numdiff import approx_derivative

from scripts.liquidity.compare_csi300_boundary_arima101 import (
    MODEL, arma_objective, arma_kernel, fit_arma, comparison_reference, summarize)


def test_arma_exact_products_match_explicit_whittle_posterior():
    rng = np.random.default_rng(81)
    for n in [30, 32, 512]:
        q = 4*np.sin(np.pi*np.arange(1, n//2)/n)**2
        pg = rng.exponential(size=len(q))
        for parameters in [np.array([-.4, .6, -.2]), np.array([.99, -.3, .1])]:
            value, gradient, hessian = arma_objective(parameters, q, pg, n)

            def explicit(z):
                phi, theta, ls = z
                ar = (1-phi)**2+phi*q
                ma = (1+theta)**2-theta*q
                logf = ls-np.log(2*np.pi)+np.log(ma)-np.log(ar)
                ratio = pg*np.exp(-logf)
                slope_ar = (2*(phi-1)+q)/ar
                slope_ma = (2*(theta+1)-q)/ma
                score = np.array([np.sum((ratio-1)*slope_ar),
                                  np.sum((1-ratio)*slope_ma), np.sum(1-ratio)+ls])
                objective = np.sum(logf+ratio)+.5*ls*ls+2*np.log(2)+.5*np.log(2*np.pi)
                return objective, score

            expected, score = explicit(parameters)
            np.testing.assert_allclose(value, expected, rtol=1e-12)
            np.testing.assert_allclose(gradient, score, rtol=1e-10, atol=1e-9)
            numeric = approx_derivative(lambda z: explicit(z)[1], parameters)
            np.testing.assert_allclose(hessian, numeric, rtol=3e-6, atol=1e-5)


def test_arma_fit_recovers_known_level_spectrum():
    n = 8192
    q = 4*np.sin(np.pi*np.arange(1, n//2)/n)**2
    for phi, theta in [(.7, -.25), (.9998, -.1)]:
        pg = ((1+theta)**2-theta*q)/(2*np.pi*((1-phi)**2+phi*q))
        fit = fit_arma(q, pg, n, initial_theta=-.1)
        np.testing.assert_allclose([fit['phi'], fit['theta'], fit['sigma2']],
                                   [phi, theta, 1.], atol=2e-6)
        assert fit['newton_distance'] < .01
        assert json.loads(json.dumps(fit))['n'] == n


def test_arma_conditional_forecast_matches_statsmodels():
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    rng = np.random.default_rng(6)
    history = rng.normal(size=60)
    for phi, theta in [(.6, -.2), (.9998, -.08), (-.3, .4)]:
        weights, covariance = arma_kernel(phi, theta, .7, len(history), 12)
        fitted = SARIMAX(history, order=(1, 0, 1), trend='n', initialization='stationary').filter([phi, theta, .7])
        forecast = fitted.get_forecast(12)
        np.testing.assert_allclose(weights@history, forecast.predicted_mean, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(np.diag(covariance), forecast.var_pred_mean, rtol=1e-7, atol=1e-7)
        assert np.linalg.eigvalsh(covariance).min() > 0


def test_original_comparison_uses_its_protocol_dates(tmp_path):
    from pathlib import Path
    dates = ['20211013', '20231124']
    (tmp_path/'protocol.json').write_text(json.dumps(dict(dates=dates[::-1])))
    assert comparison_reference(tmp_path, dict(seed=5)) == (tmp_path, dates)
    assert comparison_reference(tmp_path, dict(reference='/saved/full', dates=dates)) == (Path('/saved/full'), dates)


def test_summary_scores_identical_targets_without_arima110(tmp_path):
    import pandas as pd
    reference, out = tmp_path/'comparison', tmp_path/'output'
    reference.mkdir()
    out.mkdir()
    rows = []
    for model in ['SpectralDC_ARTFIMA', 'Matched_ARIMA011_MAP', 'ClockLastValue', 'LastValue']:
        for horizon in [60, 300]:
            rows.append(dict(date='20211013', day_index=20, origin='09:31', event='opening',
                horizon=horizon, model=model, actual=3., prediction=4., se=1., ae=1.,
                crps=.5, lower95=2., upper95=5., cover95=True, interval95=3.))
    pd.DataFrame(rows).to_parquet(reference/'forecasts.parquet', index=False)
    folder=out/'days/20211013'
    folder.mkdir(parents=True)
    added=pd.DataFrame(rows[:2]).assign(model=MODEL)
    added.to_parquet(folder/'forecasts.parquet', index=False)
    (folder/'completed.json').write_text(json.dumps(dict(date='20211013')))
    summarize(reference, out, ['20211013'], dict(seed=7))
    scores=pd.read_csv(out/'scores.csv')
    assert len(scores)==10 and scores['n'].eq(1).all()
    assert scores.interval_score95.eq(3.).all()
    pairs=pd.read_csv(out/'paired_rmse.csv')
    assert len(pairs)==4 and 'Matched_ARIMA110_MAP' not in set(pairs.baseline)
    assert '1-date' in (out/'REPORT.txt').read_text()
