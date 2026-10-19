import numpy as np
from scipy.linalg import toeplitz
from scripts.liquidity.compare_csi300_forecasts import (
    artfima_kernel, kernel, ma_profile, score_samples, transform, window_reason)


def test_ma_likelihood_matches_dense_covariance():
    x=np.random.default_rng(3).normal(size=30)
    for theta in [-.8,0.,.7]:
        cov=toeplitz(np.r_[1+theta*theta,theta,np.zeros(len(x)-2)])
        sigma=x@np.linalg.solve(cov,x)/len(x)
        expected=.5*(np.log(sigma)+np.linalg.slogdet(cov)[1]/len(x))
        actual,variance=ma_profile(theta,x)
        np.testing.assert_allclose([actual,variance],[expected,sigma],rtol=1e-12)


def test_fractional_zero_matches_exact_ar1_prediction():
    p=dict(phi=.8,sigma2=.3)
    w,c,_=kernel('AR1',p,30,12)
    fw,fc,_=artfima_kernel(np.array([.8,-5.,np.log(.3),0.]),30,12)
    np.testing.assert_allclose(fw,w,atol=1e-10)
    np.testing.assert_allclose(fc,c,atol=1e-10)


def test_integrated_zero_matches_random_walk():
    for model,parameters in [('ARIMA110',dict(phi=0.,sigma2=.3)),('ARIMA011',dict(theta=0.,sigma2=.3))]:
        w,c,_=kernel(model,parameters,30,12)
        np.testing.assert_allclose(w[:,-1],1)
        np.testing.assert_allclose(w[:,:-1],0)
        np.testing.assert_allclose(c,.3*np.minimum.outer(np.arange(1,13),np.arange(1,13)))


def test_ma_forecast_matches_gaussian_conditioning():
    theta=.7;context=30;horizon=12
    w,c,_=kernel('ARIMA011',dict(theta=theta,sigma2=1.),context,horizon)
    diff=np.random.default_rng(4).normal(size=context-1)
    history=np.r_[0,np.cumsum(diff)]
    total=context-1+horizon
    cov=toeplitz(np.r_[1+theta*theta,theta,np.zeros(total-2)])
    cross=cov[context-1:,:context-1]
    future_mean=cross@np.linalg.solve(cov[:context-1,:context-1],diff)
    future_cov=cov[context-1:,context-1:]-cross@np.linalg.solve(cov[:context-1,:context-1],cross.T)
    integrate=np.tril(np.ones((horizon,horizon)))
    np.testing.assert_allclose(w@history,history[-1]+np.cumsum(future_mean),atol=1e-12)
    np.testing.assert_allclose(c,integrate@future_cov@integrate.T,atol=1e-12)


def test_forecast_windows_and_empirical_crps():
    a=np.ones(14220)
    assert window_reason(a,0,7200,7100,600,300)=='session_boundary'
    a[1799]=np.nan
    assert window_reason(a,0,7200,1800,600,300)=='missing_history'
    a[1799]=1;a[1850]=np.nan
    assert window_reason(a,0,7200,1800,600,300)=='missing_target'
    draws=np.array([1.,2.,3.,4.]);actual=2.4
    expected=np.abs(draws-actual).mean()-.5*np.abs(draws[:,None]-draws[None,:]).mean()
    np.testing.assert_allclose(score_samples(actual,2.5,draws)['crps'],expected)


def test_transform_fitted_on_prefix_only():
    a=np.exp(np.random.default_rng(4).normal(size=(4,14220)))
    profile,scale,y=transform(a[:3])
    a[3]*=1000
    other,sd,z=transform(a[:3])
    np.testing.assert_array_equal(profile,other)
    np.testing.assert_array_equal(y,z)
    assert sd==scale
    np.testing.assert_allclose([y.mean(),y.std()],[0.,1.],atol=1e-12)
