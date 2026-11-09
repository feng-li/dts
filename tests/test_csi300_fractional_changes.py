import numpy as np
from scipy.optimize._numdiff import approx_derivative
from dts.mcmc import whittle_log_posterior_jit
from scripts.liquidity.probe_csi300_fractional_changes import target,increment_predictor,minute_profile,training_changes
from scripts.liquidity.compare_csi300_forecasts import kernel


def test_change_target_is_native_manuscript_target():
    w=np.linspace(.001,3.1,61);q=4*np.sin(w/2)**2;pg=np.random.default_rng(3).exponential(size=len(q))
    for x in [np.array([-.3,-5.,.2,-.1]),np.array([.7,-3.,-.5,.4])]:
        value,g=target(x,q,pg,4)
        native=-float(whittle_log_posterior_jit(x,0,1,pg,True,w,4,0.,1.,3))
        np.testing.assert_allclose(value,native,rtol=1e-10)
        numeric=approx_derivative(lambda z:target(z,q,pg,4)[0],x).ravel()
        np.testing.assert_allclose(g,numeric,rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(sum(target(x,q[j::4],pg[j::4],4)[0] for j in range(4)),target(x,q,pg,1)[0],rtol=1e-12)


def test_zero_fractional_component_recovers_arima011():
    theta=-.4;sigma=.3;context=30;horizon=12
    w,cov=increment_predictor(np.array([theta,-5.,np.log(sigma),0.]),context-1,horizon)
    old,oldcov,_=kernel('ARIMA011',dict(theta=theta,sigma2=sigma),context,horizon)
    history=np.random.default_rng(8).normal(size=context)
    np.testing.assert_allclose(history[-1]+w@np.diff(history),old@history,atol=1e-12)
    np.testing.assert_allclose(cov,oldcov,atol=1e-12)


def test_changes_predictor_is_invariant_to_historical_level():
    w,cov=increment_predictor(np.array([-.3,-5.,-.8,.2]),29,12)
    history=np.random.default_rng(5).normal(size=30)
    prediction=history[-1]+w@np.diff(history)
    shifted=history+2.
    np.testing.assert_allclose(shifted[-1]+w@np.diff(shifted),prediction+2.,atol=1e-12)
    assert np.linalg.eigvalsh(cov).min()>0


def test_clock_profile_is_frozen_from_training_and_recovers_pulse():
    pulse=np.tile(np.r_[.2,np.zeros(59)],237)
    train=np.exp(np.tile(pulse,(3,1)))
    base=np.zeros(14220)
    profile=minute_profile(train,base)
    np.testing.assert_allclose(profile,pulse-pulse.mean(),atol=1e-12)
    # A later day's changed scale cannot enter the function's training slice.
    all_days=np.r_[train,train[:1]*100]
    np.testing.assert_allclose(minute_profile(all_days[:3],base),profile,atol=1e-12)


def test_no_boundary_jump_is_treated_as_a_second_change():
    train=np.ones((2,14220));train[0,7200:]*=2.;train[1]*=3.
    train[0,100:105]=np.nan
    changes,audit=training_changes(train,np.zeros(14220),'exclude')
    np.testing.assert_array_equal(changes,np.zeros(len(changes)))
    assert audit['boundary_changes']==4
    kept,_=training_changes(train,np.zeros(14220),'keep')
    assert np.count_nonzero(kept)==2
