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


def test_analytic_curvature_matches_native_autodiff_near_cancellation():
    import jax
    import jax.numpy as jnp
    from scripts.liquidity.probe_csi300_fractional_changes import spectrum_hessian
    omega=2*np.pi*np.arange(1,500)/24_300_012
    q=4*np.sin(omega/2)**2
    pg=np.random.default_rng(44).exponential(.2,size=len(q))
    point=np.array([-.998,-4.6,-.01,.94])
    native=lambda x:-whittle_log_posterior_jit(x,0,1,jnp.asarray(pg),True,
                                             jnp.asarray(omega),200,0.,1.,3)
    expected=np.asarray(jax.hessian(native)(jnp.asarray(point)))
    np.testing.assert_allclose(spectrum_hessian(point,q,pg,200),expected,rtol=2e-8,atol=1e-5)


def test_target_keeps_gaussian_tails_and_saturated_tempering():
    omega=np.linspace(.001,3.1,100);q=4*np.sin(omega/2)**2;pg=np.ones(len(q))*.2
    for point in [np.array([-.2,-60.,0.,.1]),np.array([-.2,8.,0.,21.])]:
        value,gradient=target(point,q,pg,200)
        expected=-float(whittle_log_posterior_jit(point,0,1,pg,True,omega,200,0.,1.,3))
        np.testing.assert_allclose(value,expected,rtol=1e-12)
        assert np.isfinite(gradient).all()


def test_map_search_finds_near_cancellation_on_complete_spectrum():
    from scripts.liquidity.probe_csi300_fractional_changes import fit_spectrum
    omega=2*np.pi*np.arange(1,8192)/16384;q=4*np.sin(omega/2)**2
    truth=np.array([-.99,-4.,-.1,.94])
    pg=np.exp(truth[2])*((1+truth[0])**2-truth[0]*q)/(2*np.pi)
    pg*=((-np.expm1(-np.exp(truth[1])))**2+np.exp(-np.exp(truth[1]))*q)**(-truth[3])
    fit=fit_spectrum(q,pg,groups=200,starts=[[0.,-10.,-.1,-.08]])
    assert fit['passed'] and fit['full_frequency_starts']>=2
    assert fit['theta'][0]<-.9 and fit['theta'][3]>.8
    assert fit['negative_log_posterior']<=target(truth,q,pg,200)[0]+1e-6
    assert len(fit['map_candidates'])>=2


def test_map_search_expands_active_computational_d_bound():
    from scripts.liquidity.probe_csi300_fractional_changes import optimize_spectrum_start
    q=np.linspace(.001,3.999,3001)
    truth=np.array([.2,-4.,.01,3.5])
    pg=np.exp(truth[2])*((1+truth[0])**2-truth[0]*q)/(2*np.pi)
    pg*=((-np.expm1(-np.exp(truth[1])))**2+np.exp(-np.exp(truth[1]))*q)**(-truth[3])
    fit=optimize_spectrum_start([.2,-4.,.01,1.],q,pg,200)
    assert fit['passed'] and fit['search_expansions']>=1
    assert fit['search_bounds'][3][1]>2 and fit['theta'][3]>2


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
