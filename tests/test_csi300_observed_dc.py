"""Numerical contracts for the specialized large-series likelihood."""
import numpy as np
from scipy.optimize._numdiff import approx_derivative

from scripts.liquidity.fit_csi300_observed_dc import objective, diagnostics, chain
from dts.mcmc import whittle_log_posterior_jit


def test_specialized_target_gradient_and_partition():
    omega = np.linspace(.000001, 3.1, 101)
    q = 4*np.sin(omega/2)**2
    pg = np.random.default_rng(1).exponential(size=len(omega))
    for theta in [np.array([.7,-5.,-2.,.3]), np.array([-.4,-12.,-5.,1.2]),
                  np.array([.9,-1.,1.,-.2])]:
        value, gradient = objective(theta,q,pg,80)
        reference = -float(whittle_log_posterior_jit(theta,1,0,pg,True,omega,80,0.,1.,3))
        np.testing.assert_allclose(value, reference, rtol=1e-8)
        numeric = approx_derivative(lambda t:objective(t,q,pg,80)[0],theta).ravel()
        np.testing.assert_allclose(gradient,numeric,rtol=1e-4,atol=.01)
        np.testing.assert_allclose(sum(objective(theta,q[k::4],pg[k::4],4)[0] for k in range(4)),
                                   objective(theta,q,pg,1)[0],rtol=1e-12)


def test_diagnostics_detect_between_chain_disagreement():
    draws = np.random.default_rng(1).normal(size=(4,3000,4))
    rhat, ess = diagnostics(draws)
    assert max(rhat)<1.01 and min(ess)>400
    draws[0,:,0] += 5
    rhat, ess = diagnostics(draws)
    assert rhat[0]>1.1 and ess[0]<400


def test_compiled_sampler_burn_and_seed():
    q=np.array([.1,.2]);pg=np.array([.1,.2]);start=np.array([.7,-5.,-2.,.3])
    first,accept=chain(q,pg,80,start,np.eye(4)*.001,20,100,1)
    second,_=chain(q,pg,80,start,np.eye(4)*.001,20,100,1)
    assert first.shape==(20,4) and np.isfinite(first).all() and 0<=accept<=1
    np.testing.assert_array_equal(first,second)
