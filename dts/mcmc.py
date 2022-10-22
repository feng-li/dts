import statsmodels.api as sm
import autograd.numpy as np
# from numdifftools import Hessian as Hess_finite_diff
import scipy.stats as sps
from scipy.stats import multivariate_normal
import autograd.scipy.stats as sps_autograd
import pandas as pd
import pickle
import sys, os, platform

from tqdm import tqdm



def f_ARTFIMA(omega, phi, theta, var, d, lambda_):
    '''
    Spectral density function
    '''
    TFI = np.abs(1 - np.exp(-(lambda_ + 1j*omega)))**(-2*d)
    if phi.any():
        log_arg_phi = np.outer(-1j*omega,np.arange(1, len(phi)+1))
        vv1 = 1/(1 - np.sum(phi * np.exp(log_arg_phi),1))
    else:
        vv1 = 1

    if theta.any():
        log_arg_theta = np.outer(-1j*omega,np.arange(1, len(theta)+1))
        vv2 = (1 + np.sum(theta * np.exp(log_arg_theta),1))
    else:
        vv2 = 1

    f = TFI * (var/(2*np.pi)) * (np.real(vv1)**2 + np.imag(vv1)**2)*(np.real(vv2)**2 + np.imag(vv2)**2)
    return f


def whittle_log_likelihood(params, q, p, I_pg, TFI_term, omega_shard):
    '''Whittle likelihood function

    #do ARTFIMA lambda and d in SD
    q: lag of AR
    p: lag of MA
    I_pg: periodogram of data
    '''
    if TFI_term: # ARTFIMA model
        d = params[-1]
        lambda_ = np.exp(params[-3])
        var = np.exp(params[-2])
    else: # ARMA model
        d = 0
        lambda_ = 0
        var = np.exp(params[-1])
    if q>0:
        phi = np.array(reparam(params[:q]))
    else:
        phi = np.array([])

    if p>0:
        theta = np.array(reparam(params[q:q+p], MA = True))
    else:
        theta = np.array([])

    fj = f_ARTFIMA(omega_shard, phi, theta, var, d, lambda_)
    log_like = -(np.log(fj) + I_pg/fj)
    return log_like


def exact_log_likelihood_arma(data, params, q, p):
    '''Exact likelihood function

    '''
    phi = np.array(reparam(params[:q]))
    theta = np.array(reparam(params[q:q+p], MA = True))
    var = np.exp(params[-1])
    ans = sm.tsa.innovations.arma_loglike(data, phi, theta, sigma2=var)
    return ans

def log_prior(params, mu, sd, Last_ARMA, TFI_term):
    '''
    Prior distribution function(fractionated prior)

    '''

    if (np.abs(params[:-Last_ARMA]) < 1).all():
        prior_process_params = -len(params[:-Last_ARMA])*np.log(2)
    else:
        prior_process_params = -np.inf

    if TFI_term:
        prior_d_params = sps_autograd.norm.logpdf(params[-1], loc = mu, scale = sd)
        prior_lambda_params = sps_autograd.norm.logpdf(params[-3], loc = mu, scale = sd)
        prior_var_params = sps_autograd.norm.logpdf(params[-2], loc = mu, scale = sd)
    else:
        prior_d_params = 0
        prior_lambda_params = 0
        prior_var_params = sps_autograd.norm.logpdf(params[-1], loc = mu, scale = sd)

    return (prior_process_params + prior_d_params + prior_lambda_params + prior_var_params)


# 2.6 partial autocorrelation transform function
def reparam(params, MA = False):
    """
    Transforms params to induce stationarity/invertability.
    Takes as input parameters in the partial auto-correlation parameterization and returns parameters
    that are on the ordinary parameterization.
    """
    newparams = np.array(params, copy=True)
    tmp = np.array(params, copy=True)

    for j in range(1,len(params)):
        if not MA:
            tmp_new = tmp[:j] - np.array([(newparams[j]*newparams[j-k-1]) for k in range(j)])
        else:
            tmp_new = tmp[:j] + np.array([(newparams[j]*newparams[j-k-1]) for k in range(j)])

        tmp = np.hstack([tmp_new, newparams[j:]])
        newparams = np.hstack((tmp[:j],newparams[j:]))

    return newparams



# 2.7 Metropolis algorithm
def sampler(q, p, data, I_pg, TFI_term, omega_shard, n_samples, paramsStar, proposal_width, Burn_in, params_prior_mu=0, params_prior_sd=1., exact = False):

    if TFI_term:
        n_params = q + p + 3
        Last_ARMA = 3
    else:
        n_params = q + p + 1
        Last_ARMA = 1

    params_init = paramsStar
    params_current = params_init
    posterior_samples = np.zeros((n_samples, n_params))
    log_p = np.zeros(n_samples)
    Acceptance = np.zeros((n_samples, 1))

    # Current log likelihood
    if exact:
        log_likelihood_current = exact_log_likelihood_arma(data, params_current, q, p)
    else:
        log_likelihood_current = whittle_log_likelihood(params_current, q, p, I_pg, TFI_term, omega_shard)

    # Current log prior
    log_prior_current = log_prior(params_current, params_prior_mu, params_prior_sd, Last_ARMA, TFI_term)

    #Current log posterior
    log_p_current = np.sum(log_likelihood_current) + np.sum(log_prior_current)

    # bar = progressbar.progressbar(range(n_samples))
    for i in tqdm(range(n_samples)):

        # New position:
        params_proposal = sps.multivariate_normal.rvs(mean = params_current, cov = proposal_width)
        if (np.abs(params_proposal[:-Last_ARMA]) < 1).all():

        # Proposal log likelihood
            if exact:
                log_likelihood_proposal = exact_log_likelihood_arma(data, params_proposal, q, p)
            else:
                log_likelihood_proposal = whittle_log_likelihood(params_proposal, q, p, I_pg, TFI_term, omega_shard)

        else:
            log_likelihood_proposal = -np.inf
        # Proposal log prior
        log_prior_proposal = log_prior(params_proposal, params_prior_mu, params_prior_sd, Last_ARMA, TFI_term)

        # Proposal log posterior
        log_p_proposal = np.sum(log_likelihood_proposal) + np.sum(log_prior_proposal)

        # Accept ratio
        alpha = np.min([1, np.exp(log_p_proposal - log_p_current)])
        accept = np.random.rand() < alpha

        if accept.any():
            # Update position
            params_current = params_proposal
            log_p_current = log_p_proposal
            Acceptance[i] = 1


        posterior_samples[i, :] =  params_current
        log_p[i] = log_p_current
    return posterior_samples[Burn_in:], log_p[Burn_in:].T, Acceptance
