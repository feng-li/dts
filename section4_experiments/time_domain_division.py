#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 17:41:23 2025

@author: zixuanwang


Naive Time Domain Divide-and-Conquer

"""





import statsmodels.api as sm
import autograd.numpy as np
from autograd import grad, hessian, jacobian
from numdifftools import Hessian as Hess_finite_diff
from numpy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, Bounds, basinhopping
import autograd.scipy.stats as sps_autograd
import progressbar
import pandas as pd
import pickle
import sys, os, platform
import warnings
from scipy.signal import lfilter

'''
Consensus Monte Carlo Algorithm with spectral density and Whittle Likelihood for time series
Line Table of Contents    
  1. Divide data into G shards y_1...y_G.
    1.1 define data and model
    1.2 the acutal data is periodgram of the original dataset     
    1.3 divide periodgram:I_PG into G shards

    
  2. Run G separate Monte Carlo algorithms to sample posterior, with each shard using the fractionated prior
    2.1 spectral density function
    2.2 Whittle likelihood function
    2.3 exact likelihood function
    2.4 prior distribution function(fractionated prior)
    2.5 posterior distribution function
    2.6 partial autocorrelation transform function
    2.7 Metropolis algorithm
    2.8 Compute a proposal width for all MCMC
    2.9 simulated parallel sampling(but actually here we do it one by one in a for loop, by using the group index:ind)
           2.9.1 each worker gets its shard of data(periodgram)
           2.9.2 find own MAP(paramsStar), as start point
           2.9.3 run sampling, jackknife bias correction and restore all results as PA_samples
           2.9.4 calculate sample variance(coveriance) of each shard, inverse as the weights - w()


  3. Combine the draws across shards using weighted averages
    3.1 combine the draws: sum all weights -> inverse -> multiply each shard's weight and draws -> add togather
    3.2 transform partial autocorrelated params to ARMA params


'''
gtol = 1e-4 
max_iter_optim = 500 
np.random.seed(10)

#################################################
# 1. Divide data y into S shards y_1...y_s.
#################################################

# 1.1 define data and model

if platform.system() == 'Darwin':
    proj_path = '/Users/' + os.getenv("USER") + '/OneDrive - UTS/Project 1/'
elif platform.system() == 'Windows':
    proj_path = '/Users/' + os.getlogin() + '/OneDrive - UTS/Project 1/'
else:
    raise ValueError()      


#data = np.load(proj_path + 'Datasets/Vancouver_AR2_TFI_MA1.npy') 
data = np.load(proj_path + 'Datasets/Bromma_AR2_TFI_MA2.npy') 
#data = np.loadtxt(proj_path + 'Datasets/SimARTFIMA11_short.txt')

warnings.filterwarnings("ignore", category=RuntimeWarning)

n = len(data)
q = 1
p = 0


n_params = q + p + 1
Last_ARMA = 1

   



#1.3 divide periodgram:I_PG into G shards
G = 1   # Number of groups
S = [int(item*(n-1)/(G)) + np.arange(0, int((n-1)/(G))) for item in range(G)]

def exact_log_likelihood_arma(data, params, q, p):
    phi = np.array(reparam(params[:q]))
    theta = np.array(reparam(params[q:q+p], MA = True))
    var = np.exp(params[-1])
    ans = sm.tsa.innovations.arma_loglike(data, phi, theta, sigma2=var)
    return ans

# 2.3 exact likelihood function
def log_likelihood_arma(data, params, q, p):
    """
    Autograd‐compatible ARMA(q,p) innovations log‐likelihood.

    data   : 1D array of observations, shape (T,)
    params : 1D array of length q + p + 1
             [unconstrained AR params, unconstrained MA params, log-variance]
    q, p   : AR and MA orders
    """
    # 1) unpack & reparametrise
    phi   = np.array(reparam(params[:q]))                  # shape (q,)
    theta = np.array(reparam(params[q : q+p], MA=True))    # shape (p,)
    sigma2 = np.exp(params[-1])                            # σ² > 0

    T = data.shape[0]
    sum_sq = np.array(0.0)             # accumulator for ∑ e_t²

    # 2) rolling buffer for last p residuals (initialised to zero)
    e_buf = np.zeros(p)

    # 3) for each t, compute e_t and accumulate its square
    for t in range(T):
        # AR term: φ₁ y_{t-1} + … + φ_q y_{t-q}
        if q > 0:
            lags = data[t - np.arange(1, q+1)]
            ar_term = np.sum(phi * lags)
        else:
            ar_term = 0.0

        # MA term: θ₁ e_{t-1} + … + θ_p e_{t-p}
        if p > 0:
            ma_term = np.sum(theta * e_buf)
        else:
            ma_term = 0.0

        # innovation
        e_t = data[t] - ar_term - ma_term
        sum_sq = sum_sq + e_t**2

        # update buffer (drop oldest, append e_t)
        if p > 0:
            e_buf = np.concatenate((e_buf[1:], np.array([e_t])))

    # 4) final log‐likelihood
    loglik = -0.5 * T * np.log(2 * np.pi * sigma2) \
             - sum_sq / (2 * sigma2)

    return loglik


# 2.4 prior distribution function(fractionated prior)
prior_mean_ginv_lambda_param = 0 
prior_std_ginv_lambda_param = 1

prior_mean_ginv_d_param = 1.8 
prior_std_ginv_d_param = .5

prior_mean_ginv_sigma2_param = 0 #np.log(np.var(x)) 
prior_std_ginv_sigma2_param = 1

def log_prior(theta):

         
    if not any(abs(theta[:-Last_ARMA]) > 1):
        prior_process_params = -len(theta[:-Last_ARMA]) * np.log(2)
    else:      
        prior_process_params = -np.inf
                

    prior_ginv_sigma2_param = sps_autograd.norm.logpdf(theta[-1], loc = prior_mean_ginv_sigma2_param, scale = prior_std_ginv_sigma2_param)
        
    return (prior_process_params + prior_ginv_sigma2_param)/G







# 2.5 posterior distribution function
#log_p = lambda x: log_prior(x, 0, 1, Last_ARMA) + np.sum(whittle_log_likelihood(x, q, p, I_pg_shard, TFI_term))

def log_p(x):
    return log_prior(x) + exact_log_likelihood_arma(data_shard, x, q, p) + log_likelihood_arma(data_shard, x, q, p)




# 2.6 partial autocorrelation transform function
def reparam(params, MA = False):
    """
    Transforms params to induce stationarity/invertability.
    Takes as input parameters in the partial auto-correlation parameterization and returns parameters that are on the ordinary parameterization.
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
def sampler(q, p, data, n_samples):

    params_init = paramsStar
    params_current = params_init
    posterior_samples = np.zeros((n_samples, n_params))
    log_p = np.zeros(n_samples)
    Acceptance = np.zeros((n_samples, 1))
    
    # Current log likelihood
    log_likelihood_current = exact_log_likelihood_arma(data, params_current, q, p)

    # Current log prior
    log_prior_current = log_prior(params_current)
    
    #Current log posterior
    log_p_current = np.sum(log_likelihood_current) + np.sum(log_prior_current)
   
    bar = progressbar.progressbar(range(n_samples))
    for i in bar:
        
        # New position:
        params_proposal = sps.multivariate_normal.rvs(mean = params_current, cov = proposal_width)
        if (np.abs(params_proposal[:-Last_ARMA]) < 1).all():
            
        
            log_likelihood_proposal = exact_log_likelihood_arma(data, params_proposal, q, p)

        
        else:
            log_likelihood_proposal = -np.inf
        # Proposal log prior          
        log_prior_proposal = log_prior(params_proposal)
        
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





# 2.9 simulate a parallel sampling(here we going is in a for loop, by using the group index:ind)

# MCMC lenth and Burn-in
n_samples = 15000
Burn_in = int(5000)

#Sampling
draws = []
log_pi = []
Acceptance = np.zeros((n_samples*G, 1))
w = []
thetaStar = []
cov = []

params = 0.1*np.ones(n_params)
params = params + sps.norm.rvs(0, 0.01, size = len(params))


#2.9.2 find own MAP(paramsStar), as start point
def obj(x): return -log_p(x)
r_logp, H_logp = grad(log_p), hessian(log_p)


lb = [-1]*len(params) # Constrains it to the stationary region after using the partial autocorrelation parameterisation.
ub = [1]*len(params)

    
lb[-1:] = [-30]
ub[-1:] = [30]

bnds = Bounds(lb, ub, keep_feasible=False)

    


bar = progressbar.progressbar(range(G))
for ind in bar:
    # 2.9.1 each worker gets its shard of data(periodgram)
    data_shard = data[S[ind]]

 
    res = minimize(obj, method = 'trust-constr', x0 = params, options={'gtol': 1e-4, 'maxiter':max_iter_optim}, bounds = bnds)

    #res = basinhopping(obj, x0 = params, niter=100, stepsize=1., callback=None, minimizer_kwargs={"method": "trust-constr", "jac":jcb, "hess":hs}, seed=15)
    paramsStar = res.x 
    #assert(res.success)
    sigma = np.linalg.inv(-H_logp(paramsStar))
    

    print('\nMAP%s' %(ind+1), np.round(paramsStar, 2))
    proposal_width = (2.38/np.sqrt(n_params))*sigma



    
    
    draw, log_pi_g, Acceptance = sampler(q, p, data_shard, n_samples)


    '''
    data_shard_jackknife = data_shard[::10]
    I_pg_shard_jackknife = I_pg_shard[::10]
    omega_shard = omega_full[I[ind]][::10]
    
    draw_jackknife, log_pi_g_jackknife, Acceptance_jackknife = sampler(q, p, data_shard_jackknife, I_pg_shard_jackknife, TFI_term, n_samples, exact=exact_L)
    
    bias_correction = (np.mean(draw_jackknife[:,0]) - np.mean(draw[:,0]))*(1/9)
        
    draw = draw - bias_correction
    '''
    
    draws.append(draw)
    log_pi.append(log_pi_g)
    thetaStar.append(paramsStar)
    cov.append(sigma)
    
    # 2.9.4 calculate sample variance(coveriance) of each shard, inverse them as the weights.
    #w.append(np.linalg.inv(sigma))
    w.append(np.linalg.inv(np.cov(draw, rowvar=False)))
    
    AcceptanceRate = np.mean(Acceptance)
    print('\nAcceptanceRate:', AcceptanceRate)

    
#################################################
# 3. Combine the draws across shards using weighted averages
#################################################

# 3.1 combine the draws: sum all weights -> inverse -> multiply each shard's weight and draws -> sum
posterior_samples = []
posterior_samples = np.dot(np.linalg.inv(sum(w)), sum(np.dot(w[ind], draws[ind].T) for ind in range(G))).T
    

posterior_samples_2 = []
posterior_samples_2 = np.dot(np.linalg.inv(sum(w)), sum(np.dot(w[ind], draws[ind].T) for ind in range(G))).T


# 3.2 transform partial autocorrelated params to ARMA params
phi = np.array(posterior_samples_2[:,:q], copy = True)
for i in range(n_samples-Burn_in):
    phi[i] = reparam(phi[i])


theta = np.array(posterior_samples_2[:,q:q+p], copy = True)
for i in range(n_samples-Burn_in):
    theta[i] = reparam(theta[i], MA = True)
    


'''


phi1 = phi[:,0]
#phi2 = phi[:,1]


theta1 = theta[:,0]
#theta2 = theta[:,1]
#theta3 = theta[:,2]

log_sigma2 = posterior_samples[:,-1]

d = posterior_samples[:,-1]
lambda_ = np.exp(posterior_samples[:,-3])



'''


for ind in range(G):
    sns.kdeplot(draws[ind][:,0])

for ind in range(G):
    sns.kdeplot(draws[ind][:,3])







