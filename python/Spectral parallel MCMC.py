# -*- coding: utf-8 -*-
"""
DistributedTimeSeries

    Spectral parallel MCMC

@author: Zi

Last edited: 22/9/2022
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
from scipy.optimize import minimize, Bounds
import autograd.scipy.stats as sps_autograd
import progressbar
import pandas as pd
import pickle
import sys, os, platform

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
np.random.seed(123)

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
data = np.loadtxt(proj_path + 'Datasets/SimARTFIMA11_short.txt')



n = len(data)
q = 1
p = 1
TFI_term = False
exact_L = True

if TFI_term:
   n_params = q + p + 3
   Last_ARMA = 3
else:
   n_params = q + p + 1
   Last_ARMA = 1

   

# 1.2 the acutal data is periodgram of the original dataset
def p_gram(x):  # Construct Periodogram
    id = int(np.floor((len(x)-1)/2))
    return np.square(np.abs(x[0:(id)]))/(2 * np.pi * len(x))
I_pg_full = p_gram(fft(data))   



#1.3 divide periodgram:I_PG into G shards
G = 10   # Number of groups
I = [item + np.arange(0, int(np.floor((n-1)/2)), G) for item in range(G)]
S = [int(item*(n-1)/(G)) + np.arange(0, int((n-1)/(G))) for item in range(G)]
k = np.arange(1,int((n/G)/2)+1) # k is used to determine Fourier frequencies and sum of Whittle likelihood

#################################################
# 2. Run G separate Monte Carlo algorithms to sample posterior, with each shard using the fractionated prior
#################################################

# 2.1 spectral density function
def f_ARTFIMA(omega, phi, theta, var, d, lambda_): 
    

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



# 2.2 Whittle likelihood function
def whittle_log_likelihood(params, q, p, I_pg, TFI_term): #do ARTFIMA lambda and d in SD
    '''
    q: lag of AR
    p: lag of MA
    I_pg: periodogram of data
    '''        
    if TFI_term: #ARTFIMA model
        d = params[-1]
        lambda_ = np.exp(params[-3])
        var = np.exp(params[-2])
    else: #ARMA model
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



# 2.3 exact likelihood function
def exact_log_likelihood_arma(data, params, q, p):
    phi = np.array(reparam(params[:q]))
    theta = np.array(reparam(params[q:q+p], MA = True))
    var = np.exp(params[-1])
    ans = sm.tsa.innovations.arma_loglike(data, phi, theta, sigma2=var)
    return ans



# 2.4 prior distribution function(fractionated prior)
def log_prior(params, mu, sd, Last_ARMA):
    
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
        
    return (prior_process_params + prior_d_params + prior_lambda_params + prior_var_params)/G



# 2.5 posterior distribution function
log_p = lambda x: log_prior(x, 0, 1, Last_ARMA) + np.sum(whittle_log_likelihood(x, q, p, I_pg_shard, TFI_term))






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
def sampler(q, p, data, I_pg, TFI_term, n_samples, params_prior_mu=0, params_prior_sd=1., exact = False):

    params_init = paramsStar
    params_current = params_init
    posterior_samples = np.zeros((n_samples, n_params))
    log_p = np.zeros(n_samples)
    Acceptance = np.zeros((n_samples, 1))
    
    # Current log likelihood
    if exact:
        log_likelihood_current = exact_log_likelihood_arma(data, params_current, q, p)
    else:  
        log_likelihood_current = whittle_log_likelihood(params_current, q, p, I_pg, TFI_term)

    # Current log prior
    log_prior_current = log_prior(params_current, params_prior_mu, params_prior_sd, Last_ARMA)
    
    #Current log posterior
    log_p_current = np.sum(log_likelihood_current) + np.sum(log_prior_current)
   
    bar = progressbar.progressbar(range(n_samples))
    for i in bar:
        
        # New position:
        params_proposal = sps.multivariate_normal.rvs(mean = params_current, cov = proposal_width)
        if (np.abs(params_proposal[:-Last_ARMA]) < 1).all():
            
        # Proposal log likelihood
            if exact:
                log_likelihood_proposal = exact_log_likelihood_arma(data, params_proposal, q, p)
            else:    
                log_likelihood_proposal = whittle_log_likelihood(params_proposal, q, p, I_pg, TFI_term) 
        
        else:
            log_likelihood_proposal = -np.inf
        # Proposal log prior          
        log_prior_proposal = log_prior(params_proposal, params_prior_mu, params_prior_sd, Last_ARMA)
        
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

params = 0.01*np.ones(n_params)
params = params + sps.norm.rvs(0, 0.1, size = len(params))


omega_full = 2*np.pi*np.arange(1,int(n/2)+1)/len(data)

#2.9.2 find own MAP(paramsStar), as start point
def obj(params): return -log_p(params)
jcb = grad(obj)
r_logp, H_logp = grad(log_p), hessian(log_p)
hs = hessian(obj)

lb = [-1]*len(params) # Constrains it to the stationary region after using the partial autocorrelation parameterisation.
ub = [1]*len(params)

    
if TFI_term:   
    lb[-3:] = [-30, -30, -30]
    ub[-3:] = [30, 30, 30]
else:
    lb[-1:] = [-30]
    ub[-1:] = [30]

bnds = Bounds(lb, ub, keep_feasible=True)

    


bar = progressbar.progressbar(range(G))
for ind in bar:
    # 2.9.1 each worker gets its shard of data(periodgram)
    data_shard = data[S[ind]]
    I_pg_shard = I_pg_full[I[ind]]
    omega_shard = omega_full[I[ind]]
    if exact_L:
        def p_gram_shard(x):
            id = int(np.floor((len(x))/2))
            return np.square(np.abs(x[0:(id)]))/(2 * np.pi * len(x))
        if G > 1:
            I_pg_shard = p_gram_shard(fft(data_shard))   
            omega_shard = 2*np.pi*np.arange(1,int(len(data_shard)/2)+1)/len(data_shard)
 
    res = minimize(obj, bounds=bnds, jac = jcb, hess = hs, method = 'trust-constr', x0 = params)
    paramsStar = res.x 
    assert(res.success)
    sigma = np.linalg.inv(-H_logp(paramsStar))
    
    print('\nMAP%s' %(ind+1), paramsStar)
    proposal_width = (2.38/np.sqrt(n_params))*sigma



    
    
    # 2.9.3 run sampling, jackknife bias correction and restore all results as draws    
    draw, log_pi_g, Acceptance = sampler(q, p, data_shard, I_pg_shard, TFI_term, n_samples, exact=exact_L)


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
    

# 3.2 transform partial autocorrelated params to ARMA params
phi = np.array(posterior_samples[:,:q], copy = True)
for i in range(n_samples-Burn_in):
    phi[i] = reparam(phi[i])


theta = np.array(posterior_samples[:,q:q+p], copy = True)
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

