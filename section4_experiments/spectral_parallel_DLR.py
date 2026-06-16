
"""
Created on Thu May  9 12:20:54 2024

@author: zixuanwang

ARTIFMA(2,1)

This code samples DLR AR2 via spectral method, results are used to draw Table 1.
"""

import matplotlib, time, copy
import matplotlib.pyplot as plt
import autograd.numpy as np
from numpy.fft import fft
import autograd.scipy.stats as sps_autograd
from autograd import grad, hessian
from autograd.scipy.special import i0 as i0_autograd
import scipy.stats as sps
from scipy.optimize import minimize, Bounds, basinhopping
from scipy.linalg import toeplitz, solve_toeplitz, solve
from scipy.special import gammaln
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import pandas as pd
import sys, os, platform
import warnings
import progressbar
from operator import itemgetter 



gtol = 1e-4 
max_iter_optim = 500 
seed = 10
np.random.seed(seed)


if platform.system() == 'Darwin':
    proj_path = '/Users/' + os.getenv("USER") + '/OneDrive - UTS/Project 1/'
elif platform.system() == 'Windows':
    proj_path = '/Users/' + os.getlogin() + '/OneDrive - UTS/Project 1/'
else:
    raise ValueError()      

#data = np.load(proj_path + 'Datasets/DC-BATS_Ou.npy')
#data = np.load(proj_path + 'Datasets/DC-BATS_AR2_LM.npy')
#x_sim, y_sim, residual_sim = data[0], data[1], data[2]

x_sim = np.load('/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/results/draws/spark/Ou_LM/LM/X.npy')
y_sim = np.load('/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/results/draws/spark/Ou_LM/LM/y.npy')



# Simulation parameters
T = len(y_sim)  # Number of observations
n = int(np.floor((T-1)/2))


# Apply FFT to both exogenous variables and the dependent variable
x_hat = fft(x_sim.squeeze())   # FFT along the rows (across time)
y_hat = fft(y_sim)



# Dynamic Periodogram function
def I_pg_func(y_hat, x_hat, params):
    beta = params[2]#np.squeeze(params[-Last_ARMA: num_exog - Last_ARMA])
    return np.square(np.abs(y_hat - np.dot(x_hat, beta))) / (2 * np.pi * len(y_hat))

ind_full = np.arange(0, int(np.floor((len(y_sim)-1)/2)))



q = 2
p = 0
num_exog = 1#x_sim.ndim
TFI_term = False

if TFI_term:
   n_params = q + p + 3 + num_exog
   Last_ARMA = 3 + num_exog
   additional_names = ['log_lambda', 'log_sigma', 'd']
else:
   n_params = q + p + 1 + num_exog
   Last_ARMA = 1 + num_exog
   additional_names = ['log_sigma']



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
   


def f_ARTFIMA(id, phi, theta, var, d, lambda_):
    omega = (2*np.pi*id/T)
    
    FI_term = np.abs(1 - np.exp(-(lambda_ + 1j*omega)))**(-2*d)

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
                

    f = FI_term * (var/(2*np.pi)) * (np.real(vv1)**2 + np.imag(vv1)**2) \
                *(np.real(vv2)**2 + np.imag(vv2)**2)

    return f



def whittle_log_likelihood(params, y, x, q, p, TFI_term, I_pg=None, ind=None, return_sum=False):
    
    if TFI_term:
        # Tempered fractional difference
        d = params[-1]  # ARTFIMA model - d is unrestricted, may take any values (except integers, occurs only of sets with measure zero)
        lambda_ = np.exp(params[-3])
        var = np.exp(params[-2])
    else:
        var = np.exp(params[-1])
        d = 0  # ARMA process
        lambda_ = 0  # ARMA process

    if q > 0:
        phi = np.array(reparam(params[:q]))
    else:
        phi = np.array([])

    if p > 0:
        theta = np.array(reparam(params[q:q+p], MA=True))
    else:
        theta = np.array([])


    if ind is None:
        ind = np.arange(n)

    fj = f_ARTFIMA(np.array(ind), phi, theta, var, d, lambda_)
    
    if I_pg is None:
        I_pg = I_pg_func(y, x, params)

    if not return_sum:
        log_like = - (np.log(fj) + I_pg[ind] / fj)
    else:
        log_like = np.sum(-(np.log(fj) + I_pg[ind] / fj))

    return log_like





def log_prior(params, mu=0, sd=1):
    
    if (np.abs(params[:-Last_ARMA]) < 1).all():
        prior_process_params = -len(params[:-Last_ARMA])*np.log(2)
    else:      
        prior_process_params = -np.inf
    
    prior_beta_params = sps_autograd.norm.logpdf(params[-2], loc = mu, scale = sd)
    
    
    if TFI_term:
        prior_d_params = sps_autograd.norm.logpdf(params[-1], loc = mu, scale = sd)
        prior_lambda_params = sps_autograd.norm.logpdf(params[-3], loc = mu, scale = 100)
        prior_var_params = sps_autograd.norm.logpdf(params[-2], loc = mu, scale = sd)
    else:
        prior_d_params = 0
        prior_lambda_params = 0
        prior_var_params = sps_autograd.norm.logpdf(params[-1], loc = mu, scale = sd)  
        
    return (prior_process_params + prior_beta_params + prior_d_params + prior_lambda_params + prior_var_params)/G

log_p = lambda params: whittle_log_likelihood(params, y_hat, x_hat, q, p, TFI_term, ind=I[i], return_sum=True) + log_prior(params)



def sampler(theta_init, ind, proposal_width, n_samples):

    params_init = theta_init
    params_current = params_init
    posterior_samples = np.zeros((n_samples, n_params))
    log_pi = np.zeros(n_samples)
    Acceptance = np.zeros((n_samples, 1))
    


    # Current log prior
    #log_prior_current = log_prior(params_current)
    
    #Current log posterior
    log_p_current = whittle_log_likelihood(params_current, y_hat, x_hat, q, p, TFI_term, ind=ind, return_sum=True) + log_prior(params_current)
   
    bar = progressbar.progressbar(range(n_samples))
    for i in bar:
        
        # New position:
        params_proposal = sps.multivariate_normal.rvs(mean = params_current, cov = proposal_width)
        log_p_proposal = whittle_log_likelihood(params_proposal, y_hat, x_hat, q, p, TFI_term, ind=ind, return_sum=True) + log_prior(params_proposal)
        
        # Accept ratio
        alpha = np.min([1, np.exp(log_p_proposal - log_p_current)])
        accept = np.random.rand() < alpha
        
        if accept.any():
            # Update position
            params_current = params_proposal
            log_p_current = log_p_proposal            
            Acceptance[i] = 1
            
            
        posterior_samples[i, :] =  params_current
        log_pi[i] = log_p_current
    return posterior_samples[Burn_in:], log_pi[Burn_in:].T, Acceptance








G = 16  # Number of groups
I = [item + np.arange(0, int(np.floor((T-1)/2)), G) for item in range(G)]



n_samples = 6000
Burn_in = int(1000)
params = 0.1*np.ones(n_params)


def obj(prm): return -log_p(prm)
jcb = grad(obj)
r_logp, H_logp = grad(log_p), hessian(log_p)
hs = hessian(obj)

lb = [-5]*len(params) # Constrains it to the stationary region after using the partial autocorrelation parameterisation.
ub = [5]*len(params)

    
lb[:2] = [-1, -1]
ub[:2] = [1, 1]

bnds = Bounds(lb, ub, keep_feasible=True)

#Sampling
draws = []
log_pi = []
Acceptance = np.zeros((n_samples*G, 1))
w = []
thetaStar = []
cov = []


bar = progressbar.progressbar(range(G))
for i in bar:


    res = minimize(obj, 
                   jac = jcb, 
                   method = 'trust-constr', 
                   x0 = params, 
                   hess = hs,
                   bounds = bnds,
                   options={'gtol': 1e-4, 'maxiter':max_iter_optim}
                   )
    print('\nMAP%s' %(i+1), np.round(res.x, 2))
    
    paramsStar = res.x 
    #assert(res.success)
    sigma = np.linalg.inv(-H_logp(paramsStar))
    proposal_width = (2.38/np.sqrt(n_params))*sigma
    draw, log_pi_g, Acceptance = sampler(paramsStar, I[i], proposal_width, n_samples)
    
    draws.append(draw)
    log_pi.append(log_pi_g)
    thetaStar.append(paramsStar)
    cov.append(sigma)
    
    # 2.9.4 calculate sample variance(coveriance) of each shard, inverse them as the weights.
    #w.append(np.linalg.inv(sigma))
    w.append(np.linalg.inv(np.cov(draw, rowvar=False)))
    
    AcceptanceRate = np.mean(Acceptance)
    print('\nAcceptanceRate:', AcceptanceRate)



posterior_samples = []
posterior_samples = np.dot(np.linalg.inv(sum(w)), sum(np.dot(w[ind], draws[ind].T) for ind in range(G))).T


# 3.2 transform partial autocorrelated params to ARMA params
phi = np.array(posterior_samples[:,:2], copy = True)
for i in range(n_samples-Burn_in):
    phi[i] = reparam(phi[i])

    


sys.exit()



beta_G1 = posterior_samples[:,-2]
phi1_G1 = phi[:,0]
phi2_G1 = phi[:,1]
sigma2_G1 = np.exp(posterior_samples[:,-1])


beta_G16 = posterior_samples[:,-2]
phi1_G16 = phi[:,0]
phi2_G16 = phi[:,1]
sigma2_G16 = np.exp(posterior_samples[:,-1])



beta_G20 = posterior_samples[:,-2]
phi1_G20 = phi[:,0]
phi2_G20 = phi[:,1]
sigma2_G20 = np.exp(posterior_samples[:,-1])




sns.kdeplot(beta_G1)
sns.kdeplot(beta_G16)


sns.kdeplot(phi1_G1)
sns.kdeplot(phi1_G16)

sns.kdeplot(phi2_G1)
sns.kdeplot(phi2_G16)


sns.kdeplot(sigma2_G1)
sns.kdeplot(sigma2_G16)














































