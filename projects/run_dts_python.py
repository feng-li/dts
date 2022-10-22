"""
DistributedTimeSeries

    Spectral parallel MCMC

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


"""
import os

import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
from numpy.fft import fft
import scipy.stats as sps
from autograd import grad, hessian, jacobian
from scipy.optimize import minimize, Bounds
from tqdm import tqdm  # progressbar

gtol = 1e-4
max_iter_optim = 500
np.random.seed(123)

#################################################
# 1. Divide data y into S shards y_1...y_s.
#################################################

# 1.1 define data and model

project_path = "~/code/dts/projects"

# temporary setting to allow for importing `dts` from parent directory.
import sys, pathlib
wrkDir = pathlib.Path(project_path).expanduser()
codeDir = str(wrkDir.parent)
sys.path.insert(1, codeDir)

from dts.mcmc import *

# data = np.load(proj_path + 'Datasets/Vancouver_AR2_TFI_MA1.npy')
data = np.loadtxt(os.path.expanduser(project_path) + '/../dts/data/SimARTFIMA11.txt')

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

# 2.5 posterior distribution function
#2.9.2 find own MAP(paramsStar), as start point
log_p = lambda x: log_prior(x, 0, 1, Last_ARMA, TFI_term) / G + np.sum(whittle_log_likelihood(x, q, p, I_pg_shard, TFI_term, omega_shard))
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

for ind in tqdm(range(G)):
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
    draw, log_pi_g, Acceptance = sampler(q, p, data_shard, I_pg_shard, TFI_term, omega_shard, n_samples, paramsStar, proposal_width, Burn_in, exact=exact_L)


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
