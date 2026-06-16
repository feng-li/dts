#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:09:03 2025

@author: zixuanwang
"""



import autograd.numpy as np
from autograd import grad, hessian, jacobian
from numpy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, Bounds
from scipy.special import logsumexp
import autograd.scipy.stats as sps_autograd
import progressbar
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow
from gpflow.mean_functions import MeanFunction
from IPython.display import display
import statsmodels.api as sm
from gpflow.utilities import print_summary, positive 
import sys, os, platform
import pandas as pd
import time


if platform.system() == 'Darwin':
    proj_path = '/Users/' + os.getenv("USER") + '/OneDrive - UTS/Project 1/'
elif platform.system() == 'Windows':
    proj_path = '/Users/' + os.getlogin() + '/OneDrive - UTS/Project 1/'
else:
    raise ValueError()      


path = 'results/draws/spark/'
dataset = 'Sim_AR1_TFI_MA1/'#'Vanc_ARMA12/'
#data = np.loadtxt(proj_path + 'Datasets/SimARTFIMA11_short.txt')
#data = np.load(proj_path + 'Datasets/Vancouver_AR2_TFI_MA1.npy') 
draws = np.load(proj_path + path + dataset + 'G8.npy')[:,::10]
logpi = np.load(proj_path + path + dataset + 'log_pi.npy')[:,::10]
#draws_G1 = np.load(proj_path + path + dataset + 'G1.npy')




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


class QuadraticMean(MeanFunction):
    """
    GPflow mean function
      m(x) = β0 + β1^T x + β2 * (x^T V_inv x),
    with β2 < 0.
    """

    def __init__(self, V_inv: np.ndarray):
        super().__init__()
        D = V_inv.shape[0]
        # store V^{-1}
        self.Vinv = tf.constant(V_inv, dtype=tf.float64)  # shape [D,D]

        # intercept β0
        self.beta0 = gpflow.Parameter(0.0, dtype=tf.float64)

        # linear term β1 (D×1)
        self.beta1 = gpflow.Parameter(
            np.zeros((D,1)), dtype=tf.float64
        )

        # raw positive parameter for |β2|, we'll negate it
        self.raw_b2 = gpflow.Parameter(
            1.0, transform=positive(), dtype=tf.float64
        )

    def __call__(self, X: tf.Tensor) -> tf.Tensor:
        """
        X: [N, D]  → returns [N, 1] vector m(X)
        """
        # linear + intercept
        lin = tf.matmul(X, self.beta1) + self.beta0        # [N,1]

        # quadratic form x^T V^{-1} x
        # (X @ V_inv) * X  does broadcast multiply, then sum over dim=1
        quad = tf.reduce_sum((X @ self.Vinv) * X, axis=1, keepdims=True)  # [N,1]

        # β2 = -raw_b2 ensures negativity
        beta2 = -self.raw_b2

        return lin + beta2 * quad


G = 8
n_params = 5#draws_G1.shape[1]

N = 10000
M = 1000

draws_for_consensus = np.load(proj_path + path + dataset + 'G8.npy')
mus   = np.array([draws_for_consensus[g].mean(axis=0)           for g in range(G)])        # (G, n_params)
covs  = np.array([np.cov(draws_for_consensus[g], rowvar=False)  for g in range(G)])        # (G, n_params, n_params)
precs = np.linalg.inv(covs)                                                    # (G, n_params, n_params)

# 2) aggregate into global mean & covariance
prec_sum         = np.sum(precs, axis=0)                                     # (n_params, n_params)
Sigma_M          = np.linalg.inv(prec_sum)                                   # (n_params, n_params)
weighted_sum     = np.sum([precs[g] @ mus[g] for g in range(G)], axis=0)     # (n_params,)
mu_M             = Sigma_M @ weighted_sum                                   # (n_params,)


weighted = sum(precs[g] @ draws_for_consensus[g].T for g in range(G))                  # (n_params, T)
consensus_samples = (Sigma_M @ weighted).T                                        # (T, n_params)
proposal_mean = consensus_samples.mean(0)
proposal_cov = np.cov(consensus_samples, rowvar=False)

#proposal distribution: t-distribution, df=5, mean=MAP, cov = inv-hess
theta_proposal = sps.multivariate_t.rvs(loc = proposal_mean, shape = proposal_cov, df = 5, size = N)#sps.multivariate_t.rvs(loc=paramsStar, shape=sigma, df=5, size=(N,1))
q_theta = sps.multivariate_t.pdf(theta_proposal, loc=proposal_mean, shape = proposal_cov, df=5)#sps.norm.pdf(theta_proposal, loc=np.mean(phi_MCMC), scale=np.var(phi_MCMC))


all_draws = np.vstack(draws)      # shape (G*J, d)
V = np.cov(all_draws, rowvar=False)    # (d,d)
V_inv = np.linalg.inv(V)




Mu = 0
Sigma = 0
bar = progressbar.progressbar(range(G))
for ind in bar:
    offset = np.mean(logpi[ind])
    X = draws[ind].reshape(-1,n_params)
    Y = logpi[ind].reshape(-1,1) - offset#logpi[ind] - offset#
    V = np.cov(X, rowvar=False)
    V_inv = np.linalg.inv(V)
    mean_fn = QuadraticMean(V_inv)

    #kernel = gpflow.kernels.SquaredExponential(lengthscales=np.ones(n_params))
    init_ls = np.std(X, axis=0) + 1e-6
    kernel = gpflow.kernels.SquaredExponential(lengthscales=init_ls)

    GP_model = gpflow.models.GPR(
                                data=(X, Y), 
                                kernel = kernel,
                                mean_function=mean_fn,
                                noise_variance=1e-5
                                )
    print(f"[GP training] shard {ind}: start optimise")

    optimiser = gpflow.optimizers.Scipy()
    optimise_hyper_params = optimiser.minimize(GP_model.training_loss, GP_model.trainable_variables)
    print(f"[GP training] shard {ind}: optimise done")

    print_summary(GP_model)
    mean_c, cov_c = GP_model.predict_f(theta_proposal, full_cov=True)
    
    mean_c = mean_c[:, 0]    # now shape (N,)
    cov_c  = cov_c[0, :, :]  # now shape (N,N)

    Mu += mean_c# + offset
    Sigma += cov_c


sys.exit()

# --- squeeze out any singleton dims ---
print("[STEP 1] squeeze Mu / Sigma ...", flush=True)
t1 = time.time()

Mu    = np.asarray(Mu)
if Mu.ndim == 2 and Mu.shape[1] == 1:
    Mu = Mu[:,0]                # now (N,)

Sigma = np.asarray(Sigma)
if Sigma.ndim == 3 and Sigma.shape[0] == 1:
    Sigma = Sigma[0]            # now (N,N)

print(f"[STEP 1] done in {time.time() - t1:.2f} s", flush=True)



# --- symmetrise + add jitter for PD ---
print("[STEP 2] build Sigma_pd ...", flush=True)
t2 = time.time()


#Sigma = 0.5*(Sigma + Sigma.T)
jitter = 1e-6 * np.mean(np.diag(Sigma))
Sigma_pd = Sigma + np.eye(N)*jitter
print(f"[STEP 2] done in {time.time() - t2:.2f} s", flush=True)


L = np.random.multivariate_normal(Mu, Sigma, size=M).T


# --- compute log-unnormalised weights ---
log_q = np.log(q_theta)            # (N,)
log_w_unnorm = L - log_q[:,None]    # (N, M)

# --- normalising constants per realisation ---
# logZ[m] = logsumexp(log_w_unnorm[:,m]) - log N
logZ = logsumexp(log_w_unnorm, axis=0) - np.log(N)  # (M,)

# --- final log-weights per θ_i ---
# log w[i] = logsumexp(log_w_unnorm[i,:] - logZ) - log M
log_w = logsumexp(log_w_unnorm - logZ[None,:], axis=1) - np.log(M)  # (N,)
w = np.exp(log_w)
w /= w.sum()  # normalise

# --- diagnostics & usage ---
ess = 1.0/np.sum(w**2)
print(f"ESS = {ess:.1f} / {N}")

# weighted posterior mean & covariance
post_mean = (theta_proposal * w[:,None]).sum(axis=0)
post_cov  = np.cov(theta_proposal.T, aweights=w)

# optional: draw an unweighted resample
idx = np.random.choice(N, size=N, p=w)
theta_resampled = theta_proposal[idx]
        
# Convert to a DataFrame
df = pd.DataFrame({
    'θ0': theta_proposal[:, 0],
    'w':  w
})

# Now call in long‐form
sns.kdeplot(data=df,
            x='θ0',
            weights='w',
            bw_adjust=1)

#sns.kdeplot(draws_G1[:,0])














