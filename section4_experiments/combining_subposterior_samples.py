# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 02:27:20 2021

@author: Zixuan
"""



import statsmodels.api as sm
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import numpy as np
import numpy as onp
from jax import grad, hessian, jacobian
from numpy.fft import fft
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sps
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, Bounds
import jax.scipy.stats as sps_jax
import progressbar
import pandas as pd
import sys, os, platform
from sklearn.neighbors import KernelDensity
from rpy2 import robjects
from rpy2.robjects import r, numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# 1) Activate automatic conversion between NumPy and R
#numpy2ri.activate()



if platform.system() == 'Darwin':
    proj_path = '/Users/' + os.getenv("USER") + '/OneDrive - UTS/Project 1/'
elif platform.system() == 'Windows':
    proj_path = '/Users/' + os.getlogin() + '/OneDrive - UTS/Project 1/'
else:
    raise ValueError()      

r_path   = os.path.join(proj_path, 'weierstrass.R')
print(f"Sourcing R file from: {r_path}")
_ = robjects.r['source'](r_path)   # load weierstrass() & repmat() into R

# 3) Grab the R functions into Python
weierstrass = robjects.globalenv['weierstrass']
repmat      = robjects.globalenv['repmat']


path = 'results/draws/spark/'
dataset = 'Vanc_ARMA12/'#'Sim_AR1_TFI_MA1_long/'#'Bromma/'#


draws = onp.load(proj_path + path + dataset + 'G8.npy')
draws_G1 = onp.load(proj_path + path + dataset + 'G1.npy')


q = 1
p = 2
n_samples = draws.shape[1]
G = 8


TFI_term = False

if TFI_term:
   n_params = q + p + 3
   Last_ARMA = 3
else:
   n_params = q + p + 1
   Last_ARMA = 1



onp.random.seed(10)

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


def combination(method):
    """
    Combine G subposterior chains via one of three schemes:
      - "mcmc":       return up to n_samples from machine 1’s raw draws
      - "consensus":  return up to n_samples of the weighted‐average draws
      - "parametric": draw n_samples from the Gaussian approximation

    Parameters
    ----------
    draws : ndarray, shape (G, T, n_params)
        MCMC draws from each of G machines.
    method : {"mcmc","consensus","parametric"}
        Which combination strategy to use.
    n_samples : int, optional
        How many draws to return.  Defaults to T (the chain length).

    Returns
    -------
    posterior_samples : ndarray
        Combined draws of shape (n_samples, n_params).
    acceptance : list
        Empty (no MH steps here).
    """
    G, n_samples, n_params = draws.shape

    # helper #1: average theta over machines at indices t_dot
    def theta_bar(idx):
        return np.mean([draws[m, idx[m]] for m in range(G)], axis=0)

    # helper #2: sum of log‐kernel densities over machines
    def log_w_t(t_dot, θ_bar, h2I):
        total = 0.0
        for m in range(G):
            total += sps.multivariate_normal.logpdf(
                         draws[m, t_dot[m]],
                         mean=θ_bar,
                         cov=h2I
                     )
        return total
    
    # 1) subposterior means & precision
    mus   = np.array([draws[g].mean(axis=0)           for g in range(G)])        # (G, n_params)
    covs  = np.array([np.cov(draws[g], rowvar=False)  for g in range(G)])        # (G, n_params, n_params)
    precs = np.linalg.inv(covs)                                                    # (G, n_params, n_params)

    # 2) aggregate into global mean & covariance
    prec_sum         = np.sum(precs, axis=0)                                     # (n_params, n_params)
    Sigma_M          = np.linalg.inv(prec_sum)                                   # (n_params, n_params)
    weighted_sum     = np.sum([precs[g] @ mus[g] for g in range(G)], axis=0)     # (n_params,)
    mu_M             = Sigma_M @ weighted_sum                                   # (n_params,)

    # 3) dispatch
    if method == "MCMC":
        # raw draws from chain 1
        return draws_G1
    
    elif method == "average":
        posterior_samples = np.mean(draws, axis=0)
        return posterior_samples
    
    elif method == "consensus":
        # weighted average of each machine's draws
        weighted = sum(precs[g] @ draws[g].T for g in range(G))                  # (n_params, T)
        combined = (Sigma_M @ weighted).T                                        # (T, n_params)
        return combined[:n_samples]

    elif method == "parametric":
        posterior_samples = onp.random.multivariate_normal(
            mean=mu_M,
            cov=Sigma_M,
            size=n_samples
        )
        return posterior_samples
    
    elif method == "non-parametric":
        T = n_samples
        # 1) bandwidth (fixed)
        h = T ** (-1.0 / (4 + n_params))
        h2 = h**2

        # 2) fit one KDE per machine
        kdes = [
            KernelDensity(kernel="gaussian", bandwidth=h)
              .fit(draws[g])
            for g in range(G)
        ]

        # 3) initialize index vector, storage
        t_dot             = onp.random.randint(0, T, size=G)
        posterior_samples = np.zeros((n_samples, n_params))
        acceptance        = np.zeros((G, n_samples), dtype=int)
        bar               = progressbar.ProgressBar(max_value=n_samples)

        # 4) IMG sampler
        for i in bar(range(n_samples)):
            for m in range(G):
                # propose new index for subposterior m
                c_dot = t_dot.copy()
                c_dot[m] = onp.random.randint(0, T)

                # compute the "global" mean under old & new indices
                θ_old = np.mean([draws[j, t_dot[j]] for j in range(G)], axis=0)
                θ_new = np.mean([draws[j, c_dot[j]] for j in range(G)], axis=0)

                # log-density of the product of subposteriors
                log_r_old = sum(
                    kde.score_samples(θ_old.reshape(1, -1))[0] for kde in kdes
                )
                log_r_new = sum(
                    kde.score_samples(θ_new.reshape(1, -1))[0] for kde in kdes
                )

                # acceptance probability
                alpha = np.exp(log_r_new - log_r_old)
                if onp.random.rand() < alpha:
                    t_dot[m]         = c_dot[m]
                    acceptance[m, i] = 1

            # 5) draw θ_i from N(θ̄, h²/G · I)
            θ_bar = np.mean([draws[j, t_dot[j]] for j in range(G)], axis=0)
            posterior_samples[i] = sps.multivariate_normal.rvs(
                mean=θ_bar,
                cov=(h2 / G) * np.eye(n_params)
            )

        return posterior_samples

    elif method == "semi-parametric":
        # subposterior chain length
        T = draws.shape[1]

        # 1) fixed bandwidth h
        h   = T**(-1.0 / (4 + n_params))
        h2I = (h**2) * np.eye(n_params)

        # 2) initialize index vector
        t_dot = onp.random.randint(0, T, size=G)

        # 3) storage
        posterior_samples = np.zeros((n_samples, n_params))
        acceptance        = np.zeros((G, n_samples), dtype=int)
        bar               = progressbar.ProgressBar(max_value=n_samples)



        
        def log_r(idx, θbar):
            return sum(
                sps.multivariate_normal.logpdf(
                    draws[m, idx[m]],
                    mean=θbar,
                    cov=h2I
                )
                for m in range(G)
            )

        # 4) run IMG sampler with semiparametric weights
        for i in bar(range(n_samples)):
            for m in range(G):
                # propose a new index for machine m
                c_dot = t_dot.copy()
                c_dot[m] = onp.random.randint(0, T)

                # old/new global means
                θ_old = theta_bar(t_dot)
                θ_new = theta_bar(c_dot)

                # nonparametric part
                log_r_old = log_r(t_dot, θ_old)
                log_r_new = log_r(c_dot, θ_new)

                # parametric part f(θ) = N(θ | μ_M, Σ_M)
                log_f_old = sps.multivariate_normal.logpdf(θ_old, mean=mu_M, cov=Sigma_M)
                log_f_new = sps.multivariate_normal.logpdf(θ_new, mean=mu_M, cov=Sigma_M)

                # full weight ratio
                log_W_old = log_r_old + log_f_old
                log_W_new = log_r_new + log_f_new
                alpha     = np.exp(log_W_new - log_W_old)

                # accept/reject
                if onp.random.rand() < alpha:
                    t_dot[m]        = c_dot[m]
                    acceptance[m,i] = 1

            # 5) draw θ_i from N(μ_t, Σ_t), where
            #    Σ_t = (Σ_M^{-1} + G/h²·I)^{-1},   μ_t = Σ_t (Σ_M^{-1} μ_M + G/h²·θ̄)
            Sigma_t = np.linalg.inv(prec_sum + (G / h**2) * np.eye(n_params))
            mu_t    = Sigma_t @ (prec_sum @ mu_M + (G / h**2) * theta_bar(t_dot))

            posterior_samples[i] = sps.multivariate_normal.rvs(mean=mu_t, cov=Sigma_t)
        return posterior_samples
    
    elif method == "Weierstrass":
        # 5) Turn the (10 × 10000 × 5) array into a Python list of 10 matrices
        py_list = [draws[i] for i in range(draws.shape[0])]

        # 6) Convert that list into an R list of 10000×5 matrices
        with localconverter(default_converter + numpy2ri.converter):
            r_samples = robjects.conversion.py2rpy(py_list)

        # 7) Call weierstrass (leave num_sets unset so it stays NULL in R)
        combined_r = weierstrass(
            r_samples,
            para_dim=draws.shape[2]
            )

        # 8) Convert the result back to a NumPy array
        return np.array(combined_r)       
    
    elif method == "GP-IS":
        gp_is_w = onp.load(proj_path + 'results/draws/spark/' + dataset + 'gp_is_w.npy')
        weighted = sum(precs[g] @ draws[g].T for g in range(G))                 
        consensus_samples = (Sigma_M @ weighted).T  
        
        proposal_mean = consensus_samples.mean(0)
        proposal_cov = np.cov(consensus_samples, rowvar=False)

        theta_proposal = sps.multivariate_t.rvs(
            loc = proposal_mean, 
            shape = proposal_cov, 
            df = 5, 
            size = gp_is_w.shape[0]
        )

        idx = onp.random.choice(
            len(gp_is_w),
            size=len(gp_is_w),
            replace=True,
            p=gp_is_w
        )


        # 8) Convert the result back to a NumPy array
        posterior_samples = theta_proposal[idx]
    
        return posterior_samples





ar_names = ['phi'] if q == 1 else [f'phi{i}' for i in range(1, q+1)]
ma_names = ['theta'] if p == 1 else [f'theta{i}' for i in range(1, p+1)]
additional_names = ['log_lambda', 'log_sigma', 'd'] if TFI_term else ['log_sigma']
param_names = ar_names + ma_names + additional_names

# --------------------------------

methods = ["MCMC", 
           #"average",
           "consensus", 
           "parametric", 
           #"non-parametric", 
           "semi-parametric",
           "Weierstrass",
           "GP-IS"
           ]

# Build one DataFrame per method, using your param_names
df_dict = {}
for m in methods:
    samples = combination(m)   # your combine func
    # samples.shape == (n_samples, n_params)
    df = pd.DataFrame(samples, columns=param_names)
    # set up a MultiIndex so columns are (method, param_name)
    df.columns = pd.MultiIndex.from_product(
        [[m], param_names],
        names=["method", "param"]
    )
    df_dict[m] = df

# concatenate side-by-side
df_all = pd.concat(df_dict.values(), axis=1)

sys.exit()








# now pickle
#df_all.to_pickle("posterior_samples.pkl")

# Updated labels
latex_labels = {
    'phi':        r'$\phi$',
    'phi1':       r'$\phi_1$',
    'phi2':       r'$\phi_2$',
    'theta':      r'$\vartheta$',
    'theta1':     r'$\vartheta_1$',
    'theta2':     r'$\vartheta_2$',
    'log_lambda': r'$\lambda$',
    'log_sigma':  r'$\sigma^2$',
    'd':          r'$d$'
}

# Select parameters: first two (AR), one MA, last one (log_sigma)
params = param_names[:3] + param_names[-1:]

fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2))

for ax, param in zip(axes.flatten(), params):
    idx = param_names.index(param)

    is_ar_or_ma = param.startswith('phi') or param.startswith('theta')
    is_log_transform = param in ['log_lambda', 'log_sigma']

    for g in range(G):
        raw = draws[g, :, idx]

        # Apply reparam only if >1 AR or MA param
        if param.startswith('phi') and q > 1:
            all_params = draws[g, :, :q + p]
            raw = np.apply_along_axis(reparam, 1, all_params, MA=False)[:, idx]
        elif param.startswith('theta') and p > 1:
            all_params = draws[g, :, :q + p]
            raw = np.apply_along_axis(reparam, 1, all_params, MA=True)[:, idx]

        # Transform log-params to exp scale
        if is_log_transform:
            raw = np.exp(raw)

        sns.kdeplot(
            raw,
            ax=ax,
            linestyle='--',
            linewidth=1,
            alpha=0.5,
            label=f"machine {g+1}",
            bw_adjust=1.5
        )

    # Full data line
    raw_full = draws_G1[:, idx]

    if param.startswith('phi') and q > 1:
        all_params = draws_G1[:, :q + p]
        raw_full = np.apply_along_axis(reparam, 1, all_params, MA=False)[:, idx]
    elif param.startswith('theta') and p > 1:
        all_params = draws_G1[:, :q + p]
        raw_full = np.apply_along_axis(reparam, 1, all_params, MA=True)[:, idx]

    if is_log_transform:
        raw_full = np.exp(raw_full)

    sns.kdeplot(
        raw_full,
        ax=ax,
        color='darkblue',
        linestyle='-',
        linewidth=2,
        label='full data',
        bw_adjust=1.5
    )

    ax.set_xlabel(latex_labels.get(param, param), fontsize=10)
    ax.set_ylabel(rf"$\pi\left({latex_labels[param].strip('$')}\right)$", fontsize=10)

    # --- aesthetics: remove box, keep x-axis only, remove y tick numbers ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.tick_params(axis='y', left=False, labelleft=False)
    ax.tick_params(axis='x', labelsize=9)

# Shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.02),
    ncol=5,
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.show()
fig.savefig(
    "machine_kde.png",
    dpi=600,
    bbox_inches="tight"
)




xlims_dict = {}
for ax, param in zip(axes.flatten(), params):
    xlims_dict[param] = ax.get_xlim()

ylims_dict = {}
for ax, param in zip(axes.flatten(), params):
    ylims_dict[param] = ax.get_ylim()

# Define method colors (only MCMC needs to match plot 1 for now)
method_colors = {
    'MCMC': 'darkblue',  # match full-data line in Plot 1
    'average': None,
    'consensus': None,
    'parametric': None,
    'semi-parametric': '#6A5ACD',  # no green
    'weierstrass': None,
    'GP-IS': None
}

# Second plot: KDEs of combined draws (flatter aspect ratio for paper)
fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2))

for ax, param in zip(axes.flatten(), params):
    is_log_transform = param in ['log_lambda', 'log_sigma']

    for method in methods:
        values = df_all[method][param].values

        # Transform log params
        if is_log_transform:
            values = np.exp(values)

        sns.kdeplot(
            values,
            ax=ax,
            label=method,
            bw_adjust=1.0,
            color=method_colors.get(method, None)  # apply color if defined
        )

    ax.set_xlabel(latex_labels.get(param, param), fontsize=10)
    ax.set_ylabel(rf"$\pi\left({latex_labels[param].strip('$')}\right)$", fontsize=10)
    ax.set_xlim(xlims_dict[param])  # sync x-axis
    # ax.set_ylim(ylims_dict[param])  # sync y-axis

    # --- aesthetics: remove box, keep x-axis only, remove y tick numbers (keep y-label) ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)        # remove left spine (no frame)
    ax.tick_params(axis='y', left=False, labelleft=False)  # remove y ticks and labels
    ax.tick_params(axis='x', labelsize=9)       # keep x ticks

# Shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='upper center',
    ncol=5,  # len(methods) if you want
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

# Optional: align y-labels on left column for a cleaner look
# fig.align_ylabels(axes[:, 0])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
fig.savefig(
    "combined_kde.png",
    dpi=600,
    bbox_inches="tight"
)






