#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 13:11:06 2026

@author: zixuanwang

This code run MCMC on ARTFIMA(1, 0) multiple times to compare with Ou et al, results are used to draw Table 2.
"""

import os
import pickle
import warnings
import sys
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp
import jax.scipy.stats as sps_jax
from jax import grad, hessian

import scipy.stats as sps
from numpy.fft import fft
from scipy.optimize import basinhopping

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# USER SETTINGS
# ============================================================
CASE = "case2"   # "case1" or "case2"
G = 20            # manually set: 1 / 5 / 10 / 20
N_REP = 20

DATA_DIR = "/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/DC-BATS/simulated_data_ou_style/"

# MCMC
N_SAMPLES = 6000
BURN_IN = 1000

# Optimizer
GTOL = 1e-4
MAX_ITER_OPTIM = 500

# Model orders
q = 1
p = 0
TFI_term = True
exact_L = False  # keep false here

# Priors (your current choices)
prior_mean_ginv_lambda_param = 0.0
prior_std_ginv_lambda_param = 1.0
prior_mean_ginv_d_param = 0.0
prior_std_ginv_d_param = 1.0
prior_mean_ginv_sigma2_param = 0.0
prior_std_ginv_sigma2_param = 1.0

# ============================================================
# HELPERS
# ============================================================

def p_gram(x_fft):
    """Periodogram from FFT output (matches your definition)."""
    id_ = int(np.floor((len(x_fft) - 1) / 2))
    return np.square(np.abs(x_fft[0:id_])) / (2 * np.pi * len(x_fft))

def reparam(params, MA=False):
    """
    Partial autocorrelation -> AR/MA params transform (your original).
    params: 1d array
    """
    newparams = np.array(params, copy=True)
    tmp = np.array(params, copy=True)

    for j in range(1, len(params)):
        if not MA:
            tmp_new = tmp[:j] - np.array([(newparams[j] * newparams[j - k - 1]) for k in range(j)])
        else:
            tmp_new = tmp[:j] + np.array([(newparams[j] * newparams[j - k - 1]) for k in range(j)])

        tmp = np.hstack([tmp_new, newparams[j:]])
        newparams = np.hstack((tmp[:j], newparams[j:]))

    return newparams

def f_ARTFIMA(omega, phi, theta, var, d, lambda_):
    """Spectral density (your original)."""
    TFI = np.abs(1 - np.exp(-(lambda_ + 1j * omega))) ** (-2 * d)

    if phi.size > 0:
        log_arg_phi = np.outer(-1j * omega, np.arange(1, len(phi) + 1))
        vv1 = 1 / (1 - np.sum(phi * np.exp(log_arg_phi), 1))
    else:
        vv1 = 1

    if theta.size > 0:
        log_arg_theta = np.outer(-1j * omega, np.arange(1, len(theta) + 1))
        vv2 = (1 + np.sum(theta * np.exp(log_arg_theta), 1))
    else:
        vv2 = 1

    return TFI * (var / (2 * np.pi)) * (np.real(vv1) ** 2 + np.imag(vv1) ** 2) * (np.real(vv2) ** 2 + np.imag(vv2) ** 2)

def whittle_log_likelihood(params, I_pg, omega_shard, q, p, TFI_term):
    """
    Returns vector log-likelihood contributions (your original style).
    params are in PACF parametrization for AR/MA + transformed lambda/var/d.
    """
    if TFI_term:
        d = params[-1]
        lambda_ = 0.1#np.exp(params[-3])
        var = 1.0#np.exp(params[-2])
    else:
        d = 0.0
        lambda_ = 0.0
        var = np.exp(params[-1])

    if q > 0:
        phi = np.array(reparam(params[:q]))
    else:
        phi = np.array([])

    if p > 0:
        theta = np.array(reparam(params[q:q+p], MA=True))
    else:
        theta = np.array([])

    fj = f_ARTFIMA(omega_shard, phi, theta, var, d, lambda_)
    return -(np.log(fj) + I_pg / fj)

def log_prior(theta, G, Last_ARMA, TFI_term):
    """Your prior, divided by G."""
    # uniform prior on process params in [-1,1]
    prior_process_params = np.where(
        np.all(np.abs(theta[:-Last_ARMA]) <= 1.0),
        -len(theta[:-Last_ARMA]) * np.log(2),
        -np.inf,
    )

    if TFI_term:
        prior_ginv_lambda_param = sps_jax.norm.logpdf(theta[-3], loc=prior_mean_ginv_lambda_param, scale=prior_std_ginv_lambda_param)
        prior_ginv_d_param = sps_jax.norm.logpdf(theta[-1], loc=prior_mean_ginv_d_param, scale=prior_std_ginv_d_param)
        prior_ginv_sigma2_param = sps_jax.norm.logpdf(theta[-2], loc=prior_mean_ginv_sigma2_param, scale=prior_std_ginv_sigma2_param)
    else:
        prior_ginv_d_param = 0.0
        prior_ginv_lambda_param = 0.0
        prior_ginv_sigma2_param = sps_jax.norm.logpdf(theta[-1], loc=prior_mean_ginv_sigma2_param, scale=prior_std_ginv_sigma2_param)

    return (prior_process_params + prior_ginv_d_param + prior_ginv_sigma2_param + prior_ginv_lambda_param) / G

def make_log_p(I_pg_shard, omega_shard, q, p, TFI_term, G, Last_ARMA):
    """Closure so grad/hess are correct per shard."""
    def _log_p(x):
        ll = np.sum(whittle_log_likelihood(x, I_pg_shard, omega_shard, q, p, TFI_term))
        return ll + log_prior(x, G, Last_ARMA, TFI_term)
    return _log_p

def sampler(paramsStar, proposal_width, log_p_fn, n_samples, burn_in, Last_ARMA):
    """
    Random-walk MH with MVN proposal, exactly your style but simplified.
    Returns draws after burn-in (shape: (n_samples-burn_in, n_params)).
    """
    n_params = len(paramsStar)
    cur = onp.array(paramsStar, copy=True)
    lcur = log_p_fn(cur)

    out = onp.zeros((n_samples, n_params))
    acc = 0

    for i in range(n_samples):
        prop = sps.multivariate_normal.rvs(mean=cur, cov=proposal_width)

        # enforce PACF bounds for AR/MA parameters only
        if onp.all(onp.abs(prop[:-Last_ARMA]) < 1):
            lprop = log_p_fn(prop)
        else:
            lprop = -np.inf

        if onp.log(onp.random.rand()) < float(lprop - lcur):
            cur = prop
            lcur = lprop
            acc += 1

        out[i, :] = cur

    return out[burn_in:, :], acc / n_samples

# ============================================================
# MAIN
# ============================================================

def run_one_replicate(data, G):
    """
    For one dataset:
    - split into G shards (time domain indices S, and frequency indices I)
    - for each shard: MAP via basinhopping, run MH, get cov weight
    - merge via consensus
    - reparam back phi
    Returns: phi_draws (1d array)
    """
    n = len(data)

    # shard index sets (match your formulas)
    I = [item + np.arange(0, int(np.floor((n - 1) / 2)), G) for item in range(G)]
    S = [int(item * (n - 1) / G) + np.arange(0, int((n - 1) / G)) for item in range(G)]

    # periodogram & omegas
    I_pg_full = p_gram(fft(data))
    omega_full = 2 * np.pi * np.arange(1, int(n / 2) + 1) / n

    # dimensions
    if TFI_term:
        n_params = q + p + 3
        Last_ARMA = 3
    else:
        n_params = q + p + 1
        Last_ARMA = 1

    # per replicate containers
    draws = []
    weights = []

    # init param vector (PACF + [loglambda, logsigma2, d])
    params0 = 0.1 * onp.ones(n_params)

    for g in range(G):
        data_shard = data[S[g]]
        I_pg_shard = I_pg_full[I[g]]
        omega_shard = omega_full[I[g]]

        # log posterior for this shard
        log_p_fn = make_log_p(I_pg_shard, omega_shard, q, p, TFI_term, G, Last_ARMA)

        obj = lambda x: -log_p_fn(x)

        jcb = grad(obj)
        hs = hessian(obj)
        H_logp = hessian(log_p_fn)

        # basinhopping (your style)
        res = basinhopping(
            obj,
            x0=params0 + sps.norm.rvs(0, 0.01, size=n_params),
            niter=10,
            stepsize=1,
            minimizer_kwargs={"method": "trust-constr", "jac": jcb, "hess": hs},
            seed=15,
        )

        paramsStar = res.x
        sigma = onp.linalg.inv(onp.asarray(-H_logp(paramsStar), dtype=float))
        proposal_width = (2.38 / onp.sqrt(n_params)) * sigma

        draw, acc_rate = sampler(paramsStar, proposal_width, log_p_fn, N_SAMPLES, BURN_IN, Last_ARMA)

        draws.append(draw)
        weights.append(onp.linalg.inv(onp.cov(draw, rowvar=False)))

        # carry last MAP as next init baseline (optional but helps)
        params0 = paramsStar

        print(f"  shard {g+1}/{G} acc={acc_rate:.3f}")

    # merge (your exact formula)
    Wsum_inv = onp.linalg.inv(sum(weights))
    merged = onp.dot(Wsum_inv, sum(onp.dot(weights[g], draws[g].T) for g in range(G))).T

    # back-transform phi (q=1)
    phi = onp.array(merged[:, :q], copy=True)
    for i in range(phi.shape[0]):
        phi[i] = reparam(phi[i])

    return phi[:, 0].copy()

def main():
    phi_list = []

    for rep_id in range(1, N_REP + 1):
        print("\n" + "=" * 50)
        print(f"Replicate {rep_id:02d}/{N_REP}  |  CASE={CASE}  |  G={G}")
        print("=" * 50)

        data_path = os.path.join(DATA_DIR, f"{CASE}_{rep_id:03d}_y.csv")
        data = onp.loadtxt(data_path, delimiter=",", skiprows=1)

        phi_draws = run_one_replicate(data, G)
        phi_list.append(phi_draws)

        print("  saved phi draws shape:", phi_draws.shape)

    # ============================================================
    # SAVE RESULTS (ABSOLUTE PATH)
    # ============================================================
    OUT_DIR = (
        "/Users/zixuanwang/Library/CloudStorage/"
        "OneDrive-UTS/Project 1/DC-BATS/results"
    )
    os.makedirs(OUT_DIR, exist_ok=True)

    out_name = os.path.join(OUT_DIR, f"{CASE}_phi_list_G{G}.pkl")
    with open(out_name, "wb") as f:
        pickle.dump(phi_list, f)

    print("\nDONE.")
    print("Saved:", out_name)
    print("File exists:", os.path.exists(out_name))
    print("len(phi_list):", len(phi_list))
    print("phi_list[0].shape:", phi_list[0].shape)

if __name__ == "__main__":
    main()

    

sys.exit()


# ============================================================
# W1 helper functions
# ============================================================

def _quantiles_01_99(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        raise ValueError("Sample too small after filtering (need >= 10).")
    probs = np.linspace(0.01, 0.99, 99)
    try:
        return np.quantile(x, probs, method="linear")
    except TypeError:
        return np.quantile(x, probs, interpolation="linear")


def _w1_from_quantiles(qP: np.ndarray, qQ: np.ndarray, method: str = "riemann") -> float:
    qP = np.asarray(qP, dtype=float)
    qQ = np.asarray(qQ, dtype=float)
    if qP.shape != (99,) or qQ.shape != (99,):
        raise ValueError("qP and qQ must both be length-99 vectors.")
    du = 0.01
    d = np.abs(qP - qQ)

    method = method.lower()
    if method == "trap":
        return du * (0.5 * d[0] + d[1:-1].sum() + 0.5 * d[-1])
    elif method == "riemann":
        return du * d.sum()
    else:
        raise ValueError("method must be 'riemann' or 'trap'.")


def phi_distance_w1(phi_full: np.ndarray,
                    phi_cons: np.ndarray,
                    method: str = "riemann",
                    normalize: bool = True) -> dict:

    q_full = _quantiles_01_99(phi_full)
    q_cons = _quantiles_01_99(phi_cons)

    W1 = _w1_from_quantiles(q_full, q_cons, method=method)

    if not normalize:
        return {"W1": W1, "W1_rel": None}

    du = 0.01
    median_cons = q_cons[49]
    a = np.abs(q_cons - median_cons)

    if method == "trap":
        scale = du * (0.5 * a[0] + a[1:-1].sum() + 0.5 * a[-1])
    else:
        scale = du * a.sum()

    W1_rel = W1 / scale if scale > 0 else np.nan
    return {"W1": W1, "W1_rel": W1_rel}


# ============================================================
# USER SETTINGS
# ============================================================

CASE = "case2"        # or "case2"
G_BASE = 1            # baseline full posterior
G_COMP = 20            # distributed posterior you want to compare

RESULTS_DIR = (
    "/Users/zixuanwang/Library/CloudStorage/"
    "OneDrive-UTS/Project 1/DC-BATS/results"
)

# ============================================================
# LOAD PKL FILES
# ============================================================

with open(os.path.join(RESULTS_DIR, f"{CASE}_phi_list_G{G_BASE}.pkl"), "rb") as f:
    phi_list_full = pickle.load(f)

with open(os.path.join(RESULTS_DIR, f"{CASE}_phi_list_G{G_COMP}.pkl"), "rb") as f:
    phi_list_cons = pickle.load(f)

assert len(phi_list_full) == len(phi_list_cons), "Replicate count mismatch"

N_REP = len(phi_list_full)

# ============================================================
# COMPUTE W1 PER REPLICATE
# ============================================================

W1_vals = []
W1_rel_vals = []

for i in range(N_REP):
    res = phi_distance_w1(
        phi_full=phi_list_full[i],
        phi_cons=phi_list_cons[i],
        method="riemann",
        normalize=True
    )
    W1_vals.append(res["W1"])
    W1_rel_vals.append(res["W1_rel"])

# ============================================================
# SUMMARY
# ============================================================

print("\n================ W1 SUMMARY ================")
print(f"CASE = {CASE}")
print(f"Compare G={G_BASE} vs G={G_COMP}")
print(f"#replicates = {N_REP}")
print("-------------------------------------------")
print("Mean W1     :", np.mean(W1_vals))
print("Std  W1     :", np.std(W1_vals))
print("Mean W1_rel :", np.mean(W1_rel_vals))
print("Std  W1_rel :", np.std(W1_rel_vals))
print("===========================================\n")

















