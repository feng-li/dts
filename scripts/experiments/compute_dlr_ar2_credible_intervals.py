# This script computes 95% Bayesian credible intervals (CIs) for parameters
# of a Dynamic Linear Regression (DLR) model with AR(2) errors.
#
# The traces are obtained by running the DLR–AR(2) model following the
# divide-and-conquer implementation in Ou et al.
#
# Inputs:
# - Full posterior trace: MCMC draws obtained using the full dataset
# - Shard posterior traces: MCMC draws obtained separately on each data block
#
# Method:
# - For the full posterior, 95% CIs are computed directly from the pooled draws
#   using the 2.5% and 97.5% quantiles.
# - For the shard-based approach, 95% CIs are computed on each shard separately,
#   then averaged across shards following the procedure used in Ou et al.
#
# Outputs:
# - 95% credible intervals for beta, phi1, phi2, and sigmasq
#   for both full-data and shard-averaged posteriors.

###################################################
#Section 1: DC-BATS
###################################################

import numpy as np

QS = [0.025, 0.975]

# ---------- full posterior ----------

def full_ci(trace, param):
    draws = trace[param]
    if draws.ndim == 2:
        return np.quantile(draws, QS, axis=0)
    return np.quantile(draws, QS)


# ---------- shard averaging CI (scalar params) ----------

def shard_ci_average(shards, param):
    cis = [np.quantile(d[param], QS) for d in shards]
    return np.mean(cis, axis=0)


# ---------- shard averaging CI (vector params, e.g. phi) ----------

def shard_ci_average_vector(shards, param):
    cis = [np.quantile(d[param], QS, axis=0) for d in shards]
    return np.mean(cis, axis=0)


# ---------- load data ----------

full_trace_path = \
    '/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/results/draws/DC-BATS_AR2_LM/trace_fulldata.npy'

#'/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/DC-BATS-deborshee-master/in_paper/ar2_errors/trace_full.pkl.npy'

shards_trace_path = \
'/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/results/draws/DC-BATS_AR2_LM/trace_all.npy'

full = np.load(full_trace_path, allow_pickle=True).item()

shards = np.load(shards_trace_path, allow_pickle=True)


# ---------- compute CIs ----------

ci_alpha_full   = full_ci(full, "alpha")
ci_beta_full    = full_ci(full, "beta")
ci_sigmasq_full = full_ci(full, "sigmasq")
ci_phi_full     = full_ci(full, "phi")

ci_alpha_shard   = shard_ci_average(shards, "alpha")
ci_beta_shard    = shard_ci_average(shards, "beta")
ci_sigmasq_shard = shard_ci_average(shards, "sigmasq")
ci_phi_shard     = shard_ci_average_vector(shards, "phi")


# ---------- make scalars safe ----------

beta_full_lo, beta_full_hi = np.asarray(ci_beta_full).squeeze()
beta_shard_lo, beta_shard_hi = np.asarray(ci_beta_shard).squeeze()

sig_full_lo, sig_full_hi = np.asarray(ci_sigmasq_full).squeeze()
sig_shard_lo, sig_shard_hi = np.asarray(ci_sigmasq_shard).squeeze()

# Phi: transpose for readability
ci_phi_full_T  = np.asarray(ci_phi_full).T
ci_phi_shard_T = np.asarray(ci_phi_shard).T


# ---------- print results ----------

print("===== 95% Credible Intervals (Full data) =====")
print(f"beta     : [{beta_full_lo:.4f}, {beta_full_hi:.4f}]")
print(f"phi1     : [{ci_phi_full_T[0,0]:.4f}, {ci_phi_full_T[0,1]:.4f}]")
print(f"phi2     : [{ci_phi_full_T[1,0]:.4f}, {ci_phi_full_T[1,1]:.4f}]")
print(f"sigmasq  : [{sig_full_lo:.4f}, {sig_full_hi:.4f}]")

print("\n===== 95% Credible Intervals (Shard-averaged) =====")
print(f"beta     : [{beta_shard_lo:.4f}, {beta_shard_hi:.4f}]")
print(f"phi1     : [{ci_phi_shard_T[0,0]:.4f}, {ci_phi_shard_T[0,1]:.4f}]")
print(f"phi2     : [{ci_phi_shard_T[1,0]:.4f}, {ci_phi_shard_T[1,1]:.4f}]")
print(f"sigmasq  : [{sig_shard_lo:.4f}, {sig_shard_hi:.4f}]")




###################################################
#Section 2: Divide-and-conquer MCMC
###################################################

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




# --- load ---
arr_shard = np.load('/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/results/draws/spark/DLR_AR2_LM/G16.npy')  # (G,T,4)
arr_full  = np.load('/Users/zixuanwang/Library/CloudStorage/OneDrive-UTS/Project 1/results/draws/spark/DLR_AR2_LM/G1.npy')   # (T,4)

# --- consensus combine (G,T,4) -> (T,4) ---
def consensus(draws):
    G, T, d = draws.shape
    covs  = np.array([np.cov(draws[g], rowvar=False) for g in range(G)])  # (G,d,d)
    precs = np.linalg.inv(covs)                                          # (G,d,d)
    Sigma = np.linalg.inv(np.sum(precs, axis=0))                         # (d,d)

    weighted = np.zeros((d, T))
    for g in range(G):
        weighted += precs[g] @ draws[g].T
    return (Sigma @ weighted).T                                          # (T,d)

arr_cons = consensus(arr_shard)  # (5000, 4)

# 1) reparam phi (ar1, ar2)
phi_full = np.apply_along_axis(reparam, 1, arr_full[:, :2], MA=False)    # (T,2)
phi_cons = np.apply_along_axis(reparam, 1, arr_cons[:, :2], MA=False)    # (T,2)

# 2) exp sigmasq (log-scale -> original scale)
sig_full = np.exp(arr_full[:, 3])                                       # (T,)
sig_cons = np.exp(arr_cons[:, 3])                                       # (T,)

# 3) compute 95% CIs (full)
q_phi_full  = np.quantile(phi_full, [0.025, 0.975], axis=0)             # (2,2)
q_beta_full = np.quantile(arr_full[:, 2], [0.025, 0.975])               # (2,)
q_sig_full  = np.quantile(sig_full, [0.025, 0.975])                     # (2,)

# 4) compute 95% CIs (consensus-merged)
q_phi_cons  = np.quantile(phi_cons, [0.025, 0.975], axis=0)             # (2,2)
q_beta_cons = np.quantile(arr_cons[:, 2], [0.025, 0.975])               # (2,)
q_sig_cons  = np.quantile(sig_cons, [0.025, 0.975])                     # (2,)

# ---------- print results (order: beta, phi1, phi2, sigmasq) ----------
print("===== 95% Credible Intervals (Full data) =====")
print(f"beta     : [{q_beta_full[0]:.4f}, {q_beta_full[1]:.4f}]")
print(f"phi1     : [{q_phi_full[0,0]:.4f}, {q_phi_full[1,0]:.4f}]")
print(f"phi2     : [{q_phi_full[0,1]:.4f}, {q_phi_full[1,1]:.4f}]")
print(f"sigmasq  : [{q_sig_full[0]:.4f}, {q_sig_full[1]:.4f}]")

print("\n===== 95% Credible Intervals (Consensus-merged) =====")
print(f"beta     : [{q_beta_cons[0]:.4f}, {q_beta_cons[1]:.4f}]")
print(f"phi1     : [{q_phi_cons[0,0]:.4f}, {q_phi_cons[1,0]:.4f}]")
print(f"phi2     : [{q_phi_cons[0,1]:.4f}, {q_phi_cons[1,1]:.4f}]")
print(f"sigmasq  : [{q_sig_cons[0]:.4f}, {q_sig_cons[1]:.4f}]")