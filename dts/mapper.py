import autograd.numpy as np
import pandas as pd
import scipy.stats as sps
from autograd import grad, hessian
from scipy.optimize import minimize, Bounds
from numpy.fft import fft

from dts.mcmc import log_prior, whittle_log_likelihood, exact_log_likelihood_arma, sampler


def mapper(pdf, conf_model, conf_mcmc):
    """
    MCMC sampler for each data shard for Spark. The input and output should be both pandas  DataFrames
    """

    # Parameters setting
    TFI_term = conf_model['TFI_term']
    G = conf_model['partition_num']
    q = conf_model['q']
    p = conf_model['p']
    exact_L = conf_model['exact_L']

    n_samples = conf_mcmc['n_samples']
    Burn_in = conf_mcmc['Burn_in']

    # n = len(data_shard)
    # N = n * G # Assume equal partitioning

    if TFI_term:
        n_params = q + p + 3
        Last_ARMA = 3
    else:
        n_params = q + p + 1
        Last_ARMA = 1

    params = 0.01*np.ones(n_params)
    params = params + sps.norm.rvs(0, 0.1, size=len(params))

    if exact_L:
        data_shard = pdf.to_numpy()
        # 2.9.3 run sampling, jackknife bias correction and restore all results as draws
        I_pg = None
        TFI_term = None
        omega_shard = None

        def log_p(x):
            return (log_prior(x, 0, 1, Last_ARMA, TFI_term) / G +
                    np.sum(exact_log_likelihood_arma(x, params, q, p)))

    else:
        data_shard = None

        # 2.9.3 run sampling, jackknife bias correction and restore all results as draws
        def create_omega_shard(N, group_id, G):
            return 2*np.pi*np.arange(group_id, int(N/2), G)/N

        def create_p_gram_shard(x):  # Construct Periodogram
            id = int(np.floor((len(x)-1)/2))
            return np.square(np.abs(x[0:(id)]))/(2 * np.pi * len(x))

        I_pg_shard = create_p_gram_shard(fft(data_shard))
        # omega_shard = create_omega_shard()
        omega_shard = 2*np.pi*np.arange(1, int(len(data_shard)/2)+1)/len(data_shard)

        # 2.5 posterior distribution function
        # 2.9.2 find own MAP(paramsStar), as start point
        def log_p(x):
            return (log_prior(x, 0, 1, Last_ARMA, TFI_term) / G +
                    np.sum(whittle_log_likelihood(x, q, p, I_pg_shard, TFI_term, omega_shard)))


    def obj(params): return -log_p(params)
    jcb = grad(obj)

    r_logp, H_logp = grad(log_p), hessian(log_p)
    hs = hessian(obj)

    lb = [-1]*len(params)  # Constrains it to the stationary region after using the partial autocorrelation parameterisation.
    ub = [1]*len(params)

    if TFI_term:
        lb[-3:] = [-30, -30, -30]
        ub[-3:] = [30, 30, 30]
    else:
        lb[-1:] = [-30]
        ub[-1:] = [30]

    bnds = Bounds(lb, ub, keep_feasible=True)

    res = minimize(obj, bounds=bnds, jac=jcb, hess=hs,
                   method='trust-constr', x0=params)
    paramsStar = res.x
    assert(res.success)
    sigma = np.linalg.inv(-H_logp(paramsStar))

    # print('\nMAP%s' %(ind+1), paramsStar)
    proposal_width = (2.38/np.sqrt(n_params))*sigma


    # 2.9.1 each worker gets its shard of data(periodgram)
    # G = 10   # Number of groups
    # I = [item + np.arange(0, int(np.floor((n-1)/2)), G) for item in range(G)]
    # S = [int(item*(n-1)/(G)) + np.arange(0, int((n-1)/(G))) for item in range(G)]
    # k = np.arange(1,int((n/G)/2)+1) # k is used to determine Fourier frequencies and sum of Whittle likelihood

    # data_shard = data[S[ind]]
    # I_pg_shard = I_pg_full[I[ind]]

    # omega_shard = omega_full[I[ind]]

    draw, log_pi_g, Acceptance = sampler(
        q, p, data_shard, I_pg, TFI_term, omega_shard,
        n_samples, paramsStar, proposal_width, Burn_in,
        params_prior_mu=0, params_prior_sd=1., exact=exact_L)

    # Make a final Pandas DataFrame
    out_np = np.column_stack((draw, log_pi_g, Acceptance))
    out_pd_colnames = ["group_id", "log_p", 'log_p', 'Acceptance']
    out_pd = pd.DataFrame(out_np, out_pd_colnames)

    return out_pd
