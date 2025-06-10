#!/usr/bin/env python3
"""
Spectral Parallel MCMC using Spark, DFFT, and integrated modeling functions
"""

from typing import Any, Dict

import pandas as pd
import autograd.numpy as np
from autograd import grad, hessian
import autograd.scipy.stats as sps_autograd
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, Bounds
from numpy.fft import fft
import statsmodels.api as sm

# Distributed FFT implementation using Spark RDD
import numpy as _np, math, cmath


def compute_subfft(kv, P, M):
    p, items = kv
    # sort by r = (global_idx - p)//P
    items = sorted(items, key=lambda x: (x[0] - p)//P)
    vals = [v for _, v in items]
    fft_vals = _np.fft.fft(vals)
    # emit (global index k, original shard p, fft value)
    return [(r + p*M, p, fft_vals[r]) for r in range(M)]


def DFFT(df, column: str, numShards: int):
    """
    Distributed FFT over DataFrame column `column`, across `numShards` logical shards.
    Returns a Spark DataFrame with columns: shard_id, real, imag
    """
    spark = SparkSession.builder.getOrCreate()
    rdd = df.rdd.map(lambda row: row[column])
    N = rdd.count()
    P = numShards

    # 1) validate and drop tail if needed
    if (P & (P-1)) != 0:
        raise ValueError(f"P must be a power of two, but got P={P!r}")
    rem = N % P
    if rem > 0:
        print(f"Warning: dropping last {rem} elements so N is divisible by P")
        rdd = (
            rdd
            .zipWithIndex()
            .filter(lambda vi: vi[1] < N - rem)
            .map(lambda vi: vi[0])
        )
    M = (N - rem) // P

    # 2) first-stage, shard-wise M-point FFT
    subffts = (
        rdd
        .zipWithIndex()                                          # (value, idx)
        .map(lambda vi: (vi[1] % P, (vi[1], vi[0])))              # (p, (global_idx, val))
        .groupByKey(numPartitions=P)
        .flatMap(lambda kv: compute_subfft(kv, P, M))            # (k, p, complex)
        .cache()
    )

    # 3) twiddle
    twiddled = subffts.map(lambda triple: (
        triple[0],  # k
        triple[1],  # p
        triple[2] * cmath.exp(-2j * math.pi * (triple[0]//M)*(triple[0]%M) / N)
    ))

    # 4) second-stage, length-P FFT along each r
    paired = twiddled.map(lambda triple: (triple[0] % M, (triple[1], triple[2])))
    grouped = paired.groupByKey(numPartitions=M)

    def second_stage(grouped_pair):
        r, seq = grouped_pair
        seq_sorted = sorted(seq, key=lambda x: x[0])
        vals = [val for _, val in seq_sorted]
        fft_vals = _np.fft.fft(vals)
        for q, fft_v in enumerate(fft_vals):
            yield (q*M + r, q, fft_v)

    final = (
        grouped
        .flatMap(second_stage)   # yields (global_k, shard_id=q, complex)
        .sortBy(lambda triple: triple[0])  # sort by global index
    )

    # 5) build DataFrame carrying shard_id, real, imag
    row_rdd = final.map(lambda triple: (
        int(triple[1]),           # shard_id = q
        float(triple[2].real),     # real part
        float(triple[2].imag)      # imag part
    ))

    return spark.createDataFrame(row_rdd, schema=["shard_id", "real", "imag"])


def f_ARTFIMA(omega: np.ndarray, phi: np.ndarray, theta: np.ndarray,
              var: float, d: float, lambda_: float) -> np.ndarray:
    """
    Spectral density for (A)RTFIMA.
    """
    TFI = np.abs(1 - np.exp(-(lambda_ + 1j*omega)))**(-2*d)
    if phi.size:
        expp = np.exp(-1j * np.outer(omega, np.arange(1, phi.size+1)))
        denom = 1 - np.dot(expp, phi)
        vv1 = 1.0 / denom
    else:
        vv1 = 1.0

    if theta.size:
        expt = np.exp(-1j * np.outer(omega, np.arange(1, theta.size+1)))
        num = 1 + np.dot(expt, theta)
        vv2 = num
    else:
        vv2 = 1.0

    return (TFI * (var/(2*np.pi)) *
            (np.real(vv1)**2 + np.imag(vv1)**2) *
            (np.real(vv2)**2 + np.imag(vv2)**2))


def whittle_log_likelihood(params: np.ndarray, q: int, p: int,
                            I_pg: np.ndarray, TFI_term: bool,
                            omega: np.ndarray) -> np.ndarray:
    """
    Whittle log-likelihood contributions for given periodogram and params.
    """
    if TFI_term:
        d = params[-1]
        lambda_ = np.exp(params[-3])
        var = np.exp(params[-2])
    else:
        d = 0.0
        lambda_ = 0.0
        var = np.exp(params[-1])

    phi = reparam(params[:q], MA=False) if q>0 else np.array([])
    theta = reparam(params[q:q+p], MA=True) if p>0 else np.array([])

    f = f_ARTFIMA(omega, phi, theta, var, d, lambda_)
    return -(np.log(f) + I_pg / f)


def exact_log_likelihood_arma(data: np.ndarray, params: np.ndarray,
                              q: int, p: int) -> np.ndarray:
    """
    Exact Gaussian ARMA log-likelihood using innovations algorithm.
    """

    # detach your data from the AD ArrayBox before handing it to statsmodels
    if hasattr(params, "_value"):
        # pull out the underlying ndarray
        params = np.asarray(params._value, dtype=float)
    else:
        params = np.asarray(params, dtype=float)


    phi = reparam(params[:q], MA=False) if q>0 else np.array([])
    theta = reparam(params[q:q+p], MA=True) if p>0 else np.array([])
    var = np.exp(params[-1])

    return sm.tsa.innovations.arma_loglike(data, phi, theta, sigma2=var)


def log_prior(params: np.ndarray, mu: float, sd: float,
              Last_ARMA: int, TFI_term: bool, G: int) -> float:
    """
    Fractionated prior combining stationarity and normal priors.
    """
    # Stationarity/invertibility constraint
    if (np.abs(params[:-Last_ARMA]) < 1).all():
        lp = - (params.size - Last_ARMA) * np.log(2)
    else:
        return -np.inf

    # Gaussian priors on scale/diff parameters
    if TFI_term:
        lp += sps_autograd.norm.logpdf(params[-1], 0, 1)
        lp += sps_autograd.norm.logpdf(params[-3], 0, 1)
        lp += sps_autograd.norm.logpdf(params[-2], 0, 1)
    else:
        lp += sps_autograd.norm.logpdf(params[-1], 0, 1)

    return lp / G


def reparam(params: np.ndarray, MA: bool = False) -> np.ndarray:
    """
    Transform partial autocorrelation params to AR/MA coefficients.
    """
    # breakpoint()
    # newp = params.copy().astype(float)
    newp = np.array(params, dtype=float)

    for j in range(1, len(params)):
        tmp = newp[:j].copy()
        if not MA:
            tmp -= newp[j] * newp[j-1::-1]
        else:
            tmp += newp[j] * newp[j-1::-1]
        newp[:j] = tmp
    return newp


def sampler(q: int, p: int,
            data: np.ndarray, I_pg: Any,
            TFI_term: bool, omega: Any,
            n_samples: int, theta_map: np.ndarray,
            proposal_cov: np.ndarray, burn_in: int,
            params_prior_mu: float = 0.0,
            params_prior_sd: float = 1.0,
            exact: bool = False) -> Any:
    """
    Metropolis sampler for ARMA/ARTFIMA model.
    Returns draws, log_p traces, and acceptance mask.
    """
    n_params = theta_map.size
    Last_ARMA = 3 if TFI_term else 1

    draws = np.zeros((n_samples, n_params))
    logp_trace = np.zeros(n_samples)
    accepts = np.zeros(n_samples, dtype=bool)

    # Initial log posterior
    if exact:
        ll = exact_log_likelihood_arma(data, theta_map, q, p)
    else:
        ll = whittle_log_likelihood(theta_map, q, p, I_pg, TFI_term, omega)
    lp = log_prior(theta_map, 0, 1, Last_ARMA, TFI_term, G=1)
    logp_current = np.sum(ll) + lp

    for i in range(n_samples):
        proposal_cov = make_pd_eigclip(proposal_cov) # Force symmetric and positive definite.

        theta_prop = multivariate_normal.rvs(theta_map, proposal_cov)
        # check stationarity
        if (np.abs(theta_prop[:-Last_ARMA]) < 1).all():
            if exact:
                ll_prop = exact_log_likelihood_arma(data, theta_prop, q, p)
            else:
                ll_prop = whittle_log_likelihood(theta_prop, q, p, I_pg, TFI_term, omega)
        else:
            ll_prop = -np.inf

        lp_prop = log_prior(theta_prop, 0, 1, Last_ARMA, TFI_term, G=1)
        logp_prop = np.sum(ll_prop) + lp_prop

        alpha = min(1, np.exp(logp_prop - logp_current))
        if np.random.rand() < alpha:
            theta_map = theta_prop
            logp_current = logp_prop
            accepts[i] = True
        draws[i] = theta_map
        logp_trace[i] = logp_current

    return draws, logp_trace, accepts


def mapper(
    pdf: pd.DataFrame,
    conf_model: Dict[str, Any],
    conf_mcmc: Dict[str, int]
) -> pd.DataFrame:
    """
    Perform MCMC sampling on a single data shard.
    """
    # Unpack configurations
    TFI = conf_model['TFI_term']
    G = conf_model['partition_num']
    q_ = conf_model['q']
    p_ = conf_model['p']
    exact = conf_model['exact_L']
    n_samps = conf_mcmc['n_samples']
    burn = conf_mcmc['Burn_in']

    shard_id = int(pdf['shard_id'].iat[0])
    # data = pdf['value'].to_numpy()

    # Convert a complex vector from two columns
    data = pdf["real"].values + 1j * pdf["imag"].values

    # Precompute periodogram if needed
    if not exact:
        freqs = fft(data)
        I_pg = np.square(np.abs(freqs[:freqs.size//2])) / (2*np.pi*data.size)
        omega = 2*np.pi*np.arange(1, I_pg.size+1)/data.size
    else:
        I_pg, omega = None, None

    # MAP and sampling
    # reuse fns defined above
    n_params = q_ + p_ + (3 if TFI else 1)
    theta0 = 0.01*np.ones(n_params)
    theta0 += np.random.randn(n_params)*0.1

    # Define local log_p
    logp_fn = (lambda th: log_prior(th,0,1,3,TFI,G) +
               np.sum(exact_log_likelihood_arma(data,th,q_,p_))
               ) if exact else (
               lambda th: log_prior(th,0,1,3,TFI,G) +
               np.sum(whittle_log_likelihood(th,q_,p_,I_pg,TFI,omega))
               )

    obj = lambda th: -logp_fn(th)
    res = minimize(obj, x0=theta0, jac=grad(obj), hess=hessian(obj),
                   bounds=Bounds([-1]*n_params, [1]*n_params, keep_feasible=True),
                   method='trust-constr')
    theta_map = res.x
    cov_map = np.linalg.inv(-hessian(logp_fn)(theta_map))

    prop_cov = (2.38/np.sqrt(n_params))*cov_map
    prop_cov = (prop_cov + prop_cov.T)/2 # Force positive definite

    draws, logp_trace, accepts = sampler(q_,p_,data,I_pg,TFI,omega,
                                         n_samps,theta_map,prop_cov,burn,
                                         exact=exact)
    acc_rate = np.mean(accepts)

    return pd.DataFrame({
        'shard_id': [shard_id],
        'samples': [draws],
        'log_p': [logp_trace[0]],
        'acceptance_rate': [acc_rate]
    })


# def main():
#   main()
def make_pd_eigclip(A, delta=1e-8):
    """
    Force all eigenvalues of A to be >= delta by clipping.
    """
    # symmetrize
    A = (A + A.T) / 2
    vals, vecs = np.linalg.eigh(A)
    vals_clipped = np.clip(vals, delta, None)
    return (vecs * vals_clipped) @ vecs.T


if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType

    spark = SparkSession.builder.appName("SpectralParallelMCMC").getOrCreate()

    raw_df = spark.read.csv("data/SimARTFIMA11.csv", header=True, inferSchema=True)

    G = 2**5  # 2^power with DFFT
    periodogram_df = DFFT(raw_df, 'y',  numShards=G)

    conf_model = {'TFI_term': True,
                  'partition_num': G,
                  'q': 1,
                  'p': 1,
                  'exact_L': False}
    conf_mcmc = {'n_samples': 5000,
                 'Burn_in': 1000}

    schema = StructType([
        StructField('shard_id', IntegerType(), False),
        StructField('samples', ArrayType(ArrayType(DoubleType())), False),
        StructField('log_p', DoubleType(), False),
        StructField('acceptance_rate', DoubleType(), False),
    ])


    # pdf = periodogram_df.toPandas().iloc[:300,:]
    # mapper(pdf, conf_model,conf_mcmc)
    # raise Exception("Debug.")

    @pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
    def shard_mcmc(pdf):
        return mapper(pdf, conf_model, conf_mcmc)

    result_df = periodogram_df.groupBy('shard_id').apply(shard_mcmc).orderBy('shard_id')


    res_pdf = result_df.toPandas()
    print(res_pdf.head())



# if __name__ == "__main__":
#     main()
