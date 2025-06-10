#!/usr/bin/env python3
"""
Spectral Parallel MCMC using Spark, DFFT, and integrated modeling functions
"""

from typing import Any, Dict
import os
import sys
import platform
import pickle

import pandas as pd
import autograd.numpy as np
from autograd import grad, hessian, jacobian
import autograd.scipy.stats as sps_autograd
from scipy.stats import multivariate_normal
import scipy.stats as sps
from scipy.optimize import minimize, Bounds
from numpy.fft import fft
import statsmodels.api as sm
from numdifftools import Hessian as Hess_finite_diff
from pyspark.sql import functions as F

# Import model settings ???
# from your_module import q, p, TFI_term, exact_L, n_samples

# Distributed FFT implementation using Spark RDD
import numpy as _np, math, cmath


def compute_subfft(kv, P, M):
    p, items = kv
    # sort by r = (global_idx - p)//P
    items = sorted(items, key=lambda x: (x[0] - p)//P)
    vals = [v for _, v in items]            # length M
    fft_vals = _np.fft.fft(vals)             # M-point DFT
    # emit (k = r + p*M, Y_p[r])
    return [(r + p*M, fft_vals[r]) for r in range(M)]


def DFFT(df, column: str, numShards: int):
    """
    Distributed FFT over a Spark DataFrame with column 'value'.
    Returns an RDD of complex FFT values.
    """
    rdd = df.rdd.map(lambda row: row[column])
    N = rdd.count()
    P = numShards

    # assert (P & (P-1)) == 0 and N % P == 0
    # before doing any work, validate P and N:
    if (P & (P - 1)) != 0:
        raise ValueError(f"P must be a power of two, but got P={P!r}")

    rem = N % P
    if rem > 0:
        # zip each element with its global index
        print(f"N must be divisible by P, the last {rem} values will be discarded.")
        rdd = (
            rdd
            .zipWithIndex()                             # yields (value, idx) with idx from 0…N-1
            .filter(lambda vi: vi[1] < N - rem)         # keep only those with idx < N-rem
            .map(lambda vi: vi[0])                      # strip off the index
        )


    M = N // P
    # 1) M-point FFT on each of the P interleaved slices
    subffts = (
        rdd
        .zipWithIndex()                                        # (value, idx)
        .map(lambda vi: (vi[1] % P, (vi[1], vi[0])))            # (p, (idx, val))
        .groupByKey(numPartitions=P)
        .flatMap(lambda kv: compute_subfft(kv, P, M))          # (k, Y_p[r])
        .cache()
    )
    # 2) post-twiddle by W_N^(p·r)
    twiddled = subffts.map(lambda kv: (
        kv[0],
        kv[1] * cmath.exp(-2j * math.pi * (kv[0]//M)*(kv[0]%M) / N)
    ))
    # 3) gather values and do length-P FFT
    final_rdd = (
        twiddled
        .map(lambda kv: (kv[0] % M, (kv[0]//M, kv[1])))  # → (r, (p, val))
        .groupByKey(numPartitions=M)
        .flatMap(lambda kv: [
            (q*M + kv[0], fft_val)
            for q, fft_val in enumerate(
                _np.fft.fft([v for _, v in sorted(kv[1])])
            )
        ])
        .sortByKey()
        .map(lambda kv: kv[1])
    )

    # Spark DF could not handle complex
    df = spark.createDataFrame(
        final_rdd.map(lambda c: (float(c.real), float(c.imag))),
        schema=["real", "imag"])

    return df


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
    newp = params.copy().astype(float)
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

    bar = progressbar.progressbar(range(n_samples))
    for i in bar:
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
    data = pdf['value'].to_numpy()

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

    draws, logp_trace, accepts = sampler(q_,p_,data,I_pg,TFI,omega,
                                         n_samps,theta_map,prop_cov,burn,
                                         exact=exact)
    acc_rate = np.mean(accepts)

    return pd.DataFrame({
        'group_id': [shard_id],
        'samples': [draws],
        'log_p': [logp_trace[0]],
        'acceptance_rate': [acc_rate]
    })


def insert_group_id(sdf, n_groups, method):
    """
    Simple function that adds consecutive partition ids for a Spark DataFrame.

    """
    if method == "ts":
        sample_size = sdf.count()
        id = spark.range(sample_size) # Spark DataFrame with an 'id' column 0,1,2,...
        sdf = sdf.join(id)
        sample_size_per_partition = int(sample_size/n_groups)
        sdf = sdf.withColumn("group_id", F.floor(sdf.id/sample_size_per_partition))

    elif method == "random":
        sdf = sdf.withColumn("group_id", F.monotonicall_increasing_id() % n_groups)

    return sdf


# def main():
#   main()

if __name__ == "__main__":
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import pandas_udf, PandasUDFType
    from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, DoubleType

    spark = SparkSession.builder.appName("SpectralParallelMCMC").getOrCreate()

    raw_df = spark.read.csv("data/SimARTFIMA11.csv", header=True, inferSchema=True)

    G = 2**2  # 2^power with DFFT
    periodogram_df = DFFT(raw_df, 'y',  numShards=G)

    raise OSError

    conf_model = {'TFI_term': True,
                  'partition_num': G,
                  'q': 1,
                  'p': 1,
                  'exact_L': True}
    conf_mcmc = {'n_samples': 5000,
                 'Burn_in': 1000}

    schema = StructType([
        StructField('group_id', IntegerType(), False),
        StructField('samples', ArrayType(ArrayType(DoubleType())), False),
        StructField('log_p', DoubleType(), False),
        StructField('acceptance_rate', DoubleType(), False),
    ])

    @pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
    def shard_mcmc(pdf):
        return mapper(pdf, conf_model, conf_mcmc)


    result_df = (
        periodogram_df
        .groupBy('shard_id')
        .apply(shard_mcmc)
        .orderBy('group_id')
    )

    res_pdf = result_df.toPandas()
    print(res_pdf.head())

    spark.stop()


# if __name__ == "__main__":
#     main()
