#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
"""
Spectral Parallel MCMC using Spark, DFFT, and integrated modeling functions


True parallel
"""

from typing import Any, Dict

import pandas as pd
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax.numpy as np
import numpy as onp
from jax import grad, hessian
import jax.scipy.stats as sps_jax
from scipy.stats import multivariate_normal
from scipy.optimize import minimize, Bounds, basinhopping
from numpy.fft import fft
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
# Distributed FFT implementation using Spark RDD
import numpy as _np, math, cmath
import warnings


import itertools

CALL_COUNTER = itertools.count()


def compute_subfft(kv, P, M):
    p, items = kv
    items = sorted(items, key=lambda x: (x[0] - p)//P)
    vals = [v for _, v in items]
    fft_vals = _np.fft.fft(vals)
    return [(r + p*M, p, fft_vals[r]) for r in range(M)]

def DFFT(df, column: str, numShards: int):
    spark = SparkSession.builder.getOrCreate()
    rdd = df.rdd.map(lambda row: row[column])
    N = rdd.count()
    P = numShards

    if (P & (P-1)) != 0:
        raise ValueError(f"P must be a power of two, but got P={P!r}")
    rem = N % P
    if rem > 0:
        print(f"Warning: dropping last {rem} elements so N is divisible by P")
        rdd = (
            rdd.zipWithIndex()
               .filter(lambda vi: vi[1] < N - rem)
               .map(lambda vi: vi[0])
        )
    M = (N - rem) // P

    subffts = (
        rdd.zipWithIndex()
           .map(lambda vi: (vi[1] % P, (vi[1], vi[0])))
           .groupByKey(numPartitions=P)
           .flatMap(lambda kv: compute_subfft(kv, P, M))
           .cache()
    )

    twiddled = subffts.map(lambda triple: (
        triple[0],
        triple[1],
        triple[2] * cmath.exp(-2j * math.pi * (triple[0]//M)*(triple[0]%M) / N)
    ))

    paired  = twiddled.map(lambda triple: (triple[0] % M, (triple[1], triple[2])))
    grouped = paired.groupByKey(numPartitions=P)

    def second_stage(grouped_pair):
        r, seq = grouped_pair
        seq_sorted = sorted(seq, key=lambda x: x[0])
        vals = [val for _, val in seq_sorted]
        fft_vals = _np.fft.fft(vals)
        for q, fft_v in enumerate(fft_vals):
            yield (q*M + r, q, fft_v)

    final = grouped.flatMap(second_stage).sortBy(lambda triple: triple[0])

    row_rdd = final.map(lambda triple: (
        int(triple[0]),
        float(triple[2].real),
        float(triple[2].imag)
    ))
    print("DFFT computed.")
    return spark.createDataFrame(row_rdd, ["k", "real", "imag"]).sort("k")


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
                            omega: np.ndarray, alpha = 1) -> np.ndarray:
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
    return (-(np.log(f) + I_pg / f))*alpha


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
    lp = np.where(
        np.all(np.abs(params[:-Last_ARMA]) < 1.0),
        - (params.size - Last_ARMA) * np.log(2),
        -np.inf,
    )

    # Gaussian priors on scale/diff parameters
    if TFI_term:
        lp += sps_jax.norm.logpdf(params[-1], 0, 100)
        lp += sps_jax.norm.logpdf(params[-3], -2.3, 0.4)
        lp += sps_jax.norm.logpdf(params[-2], 0, 100)
    else:
        lp += sps_jax.norm.logpdf(params[-1], 0, 100)

    return lp / G


def reparam(params: np.ndarray, MA: bool = False) -> np.ndarray:
    """
    Transform partial autocorrelation params to AR/MA coefficients.
    JAX-safe real-only version.
    """

    newp = np.array(params)

    for j in range(1, len(params)):
        prev = newp[:j]          # length j
        pj = newp[j]
        rev_prev = prev[::-1]

        if not MA:
            prev_new = prev - pj * rev_prev
        else:
            prev_new = prev + pj * rev_prev


        newp = np.concatenate([prev_new, newp[j:]])

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

    draws = onp.zeros((n_samples, n_params))
    logp_trace = onp.zeros(n_samples)
    accepts = onp.zeros(n_samples, dtype=bool)

    # Initial log posterior
    if exact:
        ll = exact_log_likelihood_arma(data, theta_map, q, p)
    else:
        ll = whittle_log_likelihood(theta_map, q, p, I_pg, TFI_term, omega)
    lp = log_prior(theta_map, 0, 1, Last_ARMA, TFI_term, G)
    logp_current = np.sum(ll) + lp

    for i in range(n_samples):
        proposal_cov = make_pd_eigclip(proposal_cov) # Force symmetric and positive definite.

        theta_prop = multivariate_normal.rvs(theta_map, proposal_cov)
        # check stationarity
        if onp.all(onp.abs(theta_prop[:-Last_ARMA]) < 1):
            if exact:
                ll_prop = exact_log_likelihood_arma(data, theta_prop, q, p)
            else:
                ll_prop = whittle_log_likelihood(theta_prop, q, p, I_pg, TFI_term, omega)
        else:
            ll_prop = -np.inf

        lp_prop = log_prior(theta_prop, 0, 1, Last_ARMA, TFI_term, G)
        logp_prop = np.sum(ll_prop) + lp_prop

        alpha = min(1.0, float(np.exp(logp_prop - logp_current)))
        if onp.random.rand() < alpha:
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
    warnings.filterwarnings(
        "ignore",
        message="Casting complex values to real discards the imaginary part",
    )    
    # Unpack configurations
    TFI = conf_model['TFI_term']
    G = conf_model['partition_num']
    q_ = conf_model['q']
    p_ = conf_model['p']
    exact = conf_model['exact_L']
    n_samps = conf_mcmc['n_samples']
    burn = conf_mcmc['Burn_in']

    shard_id = int(pdf['shard_id'].iat[0])
    onp.random.seed(15)

    Last_ARMA = 3 if TFI else 1
    
    if not exact:
        I_pg = pdf["I_pg"].values.astype(float)
        omega = pdf["omega"].values.astype(float)
        data = onp.zeros(1)   # dummy, not used under Whittle
    else:
        raise NotImplementedError("exact_L=True not supported in Spark+DFFT Whittle version.")

    # MAP and sampling
    # reuse fns defined above
    n_params = q_ + p_ + (3 if TFI else 1)
    theta0 = 0.01*onp.ones(n_params)
    theta0 += onp.random.randn(n_params)*0.1

    # Define local log_p
    logp_fn = (lambda th: log_prior(th,0,1,Last_ARMA,TFI,G) +
               np.sum(exact_log_likelihood_arma(data,th,q_,p_))
               ) if exact else (
               lambda th: log_prior(th,0,1,Last_ARMA,TFI,G) +
               np.sum(whittle_log_likelihood(th,q_,p_,I_pg,TFI,omega))
               )

    obj = lambda th: -logp_fn(th)
    jcb = grad(obj)
    hs = hessian(obj)
 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        '''
        res = minimize(obj, x0=theta0, jac=grad(obj), hess=hessian(obj),
                       bounds=Bounds([-1]*n_params, [1]*n_params, keep_feasible=True),
                       method='trust-constr')
        '''
        res = basinhopping(obj, 
                           x0 = theta0, 
                           niter=100, 
                           stepsize=1, 
                           minimizer_kwargs={
                               "method": "trust-constr", 
                               "jac":jcb, 
                               "hess":hs, 
                               #"bounds": bnds,
                               }, 
                           seed=15,
                           )
        
    theta_map = res.x

    cov_map = onp.linalg.inv(onp.asarray(-hessian(logp_fn)(theta_map), dtype=float))
    np.set_printoptions(precision=2, suppress=True, floatmode="fixed") 
    print(f"[MAP] shard {shard_id}: {np.asarray(theta_map)}", flush=True)
    prop_cov = (2.38/onp.sqrt(n_params))*cov_map
    prop_cov = (prop_cov + prop_cov.T)/2 # Force positive definite

    draws, logp_trace, accepts = sampler(q_,p_,data,I_pg,TFI,omega,
                                         n_samps,theta_map,prop_cov,burn,
                                         exact=exact)
    acc_rate = np.mean(accepts)
    #print(acc_rate)
    
    data_name = data_name_for_save#"maine_demand"
    model_name = f"ARMA{q_}{p_}" if not TFI else f"ARTFIMA{q_}{p_}"
    lik_name = "Whittle"
    exp_name = f"{data_name}_{model_name}_{lik_name}_G{G}"

    out_dir = os.path.join("artifacts", exp_name)
    os.makedirs(out_dir, exist_ok=True)

    sid = int(shard_id)

    onp.save(
        os.path.join(out_dir, f"shard{sid:02d}_draws.npy"),
        draws
    )

    onp.save(
        os.path.join(out_dir, f"shard{sid:02d}_logp.npy"),
        logp_trace
    )

    print(f"[SAVE] shard {sid} → {out_dir}", flush=True)
    
    
    
    return pd.DataFrame({
        'shard_id': [shard_id],
        'samples': [draws],
        'log_p': [logp_trace],
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
    
    #raw_df = spark.read.csv("data/SimARTFIMA11.csv", header=True, inferSchema=True)
    #raw_df = spark.read.csv("projects/data/maine_demand.csv", header=True, inferSchema=True)
    raw_df = spark.read.csv("projects/data/Bromma_AR2_TFI_MA2.csv", header=True, inferSchema=True)

    #raw_df = spark.read.csv("data/Vancouver_AR2_TFI_MA1.csv", header=True, inferSchema=True)
    #raw_array = np.loadtxt('/Users/zixuanwang/Sim_AR1_TFI_MA1.txt')
    #raw_pd = pd.DataFrame({"y": raw_array})
    #raw_df = spark.createDataFrame(raw_pd)
    data_name_for_save = "Bromma"
    
    from pyspark.sql import functions as F
    
    
    G = 16
    #spark.conf.set("spark.sql.shuffle.partitions", str(G))
    # 1) DFFT gives full FFT coefficients X[k]
    dfft_df = DFFT(raw_df, "y", numShards=G).cache()
    N_full = dfft_df.count()   # 这一步会跑一次，但之后复用 cache
    id_half = int(np.floor((N_full - 1) / 2))
    
    # 2) Build periodogram exactly matching p_gram(fft(data))
    periodogram_df = (
        dfft_df
        .filter((F.col("k") >= 0) & (F.col("k") < F.lit(id_half)))
    
        .withColumn(
            "abs2",
            F.col("real") * F.col("real") + F.col("imag") * F.col("imag")
        )
        .withColumn(
            "I_pg",
            F.col("abs2") / (2 * np.pi * F.lit(N_full))
        )
        .withColumn(
            "omega",
            2 * np.pi * F.col("k") / F.lit(N_full)
        )
        .withColumn(
            "shard_id",
            F.pmod(F.col("k"), F.lit(G))   # systematic split
        )
        .select("shard_id", "I_pg", "omega")
    )
    periodogram_df = periodogram_df.repartition(G, "shard_id").cache()
    periodogram_df.count()   # 触发 cache materialize，避免后面重复算 
    
    conf_model = {'TFI_term': True,
                  'partition_num': G,
                  'q': 2,
                  'p': 2,
                  'exact_L': False}
    conf_mcmc = {'n_samples': 15000,
                 'Burn_in': 5000}

    schema = StructType([
        StructField('shard_id', IntegerType(), False),
        StructField('samples', ArrayType(ArrayType(DoubleType())), False),
        StructField('log_p', ArrayType(DoubleType()), False),
        StructField('acceptance_rate', DoubleType(), False),
    ])


    # pdf = periodogram_df.toPandas().iloc[:300,:]
    # mapper(pdf, conf_model,conf_mcmc)
    # raise Exception("Debug.")

    @pandas_udf(schema, functionType=PandasUDFType.GROUPED_MAP)
    def shard_mcmc(pdf):
        return mapper(pdf, conf_model, conf_mcmc)

    result_df = (
    periodogram_df
    .groupBy("shard_id")
    .applyInPandas(lambda pdf: mapper(pdf, conf_model, conf_mcmc), schema)
    )
    
    result_df.count()

    
    





'''
res_pdf = result_df.toPandas()
print(res_pdf.head())

samples_list = res_pdf["samples"].to_list()

samples_npy = np.array(samples_list)

samples = samples_npy[:,-10000:,:]

logpi_list = res_pdf["log_p"].to_list()

logpi_npy = np.array(logpi_list)

log_pi = logpi_npy[:,-10000:]
'''
