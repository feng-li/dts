#!/usr/bin/env python3
"""Standalone Spark spectral MCMC for dynamic regression with AR(2) errors.

The model is

    y_t = x_t' beta + epsilon_t
    epsilon_t = phi_1 epsilon_{t-1} + phi_2 epsilon_{t-2} + e_t,
    e_t ~ N(0, sigma2).

Spark computes the FFT of y and every covariate, partitions the positive
Fourier frequencies systematically, and runs one Whittle subposterior chain
per group with ``applyInPandas``. Parameters are sampled as

    [ar_partial_1, ar_partial_2, beta_1, ..., beta_k, log_sigma2].

The AR partial autocorrelations are transformed to stationary AR coefficients
when writing ``*_transformed.npy`` outputs.

GitHub references
-----------------
Lachlan Astfalck's R/Stan DC-BATS implementation, including the linear
regression model with AR(2) errors:
https://github.com/astfalckl/dcbats
https://github.com/astfalckl/dcbats/tree/main/linear_regression

The original DC-BATS research implementation by Deborshee Sen and coauthors:
https://github.com/deborsheesen/DC-BATS-deborshee

The maintained DTS project containing this standalone Spark implementation:
https://github.com/feng-li/dts

The first two repositories implement time-domain DC-BATS reference methods.
This file instead implements the frequency-domain spectral comparator.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path
from typing import Any

import autograd.numpy as np
from autograd import grad, hessian
import numpy as onp
import pandas as pd
from scipy.optimize import Bounds, basinhopping, minimize


TWO_PI = 2.0 * math.pi


class BoundedRandomStep:
    """Bounded proposal used by optional basin hopping."""

    def __init__(self, lower, upper, stepsize: float = 1.0, seed: int = 15):
        self.lower = onp.asarray(lower, dtype=float)
        self.upper = onp.asarray(upper, dtype=float)
        self.stepsize = float(stepsize)
        self.rng = onp.random.default_rng(seed)

    def __call__(self, params):
        displacement = self.rng.uniform(
            -self.stepsize,
            self.stepsize,
            size=onp.shape(params),
        )
        return onp.clip(params + displacement, self.lower, self.upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/DC-BATS_AR2_try.npy"),
        help="NPY [x,y,residual] data or a CSV file",
    )
    parser.add_argument(
        "--x-columns",
        default="x",
        help="comma-separated CSV covariate columns; ignored for NPY input",
    )
    parser.add_argument("--y-column", default="y", help="CSV response column")
    parser.add_argument(
        "--time-column",
        default=None,
        help="required CSV column used to establish chronological order",
    )
    parser.add_argument("--groups", type=int, default=10, help="Whittle groups G")
    parser.add_argument(
        "--fft-partitions",
        type=int,
        default=256,
        help="power-of-two DFFT partitions P",
    )
    parser.add_argument("--samples", type=int, default=15000)
    parser.add_argument("--burn-in", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument(
        "--basinhopping-iter",
        type=int,
        default=0,
        help="optional basin-hopping iterations after the initial MAP fit",
    )
    parser.add_argument(
        "--proposal-scale",
        type=float,
        default=None,
        help="proposal covariance multiplier; default is 2.38^2 / dimension",
    )
    parser.add_argument("--beta-prior-sd", type=float, default=1.0)
    parser.add_argument("--log-sigma2-prior-sd", type=float, default=1.0)
    parser.add_argument(
        "--trim",
        choices=["start", "end"],
        default="start",
        help="side trimmed when T is not divisible by P",
    )
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--master", default=None, help="optional Spark master URL")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dts_ar2_spark"),
    )
    return parser.parse_args()


def load_input_dataframe(spark, args: argparse.Namespace):
    """Load NPY or CSV input and return (DataFrame, x columns, y column)."""

    if args.input.suffix.lower() == ".npy":
        data = onp.load(args.input, allow_pickle=True)
        if data.ndim < 1 or len(data) < 2:
            raise ValueError(
                f"expected [x, y, residual] style NPY data, got shape {data.shape}"
            )

        x = onp.asarray(data[0], dtype=float)
        y = onp.asarray(data[1], dtype=float).reshape(-1)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2 or x.shape[0] != len(y):
            raise ValueError(f"incompatible x and y shapes: {x.shape} and {y.shape}")

        x_columns = [f"x{index + 1}" for index in range(x.shape[1])]
        y_column = "y"
        frame = pd.DataFrame(x, columns=x_columns)
        frame[y_column] = y
        time_column = "__time_index"
        frame.insert(0, time_column, onp.arange(len(frame), dtype=onp.int64))
        return spark.createDataFrame(frame), x_columns, y_column, time_column

    x_columns = [item.strip() for item in args.x_columns.split(",") if item.strip()]
    if not x_columns:
        raise ValueError("--x-columns must contain at least one column")
    if not args.time_column:
        raise ValueError("--time-column is required for CSV input")

    raw = spark.read.option("header", True).option("inferSchema", True).csv(str(args.input))
    required = [args.time_column] + x_columns + [args.y_column]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"input is missing columns: {', '.join(missing)}")

    from pyspark.sql import functions as F

    selected = raw.select(
        F.col(args.time_column),
        *[
            F.col(column).cast("double").alias(column)
            for column in x_columns + [args.y_column]
        ],
    ).dropna()
    return selected, x_columns, args.y_column, args.time_column


def build_indexed_rdd(
    dataframe,
    columns: list[str],
    partitions: int,
    trim: str,
    time_column: str,
):
    """Assign one stable time index shared by all columns and trim for DFFT."""

    if partitions < 1 or partitions & (partitions - 1):
        raise ValueError(f"--fft-partitions must be a power of two, got {partitions}")

    indexed = (
        dataframe.orderBy(time_column).select(*columns)
        .rdd.map(lambda row: tuple(float(row[column]) for column in columns))
        .zipWithIndex()
        .map(lambda value_index: (int(value_index[1]), value_index[0]))
        .cache()
    )
    n_original = indexed.count()
    if n_original < partitions:
        raise ValueError(
            f"number of observations {n_original} is smaller than P={partitions}"
        )

    remainder = n_original % partitions
    if remainder == 0:
        return indexed, n_original, 0

    n_effective = n_original - remainder
    if trim == "start":
        trimmed = (
            indexed.filter(lambda item: item[0] >= remainder)
            .map(lambda item: (item[0] - remainder, item[1]))
            .cache()
        )
    else:
        trimmed = indexed.filter(lambda item: item[0] < n_effective).cache()

    trimmed.count()
    indexed.unpersist()
    return trimmed, n_effective, remainder


def compute_subfft(group, partitions: int, block_size: int):
    shard, items = group
    ordered = sorted(items, key=lambda item: (item[0] - shard) // partitions)
    values = [value for _, value in ordered]
    fft_values = onp.fft.fft(values)
    return [
        (frequency + shard * block_size, shard, fft_values[frequency])
        for frequency in range(block_size)
    ]


def distributed_fft(
    spark,
    indexed_rdd,
    value_position: int,
    n_obs: int,
    partitions: int,
    prefix: str,
):
    """Two-stage decimation-in-time FFT following legacySpark.py."""

    block_size = n_obs // partitions
    subffts = (
        indexed_rdd.map(
            lambda item: (
                item[0] % partitions,
                (item[0], item[1][value_position]),
            )
        )
        .groupByKey(numPartitions=partitions)
        .flatMap(lambda group: compute_subfft(group, partitions, block_size))
        .cache()
    )

    twiddled = subffts.map(
        lambda item: (
            item[0],
            item[1],
            item[2]
            * cmath.exp(
                -2j
                * math.pi
                * (item[0] // block_size)
                * (item[0] % block_size)
                / n_obs
            ),
        )
    )
    grouped = twiddled.map(
        lambda item: (item[0] % block_size, (item[1], item[2]))
    ).groupByKey(numPartitions=partitions)

    def second_stage(group):
        frequency, sequence = group
        ordered = sorted(sequence, key=lambda item: item[0])
        fft_values = onp.fft.fft([value for _, value in ordered])
        for outer_frequency, value in enumerate(fft_values):
            yield (outer_frequency * block_size + frequency, value)

    rows = grouped.flatMap(second_stage).map(
        lambda item: (
            int(item[0]),
            float(item[1].real),
            float(item[1].imag),
        )
    )
    result = spark.createDataFrame(
        rows,
        schema=f"k long, {prefix}_real double, {prefix}_imag double",
    ).cache()
    result.count()
    subffts.unpersist()
    return result


def build_frequency_groups(
    spark,
    indexed_rdd,
    n_obs: int,
    partitions: int,
    groups: int,
    n_exog: int,
):
    """Create systematic Whittle groups with y and X FFT coefficients."""

    from pyspark.sql import functions as F

    n_frequencies = (n_obs - 1) // 2
    if groups < 1 or groups > n_frequencies:
        raise ValueError(f"G must satisfy 1 <= G <= {n_frequencies}, got {groups}")

    fft_frames = [
        distributed_fft(spark, indexed_rdd, n_exog, n_obs, partitions, "y")
    ]
    for index in range(n_exog):
        fft_frames.append(
            distributed_fft(
                spark,
                indexed_rdd,
                index,
                n_obs,
                partitions,
                f"x{index + 1}",
            )
        )

    joined = fft_frames[0]
    for frame in fft_frames[1:]:
        joined = joined.join(frame, on="k", how="inner")

    columns = ["k", "shard_id", "omega", "y_real", "y_imag"]
    for index in range(n_exog):
        columns.extend([f"x{index + 1}_real", f"x{index + 1}_imag"])

    frequency_data = (
        joined.filter((F.col("k") >= 1) & (F.col("k") <= n_frequencies))
        .withColumn("omega", F.lit(TWO_PI) * F.col("k") / F.lit(n_obs))
        .withColumn("shard_id", F.pmod(F.col("k") - F.lit(1), F.lit(groups)))
        .select(*columns)
        .repartition(groups, "shard_id")
        .sortWithinPartitions("k")
        .cache()
    )
    actual_frequencies = frequency_data.count()
    if actual_frequencies != n_frequencies:
        raise RuntimeError(
            f"DFFT produced {actual_frequencies} positive frequencies; "
            f"expected {n_frequencies}"
        )

    for frame in fft_frames:
        frame.unpersist()
    return frequency_data, n_frequencies


def reparam_ar2(partial):
    """Transform AR(2) partial autocorrelations to ordinary coefficients."""

    return np.array([partial[0] * (1.0 - partial[1]), partial[1]])


def regression_log_prior(
    params,
    n_exog: int,
    n_groups: int,
    beta_sd: float,
    log_sigma2_sd: float,
):
    partial = params[:2]
    beta = params[2 : 2 + n_exog]
    log_sigma2 = params[-1]

    process_prior = np.where(
        np.all(np.abs(partial) < 1.0),
        -2.0 * np.log(2.0),
        -np.inf,
    )
    beta_prior = np.sum(
        -0.5 * (beta / beta_sd) ** 2 - np.log(beta_sd) - 0.5 * np.log(TWO_PI)
    )
    variance_prior = (
        -0.5 * (log_sigma2 / log_sigma2_sd) ** 2
        - np.log(log_sigma2_sd)
        - 0.5 * np.log(TWO_PI)
    )
    return (process_prior + beta_prior + variance_prior) / n_groups


def regression_whittle_log_likelihood(
    params,
    y_real,
    y_imag,
    x_real,
    x_imag,
    omega,
    n_obs: int,
    n_exog: int,
):
    beta = params[2 : 2 + n_exog]
    phi = reparam_ar2(params[:2])
    sigma2 = np.exp(params[-1])

    periodogram = residual_periodogram(
        y_real,
        y_imag,
        x_real,
        x_imag,
        beta,
        n_obs,
    )

    denominator_real = (
        1.0 - phi[0] * np.cos(omega) - phi[1] * np.cos(2.0 * omega)
    )
    denominator_imag = phi[0] * np.sin(omega) + phi[1] * np.sin(2.0 * omega)
    density = (sigma2 / TWO_PI) / (
        denominator_real**2 + denominator_imag**2
    )
    return -np.sum(np.log(density) + periodogram / density)


def residual_periodogram(
    y_real,
    y_imag,
    x_real,
    x_imag,
    beta,
    n_obs: int,
):
    """Periodogram of y - X beta from aligned Fourier coefficients."""

    residual_real = y_real - np.dot(x_real, beta)
    residual_imag = y_imag - np.dot(x_imag, beta)
    return (residual_real**2 + residual_imag**2) / (TWO_PI * n_obs)


def make_positive_definite(matrix, minimum_eigenvalue: float = 1e-8):
    matrix = onp.asarray(matrix, dtype=float)
    symmetric = (matrix + matrix.T) / 2.0
    values, vectors = onp.linalg.eigh(symmetric)
    values = onp.clip(values, minimum_eigenvalue, None)
    return (vectors * values) @ vectors.T


def sample_regression(
    logp_fn,
    theta_map,
    proposal_cov,
    n_samples: int,
    burn_in: int,
    seed: int,
):
    if burn_in < 0 or burn_in >= n_samples:
        raise ValueError("burn-in must satisfy 0 <= burn-in < samples")

    rng = onp.random.default_rng(seed)
    theta_current = onp.asarray(theta_map, dtype=float)
    proposal_cov = make_positive_definite(proposal_cov)
    draws = onp.zeros((n_samples, len(theta_current)))
    logp_trace = onp.zeros(n_samples)
    accepted = onp.zeros(n_samples, dtype=bool)
    logp_current = float(logp_fn(theta_current))

    for iteration in range(n_samples):
        proposal = rng.multivariate_normal(theta_current, proposal_cov)
        if onp.all(onp.abs(proposal[:2]) < 1.0):
            logp_proposal = float(logp_fn(proposal))
        else:
            logp_proposal = -onp.inf

        if onp.log(rng.random()) < min(0.0, logp_proposal - logp_current):
            theta_current = proposal
            logp_current = logp_proposal
            accepted[iteration] = True

        draws[iteration] = theta_current
        logp_trace[iteration] = logp_current

    return (
        draws[burn_in:],
        logp_trace[burn_in:],
        accepted[burn_in:],
    )


def fit_shard(
    pdf: pd.DataFrame,
    model_config: dict[str, Any],
    mcmc_config: dict[str, Any],
) -> pd.DataFrame:
    """MAP fit and MCMC for one systematic frequency group."""

    pdf = pdf.sort_values("k")
    shard_id = int(pdf["shard_id"].iat[0])
    n_exog = int(model_config["n_exog"])
    n_groups = int(model_config["n_groups"])
    n_obs = int(model_config["n_obs"])
    n_params = n_exog + 3
    shard_seed = int(mcmc_config["seed"]) + shard_id
    rng = onp.random.default_rng(shard_seed)

    y_real = onp.asarray(pdf["y_real"], dtype=float)
    y_imag = onp.asarray(pdf["y_imag"], dtype=float)
    omega = onp.asarray(pdf["omega"], dtype=float)
    x_real = onp.column_stack(
        [onp.asarray(pdf[f"x{index + 1}_real"], dtype=float) for index in range(n_exog)]
    )
    x_imag = onp.column_stack(
        [onp.asarray(pdf[f"x{index + 1}_imag"], dtype=float) for index in range(n_exog)]
    )

    def logp_fn(params):
        return regression_whittle_log_likelihood(
            params,
            y_real,
            y_imag,
            x_real,
            x_imag,
            omega,
            n_obs,
            n_exog,
        ) + regression_log_prior(
            params,
            n_exog,
            n_groups,
            model_config["beta_prior_sd"],
            model_config["log_sigma2_prior_sd"],
        )

    process_limit = 1.0 - 1e-8
    lower = onp.full(n_params, -30.0)
    upper = onp.full(n_params, 30.0)
    lower[:2] = -process_limit
    upper[:2] = process_limit
    bounds = Bounds(lower, upper, keep_feasible=True)
    theta0 = onp.clip(rng.normal(0.0, 0.05, size=n_params), lower, upper)

    objective = lambda params: -logp_fn(params)
    objective_grad = grad(objective)
    optimizer_success = True
    optimizer_message = "optimization disabled"

    if mcmc_config["optimize"]:
        minimizer_options = {"maxiter": int(mcmc_config["max_iter"])}
        result = minimize(
            objective,
            theta0,
            method="L-BFGS-B",
            jac=objective_grad,
            bounds=bounds,
            options=minimizer_options,
        )
        if int(mcmc_config["basinhopping_iter"]) > 0:
            result = basinhopping(
                objective,
                result.x,
                niter=int(mcmc_config["basinhopping_iter"]),
                take_step=BoundedRandomStep(lower, upper, seed=shard_seed),
                minimizer_kwargs={
                    "method": "L-BFGS-B",
                    "jac": objective_grad,
                    "bounds": bounds,
                    "options": minimizer_options,
                },
                seed=shard_seed,
            )
            local_result = result.lowest_optimization_result
            optimizer_success = bool(local_result.success)
            optimizer_message = str(local_result.message)
        else:
            optimizer_success = bool(result.success)
            optimizer_message = str(result.message)
        theta_map = onp.asarray(result.x, dtype=float)
    else:
        theta_map = theta0

    if not onp.isfinite(float(logp_fn(theta_map))):
        raise RuntimeError(f"non-finite MAP log posterior for shard {shard_id}")

    logp_hessian = onp.asarray(hessian(logp_fn)(theta_map), dtype=float)
    precision = make_positive_definite(-logp_hessian)
    posterior_cov = onp.linalg.inv(precision)
    scale = mcmc_config["proposal_scale"]
    if scale is None:
        scale = 2.38**2 / n_params
    proposal_cov = make_positive_definite(float(scale) * posterior_cov)

    draws, logp_trace, accepted = sample_regression(
        logp_fn,
        theta_map,
        proposal_cov,
        int(mcmc_config["n_samples"]),
        int(mcmc_config["burn_in"]),
        shard_seed + 10000,
    )

    return pd.DataFrame(
        {
            "shard_id": [shard_id],
            "map_estimate": [theta_map.tolist()],
            "proposal_cov": [proposal_cov.tolist()],
            "samples": [draws.tolist()],
            "log_p": [logp_trace.tolist()],
            "acceptance_rate": [float(onp.mean(accepted))],
            "optimizer_success": [optimizer_success],
            "optimizer_message": [optimizer_message],
        }
    )


def consensus_draws(shard_draws: onp.ndarray) -> onp.ndarray:
    """Precision-weighted Consensus Monte Carlo draws."""

    weights = []
    total_precision = onp.zeros(
        (shard_draws.shape[2], shard_draws.shape[2]), dtype=float
    )
    for chain in shard_draws:
        weight = onp.linalg.pinv(onp.cov(chain, rowvar=False))
        weights.append(weight)
        total_precision += weight

    right_hand_side = onp.zeros_like(shard_draws[0])
    for chain, weight in zip(shard_draws, weights):
        right_hand_side += chain @ weight
    return right_hand_side @ onp.linalg.pinv(total_precision)


def transform_draws(draws: onp.ndarray, n_exog: int) -> onp.ndarray:
    """Convert partial AR parameters and log variance for reporting."""

    transformed = onp.asarray(draws, dtype=float).copy()
    partial1 = transformed[:, 0].copy()
    partial2 = transformed[:, 1].copy()
    transformed[:, 0] = partial1 * (1.0 - partial2)
    transformed[:, 1] = partial2
    transformed[:, 2 + n_exog] = onp.exp(transformed[:, 2 + n_exog])
    return transformed


def save_results(
    result_frame: pd.DataFrame,
    output: Path,
    n_exog: int,
    manifest: dict[str, Any],
):
    output.mkdir(parents=True, exist_ok=True)
    result_frame = result_frame.sort_values("shard_id")
    shard_draws = onp.stack(
        [onp.asarray(value, dtype=float) for value in result_frame["samples"]]
    )
    average = onp.mean(shard_draws, axis=0)
    consensus = consensus_draws(shard_draws)

    onp.save(output / "shard_draws.npy", shard_draws)
    onp.save(output / "average_draws.npy", average)
    onp.save(output / "consensus_draws.npy", consensus)
    onp.save(
        output / "average_draws_transformed.npy",
        transform_draws(average, n_exog),
    )
    onp.save(
        output / "consensus_draws_transformed.npy",
        transform_draws(consensus, n_exog),
    )

    diagnostics = result_frame[
        [
            "shard_id",
            "acceptance_rate",
            "optimizer_success",
            "optimizer_message",
        ]
    ].copy()
    diagnostics.to_csv(output / "shard_diagnostics.csv", index=False)
    with (output / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)


def main() -> None:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.burn_in < 0 or args.burn_in >= args.samples:
        raise ValueError("--burn-in must satisfy 0 <= burn-in < samples")
    if args.beta_prior_sd <= 0 or args.log_sigma2_prior_sd <= 0:
        raise ValueError("prior standard deviations must be positive")

    builder = SparkSession.builder.appName("DTS-AR2-Spectral-MCMC")
    if args.master:
        builder = builder.master(args.master)
    spark = builder.config("spark.sql.shuffle.partitions", str(args.groups)).getOrCreate()

    try:
        data_frame, x_columns, y_column, time_column = load_input_dataframe(spark, args)
        input_columns = x_columns + [y_column]
        indexed_rdd, n_obs, trimmed = build_indexed_rdd(
            data_frame,
            input_columns,
            args.fft_partitions,
            args.trim,
            time_column,
        )
        frequency_data, n_frequencies = build_frequency_groups(
            spark,
            indexed_rdd,
            n_obs,
            args.fft_partitions,
            args.groups,
            len(x_columns),
        )
        indexed_rdd.unpersist()

        model_config = {
            "n_exog": len(x_columns),
            "n_groups": args.groups,
            "n_obs": n_obs,
            "beta_prior_sd": args.beta_prior_sd,
            "log_sigma2_prior_sd": args.log_sigma2_prior_sd,
        }
        mcmc_config = {
            "n_samples": args.samples,
            "burn_in": args.burn_in,
            "seed": args.seed,
            "optimize": not args.no_optimize,
            "max_iter": args.max_iter,
            "basinhopping_iter": args.basinhopping_iter,
            "proposal_scale": args.proposal_scale,
        }

        schema = StructType(
            [
                StructField("shard_id", IntegerType(), False),
                StructField("map_estimate", ArrayType(DoubleType()), False),
                StructField(
                    "proposal_cov",
                    ArrayType(ArrayType(DoubleType())),
                    False,
                ),
                StructField(
                    "samples",
                    ArrayType(ArrayType(DoubleType())),
                    False,
                ),
                StructField("log_p", ArrayType(DoubleType()), False),
                StructField("acceptance_rate", DoubleType(), False),
                StructField("optimizer_success", BooleanType(), False),
                StructField("optimizer_message", StringType(), False),
            ]
        )

        result = (
            frequency_data.groupBy("shard_id")
            .applyInPandas(
                lambda pdf: fit_shard(pdf, model_config, mcmc_config),
                schema=schema,
            )
            .cache()
        )
        completed_groups = result.count()
        if completed_groups != args.groups:
            raise RuntimeError(
                f"completed {completed_groups} Spark groups; expected {args.groups}"
            )

        spark_output = args.output / "spark_shards"
        result.write.mode("overwrite").parquet(str(spark_output))
        result_frame = result.orderBy("shard_id").toPandas()

        parameter_names = ["phi1", "phi2"]
        parameter_names.extend(
            ["beta" if len(x_columns) == 1 else f"beta{index + 1}" for index in range(len(x_columns))]
        )
        parameter_names.append("sigma2")
        manifest = {
            "input": str(args.input),
            "x_columns": x_columns,
            "y_column": y_column,
            "time_column": time_column,
            "n_obs": n_obs,
            "trimmed_observations": trimmed,
            "trim_side": args.trim,
            "positive_frequencies": n_frequencies,
            "fft_partitions": args.fft_partitions,
            "groups": args.groups,
            "samples": args.samples,
            "burn_in": args.burn_in,
            "seed": args.seed,
            "parameter_order_transformed": parameter_names,
            "beta_prior_sd": args.beta_prior_sd,
            "log_sigma2_prior_sd": args.log_sigma2_prior_sd,
            "proposal_scale": (
                args.proposal_scale
                if args.proposal_scale is not None
                else 2.38**2 / (len(x_columns) + 3)
            ),
        }
        save_results(result_frame, args.output, len(x_columns), manifest)

        frequency_data.unpersist()
        result.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
