#!/usr/bin/env python3
"""Compare DTS_AR2_Spark numerical routines with the DC-BATS R code."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from DTS_AR2_Spark import consensus_draws, regression_whittle_log_likelihood


DEFAULT_R_CANDIDATES = (
    Path("/data/student/Zixuan-UTS/DC-BATS/in_paper/ar(2)_errors/consensusMCcov.R"),
    Path("/tmp/DC-BATS-deborshee/in_paper/ar(2)_errors/consensusMCcov.R"),
)


R_DRIVER = r'''
args <- commandArgs(trailingOnly=TRUE)
# The reference function assumes class(matrix) has length one. Preserve its
# intended old-code behavior under current R, where it is c("matrix", "array").
base_class <- base::class
class <- function(object) base_class(object)[1]
source(args[1])

frequency_data <- read.csv(args[2], check.names=FALSE)
params <- scan(args[3], quiet=TRUE)
subchain_values <- scan(args[4], quiet=TRUE)
subchain_dims <- as.integer(scan(args[5], quiet=TRUE))

x_real_names <- grep("^x[0-9]+_real$", names(frequency_data), value=TRUE)
x_imag_names <- grep("^x[0-9]+_imag$", names(frequency_data), value=TRUE)
n_exog <- length(x_real_names)

x_real <- as.matrix(frequency_data[, x_real_names, drop=FALSE])
x_imag <- as.matrix(frequency_data[, x_imag_names, drop=FALSE])
beta <- params[3:(2+n_exog)]
partial1 <- params[1]
partial2 <- params[2]
phi1 <- partial1 * (1-partial2)
phi2 <- partial2
sigma2 <- exp(params[length(params)])

residual_real <- frequency_data$y_real - as.vector(x_real %*% beta)
residual_imag <- frequency_data$y_imag - as.vector(x_imag %*% beta)
periodogram <- (residual_real^2 + residual_imag^2) /
               (2*pi*as.numeric(args[6]))

omega <- frequency_data$omega
denominator_real <- 1 - phi1*cos(omega) - phi2*cos(2*omega)
denominator_imag <- phi1*sin(omega) + phi2*sin(2*omega)
density <- (sigma2/(2*pi)) /
           (denominator_real^2 + denominator_imag^2)
log_likelihood <- -sum(log(density) + periodogram/density)

subchain <- array(subchain_values, dim=subchain_dims)
r_consensus <- consensusMCcov(subchain)

write(format(log_likelihood, digits=17, scientific=TRUE), file=args[7])
write.table(
    t(r_consensus),
    file=args[8],
    row.names=FALSE,
    col.names=FALSE,
    sep=",",
    quote=FALSE
)
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r-consensus",
        type=Path,
        default=None,
        help="path to the original consensusMCcov.R",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-10)
    return parser.parse_args()


def resolve_r_source(requested: Path | None) -> Path:
    if requested is not None:
        if not requested.is_file():
            raise FileNotFoundError(requested)
        return requested
    for candidate in DEFAULT_R_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "consensusMCcov.R was not found; pass its path with --r-consensus"
    )


def make_frequency_case(rng: np.random.Generator):
    n_obs = 128
    x = rng.normal(size=n_obs)
    innovation = rng.normal(size=n_obs)
    residual = np.zeros(n_obs)
    phi = np.array([0.4, -0.6])
    for index in range(2, n_obs):
        residual[index] = (
            phi[0] * residual[index - 1]
            + phi[1] * residual[index - 2]
            + innovation[index]
        )
    y = 0.7 * x + residual

    x_hat = np.fft.fft(x)
    y_hat = np.fft.fft(y)
    indices = np.arange(1, (n_obs - 1) // 2 + 1)
    omega = 2.0 * np.pi * indices / n_obs
    frame = pd.DataFrame(
        {
            "omega": omega,
            "y_real": y_hat[indices].real,
            "y_imag": y_hat[indices].imag,
            "x1_real": x_hat[indices].real,
            "x1_imag": x_hat[indices].imag,
        }
    )

    # These are partial autocorrelations, beta, and log(sigma2).
    params = np.array([0.25, -0.30, 0.70, np.log(1.20)])
    return n_obs, frame, params


def make_consensus_case(rng: np.random.Generator) -> np.ndarray:
    groups = 4
    samples = 600
    dimension = 4
    base_covariance = np.array(
        [
            [1.00, 0.20, 0.05, 0.00],
            [0.20, 0.80, 0.10, 0.02],
            [0.05, 0.10, 1.20, 0.15],
            [0.00, 0.02, 0.15, 0.60],
        ]
    )
    chains = []
    for group in range(groups):
        mean = np.array([0.1, -0.2, 0.7, 0.0]) + 0.02 * group
        covariance = base_covariance * (1.0 + 0.1 * group)
        chains.append(rng.multivariate_normal(mean, covariance, size=samples))
    return np.stack(chains)


def main() -> None:
    args = parse_args()
    r_source = resolve_r_source(args.r_consensus)
    rng = np.random.default_rng(args.seed)
    n_obs, frequency_frame, params = make_frequency_case(rng)
    shard_draws = make_consensus_case(rng)

    python_likelihood = float(
        regression_whittle_log_likelihood(
            params,
            frequency_frame["y_real"].to_numpy(),
            frequency_frame["y_imag"].to_numpy(),
            frequency_frame[["x1_real"]].to_numpy(),
            frequency_frame[["x1_imag"]].to_numpy(),
            frequency_frame["omega"].to_numpy(),
            n_obs,
            1,
        )
    )
    python_consensus = consensus_draws(shard_draws)

    with tempfile.TemporaryDirectory(prefix="dts_ar2_r_parity_") as temp_name:
        temp = Path(temp_name)
        driver_path = temp / "driver.R"
        frequency_path = temp / "frequency.csv"
        params_path = temp / "params.txt"
        subchains_path = temp / "subchains.txt"
        dimensions_path = temp / "dimensions.txt"
        likelihood_path = temp / "likelihood.txt"
        consensus_path = temp / "consensus.csv"

        driver_path.write_text(R_DRIVER)
        frequency_frame.to_csv(frequency_path, index=False)
        np.savetxt(params_path, params, fmt="%.17g")

        # R expects dimensions [parameter, sample, group] in column-major order.
        r_subchains = np.transpose(shard_draws, (2, 1, 0))
        np.savetxt(
            subchains_path,
            r_subchains.ravel(order="F"),
            fmt="%.17g",
        )
        np.savetxt(
            dimensions_path,
            np.asarray(r_subchains.shape, dtype=int),
            fmt="%d",
        )

        subprocess.run(
            [
                "Rscript",
                str(driver_path),
                str(r_source),
                str(frequency_path),
                str(params_path),
                str(subchains_path),
                str(dimensions_path),
                str(n_obs),
                str(likelihood_path),
                str(consensus_path),
            ],
            check=True,
        )
        r_likelihood = float(likelihood_path.read_text().strip())
        r_consensus = np.loadtxt(consensus_path, delimiter=",")

    np.testing.assert_allclose(
        python_likelihood,
        r_likelihood,
        rtol=args.rtol,
        atol=args.atol,
    )
    np.testing.assert_allclose(
        python_consensus,
        r_consensus,
        rtol=args.rtol,
        atol=args.atol,
    )

    print("PASS: DLR-AR(2) Whittle likelihood matches independent R formula")
    print(f"  Python: {python_likelihood:.15g}")
    print(f"  R:      {r_likelihood:.15g}")
    print("PASS: Consensus draws match original consensusMCcov.R")
    print(f"  maximum absolute difference: {np.max(np.abs(python_consensus-r_consensus)):.3e}")
    print(f"  original R source: {r_source}")


if __name__ == "__main__":
    main()
