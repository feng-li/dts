#!/usr/bin/env python3
"""Generate shared CSV data for DLR-AR(2) Spark and Stan comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-observations", type=int, default=2048)
    parser.add_argument("--burn-in", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument("--phi1", type=float, default=0.4)
    parser.add_argument("--phi2", type=float, default=-0.6)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dlr_ar2_small.csv"),
    )
    parser.add_argument(
        "--truth-output",
        type=Path,
        default=None,
        help="default: OUTPUT with _truth.json suffix",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.n_observations < 3:
        raise ValueError("--n-observations must be at least 3")
    if args.burn_in < 0:
        raise ValueError("--burn-in must be nonnegative")
    if args.sigma2 <= 0:
        raise ValueError("--sigma2 must be positive")

    stationary = (
        abs(args.phi2) < 1
        and args.phi1 + args.phi2 < 1
        and args.phi2 - args.phi1 < 1
    )
    if not stationary:
        raise ValueError("the supplied AR(2) coefficients are not stationary")


def simulate(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    total = args.n_observations + args.burn_in
    innovations = rng.normal(0.0, np.sqrt(args.sigma2), total)
    residual = np.zeros(total)

    for index in range(2, total):
        residual[index] = (
            args.phi1 * residual[index - 1]
            + args.phi2 * residual[index - 2]
            + innovations[index]
        )

    residual = residual[args.burn_in :]
    x = rng.normal(size=args.n_observations)
    y = args.beta * x + residual
    return pd.DataFrame(
        {
            "time": np.arange(args.n_observations),
            "x": x,
            "y": y,
        }
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    truth_output = args.truth_output or args.output.with_name(
        f"{args.output.stem}_truth.json"
    )

    data = simulate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    truth_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)

    truth = {
        "n_observations": args.n_observations,
        "beta": args.beta,
        "phi1": args.phi1,
        "phi2": args.phi2,
        "sigma2": args.sigma2,
        "seed": args.seed,
        "burn_in": args.burn_in,
        "intercept": False,
    }
    with truth_output.open("w") as handle:
        json.dump(truth, handle, indent=2)

    print(f"Generated {args.output}")
    print(f"Generated {truth_output}")


if __name__ == "__main__":
    main()
