"""Spark grouped-map helpers for shard-level MCMC."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from dts.experiments import MCMCSettings, fit_whittle_shard
from dts.mcmc import ModelSpec


def mapper(pdf: pd.DataFrame, conf_model: Dict[str, Any], conf_mcmc: Dict[str, int]) -> pd.DataFrame:
    """Run Whittle MCMC for one Spark shard.

    Expected input columns are ``shard_id``, ``periodogram`` and ``omega`` as
    produced by :func:`dts.dfft.spark_periodogram_dataframe`.
    """
    required = {"shard_id", "periodogram", "omega"}
    missing = required.difference(pdf.columns)
    if missing:
        raise ValueError(f"missing Spark mapper columns: {sorted(missing)}")

    group_id = int(pdf["shard_id"].iat[0])
    model = ModelSpec(
        ar_order=int(conf_model["ar_order"]),
        ma_order=int(conf_model["ma_order"]),
        tfi_term=bool(conf_model.get("TFI_term", False)),
        exact=False,
    )
    settings = MCMCSettings(
        n_samples=int(conf_mcmc["n_samples"]),
        burn_in=int(conf_mcmc["Burn_in"]),
        seed=int(conf_mcmc.get("seed", 123)),
        optimize=bool(conf_mcmc.get("optimize", True)),
        basinhopping=bool(conf_mcmc.get("basinhopping", False)),
        basinhopping_iter=int(conf_mcmc.get("basinhopping_iter", 25)),
    )
    n_groups = int(conf_model["partition_num"])

    result = fit_whittle_shard(
        periodogram=pdf["periodogram"].to_numpy(dtype=float),
        omega=pdf["omega"].to_numpy(dtype=float),
        model=model,
        settings=settings,
        n_groups=n_groups,
        group_id=group_id,
    )

    return pd.DataFrame(
        {
            "shard_id": [group_id],
            "samples": [result.draws.tolist()],
            "map_estimate": [np.asarray(result.map_estimate, dtype=float).tolist()],
            "log_p": [float(result.log_p[-1]) if len(result.log_p) else np.nan],
            "acceptance_rate": [result.acceptance_rate],
        }
    )
