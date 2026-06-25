"""JAX runtime configuration used by differentiable DTS modules."""

from __future__ import annotations

import os


def configure_jax() -> None:
    """Configure JAX for deterministic CPU-based scientific computing.

    The replication scripts run on CPU-only machines, so the package asks JAX
    to use the CPU platform and enables 64-bit floating point arithmetic before
    importing ``jax.numpy`` in modules that need automatic differentiation.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)
