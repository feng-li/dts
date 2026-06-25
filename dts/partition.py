"""Frequency-domain transforms and partitioning utilities."""

from __future__ import annotations

from typing import List

import numpy as np


def positive_frequency_count(series_length: int) -> int:
    """Number of nonzero positive Fourier frequencies for a real series."""
    return int(np.floor((series_length - 1) / 2))


def fourier_frequencies(series_length: int) -> np.ndarray:
    """Positive Fourier frequencies, excluding the zero frequency."""
    n_freq = positive_frequency_count(series_length)
    return 2.0 * np.pi * np.arange(1, n_freq + 1) / series_length


def periodogram(data: np.ndarray) -> np.ndarray:
    """Periodogram at positive Fourier frequencies.

    The original project scripts used the FFT directly; this helper keeps that
    implementation but aligns the indexing with the manuscript, using
    k = 1, ..., floor((T - 1) / 2).
    """
    x = np.asarray(data, dtype=float)
    fft_values = np.fft.fft(x)
    n_freq = positive_frequency_count(len(x))
    return np.square(np.abs(fft_values[1 : n_freq + 1])) / (2.0 * np.pi * len(x))


def frequency_domain(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return periodogram values and matching positive Fourier frequencies."""
    return periodogram(data), fourier_frequencies(len(data))


def frequency_partition_indices(
    n_frequencies: int,
    n_groups: int,
    method: str = "systematic",
) -> List[np.ndarray]:
    """Partition frequency indices into systematic or sequential shards."""
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    if n_frequencies < n_groups:
        raise ValueError("n_groups cannot exceed the number of frequencies")

    idx = np.arange(n_frequencies)
    if method == "systematic":
        return [idx[g::n_groups] for g in range(n_groups)]
    if method == "sequential":
        return [part.astype(int) for part in np.array_split(idx, n_groups)]
    raise ValueError(f"unknown frequency partition method: {method!r}")


def time_partition_indices(series_length: int, n_groups: int) -> List[np.ndarray]:
    """Sequentially partition the time-domain observations."""
    if n_groups < 1:
        raise ValueError("n_groups must be positive")
    if series_length < n_groups:
        raise ValueError("n_groups cannot exceed the series length")
    return [part.astype(int) for part in np.array_split(np.arange(series_length), n_groups)]


def shard_frequency_domain(
    data: np.ndarray,
    n_groups: int,
    method: str = "systematic",
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(indices, periodogram, omega)`` triples for each frequency shard."""
    values, omega = frequency_domain(data)
    groups = frequency_partition_indices(len(values), n_groups, method=method)
    return [(indices, values[indices], omega[indices]) for indices in groups]
