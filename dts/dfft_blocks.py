"""Distributed FFT using NumPy-array block records.

Input records are ``(block_id, values)`` pairs. There must be ``P`` records,
each containing a contiguous block of ``T / P`` time-domain values. Output
records have the form ``(block_id, (frequency_offset, values))``. Each output
array has shape ``(P, width)`` and satisfies

``values[r, j] == FFT(x)[r * (T / P) + frequency_offset + j]``.

This block-cyclic output layout avoids a third shuffle solely to globally sort
the frequency coefficients.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

import numpy as np


def _frequency_bounds(block_id: int, block_size: int, partitions: int) -> tuple[int, int]:
    return (
        block_id * block_size // partitions,
        (block_id + 1) * block_size // partitions,
    )


def spark_fft_contiguous_blocks(
    block_rdd: Any,
    series_length: int,
    partitions: int,
) -> Any:
    """Compute an unnormalised forward FFT from contiguous array blocks.

    Parameters
    ----------
    block_rdd
        RDD containing exactly one ``(block_id, one_dimensional_array)`` record
        for every block id in ``range(partitions)``.
    series_length
        Total number of time-domain observations.
    partitions
        Number of input and output block records.

    Returns
    -------
    RDD
        Block-cyclic frequency records described in the module docstring.
    """
    if series_length <= 0:
        raise ValueError("series_length must be positive")
    if partitions <= 0:
        raise ValueError("partitions must be positive")
    if series_length % partitions:
        raise ValueError("partitions must divide series_length")

    block_size = series_length // partitions

    def split_contiguous_block(
        record: tuple[int, Any],
    ) -> Iterator[tuple[int, tuple[int, np.ndarray]]]:
        source_block, source_values = record
        source_block = int(source_block)
        if source_block < 0 or source_block >= partitions:
            raise ValueError(f"input block id out of range: {source_block}")
        values = np.asarray(source_values)
        if values.ndim != 1 or values.size != block_size:
            raise ValueError(
                f"input block {source_block} must contain {block_size} values"
            )

        global_start = source_block * block_size
        start_residue = global_start % partitions
        for strided_block in range(partitions):
            local_offset = (strided_block - start_residue) % partitions
            if local_offset >= block_size:
                continue
            strided_start = (global_start + local_offset - strided_block) // partitions
            chunk = np.ascontiguousarray(values[local_offset::partitions])
            yield strided_block, (strided_start, chunk)

    strided_chunks = block_rdd.flatMap(split_contiguous_block)

    def local_fft(
        grouped: tuple[int, Iterable[tuple[int, np.ndarray]]],
    ) -> tuple[int, np.ndarray]:
        strided_block, chunks = grouped
        sequence = np.empty(block_size, dtype=np.complex128)
        cursor = 0
        for start, chunk in sorted(chunks, key=lambda item: item[0]):
            chunk = np.asarray(chunk)
            if start != cursor or chunk.ndim != 1:
                raise ValueError(f"invalid or overlapping chunks for block {strided_block}")
            end = start + chunk.size
            if end > block_size:
                raise ValueError(f"chunk exceeds block {strided_block}")
            sequence[start:end] = chunk
            cursor = end
        if cursor != block_size:
            raise ValueError(f"incomplete input for block {strided_block}")

        transformed = np.fft.fft(sequence)
        if strided_block:
            frequencies = np.arange(block_size, dtype=np.float64)
            transformed *= np.exp(
                (-2j * np.pi * strided_block / series_length) * frequencies
            )
        return int(strided_block), transformed

    first_stage = strided_chunks.groupByKey(numPartitions=partitions).map(local_fft)

    def split_frequency_block(
        record: tuple[int, np.ndarray],
    ) -> Iterator[tuple[int, tuple[int, int, np.ndarray]]]:
        strided_block, transformed = record
        for output_block in range(partitions):
            start, end = _frequency_bounds(output_block, block_size, partitions)
            if start == end:
                continue
            yield output_block, (
                int(strided_block),
                start,
                np.ascontiguousarray(transformed[start:end]),
            )

    frequency_chunks = first_stage.flatMap(split_frequency_block)

    def finish_fft(
        grouped: tuple[int, Iterable[tuple[int, int, np.ndarray]]],
    ) -> tuple[int, tuple[int, np.ndarray]]:
        output_block, chunks = grouped
        expected_start, expected_end = _frequency_bounds(
            output_block, block_size, partitions
        )
        width = expected_end - expected_start
        matrix = np.empty((partitions, width), dtype=np.complex128)
        seen: set[int] = set()
        for strided_block, start, chunk in chunks:
            chunk = np.asarray(chunk)
            if (
                strided_block in seen
                or strided_block < 0
                or strided_block >= partitions
                or start != expected_start
                or chunk.ndim != 1
                or chunk.size != width
            ):
                raise ValueError(f"invalid frequency chunk for output block {output_block}")
            matrix[strided_block, :] = chunk
            seen.add(strided_block)
        if len(seen) != partitions:
            raise ValueError(f"incomplete frequency block {output_block}")

        return int(output_block), (expected_start, np.fft.fft(matrix, axis=0))

    return frequency_chunks.groupByKey(numPartitions=partitions).map(finish_fft)


def collect_block_spectrum(
    spectrum_rdd: Any,
    series_length: int,
    partitions: int,
) -> np.ndarray:
    """Collect and order a block FFT result for testing or small outputs."""
    if series_length % partitions:
        raise ValueError("partitions must divide series_length")
    block_size = series_length // partitions
    spectrum = np.empty(series_length, dtype=np.complex128)
    seen: set[int] = set()

    for output_block, payload in spectrum_rdd.collect():
        output_block = int(output_block)
        if output_block in seen or output_block < 0 or output_block >= partitions:
            raise ValueError(f"duplicate or invalid output block: {output_block}")
        frequency_start, values = payload
        expected_start, expected_end = _frequency_bounds(
            output_block, block_size, partitions
        )
        values = np.asarray(values)
        if (
            frequency_start != expected_start
            or values.shape != (partitions, expected_end - expected_start)
        ):
            raise ValueError(f"invalid output block shape: {output_block}")
        for radix_frequency in range(partitions):
            start = radix_frequency * block_size + expected_start
            spectrum[start : start + values.shape[1]] = values[radix_frequency, :]
        seen.add(output_block)

    expected_blocks = min(partitions, block_size)
    if len(seen) != expected_blocks:
        raise ValueError(f"received {len(seen)} output blocks; expected {expected_blocks}")
    return spectrum
