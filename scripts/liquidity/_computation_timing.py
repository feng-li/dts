"""Synchronous instrumentation around the unchanged compiled MH function.

This module is copied into a benchmark's frozen source tree. It observes the
production sampler's arguments and return value; it generates no proposals
and changes neither the transition kernel nor the random stream.
"""
from __future__ import annotations

from contextlib import contextmanager
import resource
import time


@contextmanager
def coordinate_step_timing(coordinates, barrier=None):
    """Measure explicit lowering/compilation and blocking numerical execution."""
    import jax

    original = coordinates._coordinate_steps
    record = {}

    def measured(current, value, increments, log_uniforms, independent, center,
                 independence_precision, pg, basis, groups, lambda_power=.5):
        inputs = (current, value, increments, log_uniforms, independent, center,
                  independence_precision, pg, basis, groups)
        jax.block_until_ready(inputs)
        record['inputs_ready_at'] = time.perf_counter()
        started = time.perf_counter()
        executable = original.lower(*inputs, lambda_power=lambda_power).compile()
        record['loop_compile_seconds'] = time.perf_counter() - started
        if barrier is not None:
            barrier()
        record['sampling_started_at'] = time.perf_counter()
        cpu_started = time.process_time()
        output = executable(*inputs)
        jax.block_until_ready(output)
        record['sampling_cpu_seconds'] = time.process_time() - cpu_started
        record['sampling_finished_at'] = time.perf_counter()
        record['compiled_sampling_seconds'] = (
            record['sampling_finished_at'] - record['sampling_started_at'])
        record['peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return output

    coordinates._coordinate_steps = measured
    try:
        yield record
    finally:
        coordinates._coordinate_steps = original


def critical_path_breakdown(records, action_seconds, parallel):
    """Allocate action wall time along its terminal worker (or serial workers).

    The residual includes dispatch, time before the terminal worker starts,
    result collection, and any uninstrumented worker work. It is an elapsed
    time allocation, not the sum of CPU seconds over parallel workers.
    """
    keys = ('load_seconds', 'native_map_seconds', 'coordinate_map_seconds',
            'sampling_prepare_seconds', 'loop_compile_seconds',
            'compiled_sampling_seconds', 'sampling_extract_seconds',
            'covariance_seconds', 'write_seconds')
    if parallel:
        selected = [max(records, key=lambda item: item['worker_finished_at'])]
    else:
        selected = records
    result = {key: sum(item.get(key, 0.) for item in selected) for key in keys}
    residual = action_seconds - sum(result.values())
    if residual < -1e-4:
        raise ValueError('Worker stages exceed the measured action wall time')
    result['scheduling_wait_other_seconds'] = max(0., residual)
    result['terminal_groups'] = [item['group'] for item in selected]
    return result
