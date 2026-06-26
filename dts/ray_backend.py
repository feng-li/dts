"""Optional Ray helpers for shard-level parallel execution."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import os
from typing import Any

from dts.progress import progress_bar


def effective_num_cpus(num_cpus: int | None) -> int:
    """Return the CPU count used for automatic shard backend selection."""
    if num_cpus is None:
        return max(1, os.cpu_count() or 1)
    value = int(num_cpus)
    if value < 1:
        raise ValueError("--num-cpus must be at least 1")
    return value


def resolve_shard_backend(backend: str, num_cpus: int | None) -> str:
    """Resolve the requested shard backend into ``local`` or ``ray``."""
    if backend == "auto":
        return "ray" if effective_num_cpus(num_cpus) > 1 else "local"
    if backend in {"local", "ray"}:
        return backend
    raise ValueError(f"unknown shard backend: {backend!r}")


def _require_ray():
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            "Ray backend selected but ray is not installed. Install project "
            "dependencies with `python -m pip install -e .` or install Ray with "
            "`python -m pip install ray`."
        ) from exc
    return ray


def run_indexed_ray_tasks(
    task_fn: Callable[[Any], tuple[int, Any]],
    payloads: Sequence[Any],
    *,
    address: str | None = None,
    num_cpus: int | None = None,
    task_num_cpus: float | None = None,
    progress: bool = False,
    desc: str = "ray tasks",
) -> list[Any]:
    """Run top-level indexed tasks with Ray and return results in input order."""
    if not payloads:
        return []

    ray = _require_ray()
    if not ray.is_initialized():
        init_kwargs = {
            "address": address,
            "ignore_reinit_error": True,
            "include_dashboard": False,
        }
        if address is None:
            init_kwargs["_node_ip_address"] = "127.0.0.1"
            if num_cpus is not None:
                init_kwargs["num_cpus"] = effective_num_cpus(num_cpus)
        ray.init(**init_kwargs)

    remote_options = {}
    if task_num_cpus is not None:
        remote_options["num_cpus"] = task_num_cpus
    remote_task = ray.remote(**remote_options)(task_fn) if remote_options else ray.remote(task_fn)

    refs = [remote_task.remote(payload) for payload in payloads]
    pending = list(refs)
    results: dict[int, Any] = {}
    with progress_bar(
        total=len(refs),
        desc=desc,
        unit="task",
        leave=False,
        disable=not progress,
    ) as bar:
        while pending:
            done, pending = ray.wait(pending, num_returns=1)
            for ref in done:
                index, result = ray.get(ref)
                results[index] = result
                bar.update(1)

    return [results[index] for index in range(len(refs))]
