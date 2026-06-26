"""Progress-bar helpers with a no-op fallback."""

from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


class _NullProgress:
    def __init__(self, iterable: Iterable[T] | None = None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(()) if self.iterable is None else iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def update(self, n: int = 1) -> None:
        return None

    def set_postfix(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def progress_bar(iterable: Iterable[T] | None = None, **kwargs):
    """Return a tqdm progress bar, or a no-op iterator when tqdm is unavailable."""
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _NullProgress(iterable, **kwargs)
    return tqdm(iterable, **kwargs)
