"""Path helpers for legacy figure scripts."""

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = REPO_ROOT / "data"


def find_input(filename: str) -> Path:
    """Find an input next to the script, at repo root, or under data/."""

    candidates = [
        Path.cwd() / filename,
        SCRIPT_DIR / filename,
        REPO_ROOT / filename,
        DATA_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {filename!r}. Searched:\n  {searched}"
    )
