"""Path resolution helpers."""

import os


def ensure_dirs(*dirs):
    """Create directories if they do not exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def experiment_dir(phase: str, tag: str) -> str:
    """Return results/<phase>/<tag> and ensure it exists."""
    d = os.path.join('results', phase, tag)
    os.makedirs(d, exist_ok=True)
    return d


def checkpoint_dir(tag: str) -> str:
    """Return checkpoints/<tag> and ensure it exists."""
    d = os.path.join('checkpoints', tag)
    os.makedirs(d, exist_ok=True)
    return d
