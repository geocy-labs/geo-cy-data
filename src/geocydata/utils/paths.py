"""Filesystem path helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory path if it does not already exist."""

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target

