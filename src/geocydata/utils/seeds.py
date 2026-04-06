"""Random seed helpers."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Create a reproducible NumPy random generator."""

    return np.random.default_rng(seed)

