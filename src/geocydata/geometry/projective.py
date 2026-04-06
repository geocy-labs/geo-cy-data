"""Projective geometry helpers."""

from __future__ import annotations

import numpy as np


def normalize_homogeneous(points: np.ndarray) -> np.ndarray:
    """Normalize homogeneous coordinates to unit Euclidean norm."""

    norms = np.linalg.norm(points, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize points with zero norm.")
    return points / norms


def hermitian_outer(point: np.ndarray) -> np.ndarray:
    """Return the normalized Hermitian outer product for a projective point."""

    norm_sq = float(np.vdot(point, point).real)
    if norm_sq == 0:
        raise ValueError("Cannot compute invariants for the zero vector.")
    return np.outer(point, np.conjugate(point)) / norm_sq


def complex_rescaling(rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample nonzero complex rescaling factors."""

    magnitudes = rng.uniform(0.25, 2.0, size=size)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=size)
    return magnitudes * np.exp(1j * phases)

