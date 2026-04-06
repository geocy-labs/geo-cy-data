"""Deterministic canonicalization for symmetry orbits."""

from __future__ import annotations

import numpy as np

from geocydata.geometry.projective import hermitian_outer
from geocydata.symmetry.actions import generate_orbit
from geocydata.symmetry.groups import SymmetryAction


def phase_normalize(point: np.ndarray) -> np.ndarray:
    """Fix a deterministic projective phase by making the largest coordinate real and nonnegative."""

    pivot_index = int(np.argmax(np.abs(point)))
    pivot = point[pivot_index]
    if np.isclose(abs(pivot), 0.0):
        return point
    normalized = point * np.exp(-1j * np.angle(pivot))
    if normalized[pivot_index].real < 0:
        normalized *= -1.0
    return normalized


def canonical_key(point: np.ndarray, decimals: int = 12) -> tuple[float, ...]:
    """Return a lexicographically comparable key for a homogeneous point."""

    normalized = phase_normalize(point)
    rounded = np.round(
        np.concatenate([normalized.real, normalized.imag]),
        decimals=decimals,
    )
    return tuple(float(value) for value in rounded)


def canonical_key_string(point: np.ndarray, decimals: int = 12) -> str:
    """Return a stable string form of the canonical comparison key."""

    return "|".join(f"{value:.{decimals}f}" for value in canonical_key(point, decimals=decimals))


def choose_canonical_representative(
    point: np.ndarray,
    actions: list[SymmetryAction],
) -> tuple[SymmetryAction, np.ndarray]:
    """Choose a deterministic canonical orbit representative."""

    orbit = generate_orbit(point, actions)
    return min(orbit, key=lambda item: canonical_key(item[1]))


def canonical_invariant_matrix(point: np.ndarray, actions: list[SymmetryAction]) -> np.ndarray:
    """Return the invariant matrix of the canonical representative."""

    _, canonical_point = choose_canonical_representative(point, actions)
    return hermitian_outer(phase_normalize(canonical_point))
