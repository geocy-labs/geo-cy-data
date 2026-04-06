"""Symmetry actions and orbit generation helpers."""

from __future__ import annotations

import numpy as np

from geocydata.geometry.projective import normalize_homogeneous
from geocydata.symmetry.groups import SymmetryAction


def apply_action(point: np.ndarray, action: SymmetryAction) -> np.ndarray:
    """Apply a signed coordinate permutation to a homogeneous point."""

    permuted = point[list(action.permutation)]
    signed = permuted * np.array(action.signs, dtype=np.float64)
    return normalize_homogeneous(signed[np.newaxis, :])[0]


def generate_orbit(point: np.ndarray, actions: list[SymmetryAction]) -> list[tuple[SymmetryAction, np.ndarray]]:
    """Generate the orbit images of a point under the provided actions."""

    return [(action, apply_action(point, action)) for action in actions]

