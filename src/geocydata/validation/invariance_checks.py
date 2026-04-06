"""Projective invariance validation checks."""

from __future__ import annotations

import numpy as np

from geocydata.features.invariants import invariant_matrix
from geocydata.geometry.projective import complex_rescaling


def summarize_invariant_drift(
    points: np.ndarray, rng: np.random.Generator
) -> dict[str, float]:
    """Measure invariant drift under random complex rescaling."""

    scales = complex_rescaling(rng, size=points.shape[0])
    drifts = np.zeros(points.shape[0], dtype=float)
    for idx, scale in enumerate(scales):
        baseline = invariant_matrix(points[idx])
        rescaled = invariant_matrix(points[idx] * scale)
        drifts[idx] = float(np.max(np.abs(baseline - rescaled)))
    return {
        "max": float(np.max(drifts)),
        "mean": float(np.mean(drifts)),
    }

