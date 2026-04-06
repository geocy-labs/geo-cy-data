"""Hypersurface evaluation helpers."""

from __future__ import annotations

import numpy as np


def fermat_quartic_polynomial(points: np.ndarray) -> np.ndarray:
    """Evaluate z0^4 + z1^4 + z2^4 + z3^4 on a batch of points."""

    return np.sum(points**4, axis=1)


def hypersurface_residuals(points: np.ndarray) -> np.ndarray:
    """Compute absolute residuals for the Fermat quartic equation."""

    return np.abs(fermat_quartic_polynomial(points))

