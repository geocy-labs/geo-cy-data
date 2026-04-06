"""Hypersurface validation checks."""

from __future__ import annotations

import numpy as np

from geocydata.geometry.hypersurfaces import hypersurface_residuals


def summarize_residuals(points: np.ndarray) -> dict[str, float]:
    """Summarize hypersurface residuals for a batch of points."""

    residuals = hypersurface_residuals(points)
    return {
        "max": float(np.max(residuals)),
        "mean": float(np.mean(residuals)),
    }

