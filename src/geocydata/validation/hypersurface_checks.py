"""Hypersurface validation checks."""

from __future__ import annotations

import numpy as np

from geocydata.geometry.hypersurfaces import hypersurface_residuals


def summarize_residuals(
    points: np.ndarray,
    *,
    geometry_name: str,
    parameters: dict[str, object] | None = None,
) -> dict[str, float]:
    """Summarize hypersurface residuals for a batch of points."""

    residuals = hypersurface_residuals(
        points,
        geometry_name=geometry_name,
        parameters=parameters,
    )
    return {
        "max": float(np.max(residuals)),
        "mean": float(np.mean(residuals)),
    }

