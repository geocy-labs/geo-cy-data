"""Validation report generation."""

from __future__ import annotations

from collections import Counter

import numpy as np

from geocydata.geometry.charts import select_affine_charts
from geocydata.utils.seeds import make_rng
from geocydata.validation.hypersurface_checks import summarize_residuals
from geocydata.validation.invariance_checks import summarize_invariant_drift


def build_validation_report(
    points: np.ndarray,
    *,
    n_points: int,
    seed: int | None,
    residual_tol: float = 1e-8,
    drift_tol: float = 1e-10,
) -> dict:
    """Construct a dataset-level validation report."""

    rng = make_rng(None if seed is None else seed + 1)
    residuals = summarize_residuals(points)
    drifts = summarize_invariant_drift(points, rng=rng)
    chart_ids, _ = select_affine_charts(points)
    chart_distribution = dict(sorted(Counter(chart_ids.tolist()).items()))
    warnings: list[str] = []

    if residuals["max"] > residual_tol:
        warnings.append(
            "Hypersurface residual exceeds tolerance; sampler is a smoke-test approximation."
        )
    if drifts["max"] > drift_tol:
        warnings.append("Invariant drift exceeds tolerance under random projective rescaling.")

    return {
        "n_points": n_points,
        "residual": residuals,
        "invariant_drift": drifts,
        "chart_distribution": chart_distribution,
        "tolerances": {"residual_max": residual_tol, "invariant_drift_max": drift_tol},
        "warnings": warnings,
        "passed": not warnings,
    }

