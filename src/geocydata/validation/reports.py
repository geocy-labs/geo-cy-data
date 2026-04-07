"""Validation report generation."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from geocydata.geometry.charts import select_affine_charts
from geocydata.utils.seeds import make_rng
from geocydata.validation.geometry_hooks import build_evaluation_summary
from geocydata.validation.hypersurface_checks import summarize_residuals
from geocydata.validation.invariance_checks import summarize_invariant_drift


def build_validation_report(
    points: np.ndarray,
    *,
    geometry_name: str,
    parameters: dict[str, object] | None,
    n_points: int,
    seed: int | None,
    points_df: pd.DataFrame | None = None,
    residual_tol: float = 1e-8,
    drift_tol: float = 1e-10,
) -> dict:
    """Construct a dataset-level validation report."""

    rng = make_rng(None if seed is None else seed + 1)
    resolved_parameters = parameters or {}
    residuals = summarize_residuals(
        points,
        geometry_name=geometry_name,
        parameters=resolved_parameters,
    )
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
    evaluation_summary = None
    if points_df is not None:
        evaluation_summary = build_evaluation_summary(
            points=points,
            points_df=points_df,
            geometry_name=geometry_name,
            parameters=resolved_parameters,
            seed=seed,
        )

    return {
        "geometry": geometry_name,
        "parameters": resolved_parameters,
        "n_points": n_points,
        "residual": residuals,
        "invariant_drift": drifts,
        "chart_distribution": chart_distribution,
        "geometry_evaluation_hooks": evaluation_summary,
        "tolerances": {"residual_max": residual_tol, "invariant_drift_max": drift_tol},
        "warnings": warnings,
        "passed": not warnings,
    }

