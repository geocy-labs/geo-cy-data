"""Geometry-aware evaluation summaries for bundle exports."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocydata.baselines.fubini_study import (
    affine_coordinate_matrix,
    ambient_fubini_study_metric,
    fubini_study_scalar,
    hypersurface_fubini_study_scalar,
)
from geocydata.geometry.charts import select_affine_charts
from geocydata.utils.seeds import make_rng
from geocydata.validation.invariance_checks import summarize_invariant_drift
from geocydata.validation.symmetry_checks import build_symmetry_report


def chart_consistency_summary(points: np.ndarray) -> dict[str, object]:
    """Summarize the chosen affine-chart pivots for one point cloud."""

    chart_ids, _ = select_affine_charts(points)
    pivot_magnitudes = np.abs(points[np.arange(points.shape[0]), chart_ids])
    return {
        "selected_chart_ids": sorted(set(int(value) for value in chart_ids.tolist())),
        "pivot_abs_min": float(np.min(pivot_magnitudes)),
        "pivot_abs_mean": float(np.mean(pivot_magnitudes)),
        "pivot_abs_max": float(np.max(pivot_magnitudes)),
    }


def positivity_eigenvalue_summary(
    points_df: pd.DataFrame,
    *,
    geometry_name: str,
    parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    """Summarize positivity and eigenvalue behavior for the baseline geometry proxies."""

    affine = affine_coordinate_matrix(points_df)
    min_eigenvalues: list[float] = []
    max_eigenvalues: list[float] = []
    for row in affine:
        eigenvalues = np.linalg.eigvalsh(ambient_fubini_study_metric(row))
        min_eigenvalues.append(float(np.min(eigenvalues.real)))
        max_eigenvalues.append(float(np.max(eigenvalues.real)))

    hypersurface_scalars = hypersurface_fubini_study_scalar(
        points_df,
        geometry_name=geometry_name,
        parameters=parameters,
    )
    nonpositive = int(np.sum(hypersurface_scalars <= 0.0))
    return {
        "ambient_min_eigenvalue_min": float(np.min(min_eigenvalues)),
        "ambient_min_eigenvalue_mean": float(np.mean(min_eigenvalues)),
        "ambient_max_eigenvalue_mean": float(np.mean(max_eigenvalues)),
        "nonpositive_hypersurface_scalar_count": nonpositive,
        "hypersurface_scalar_min": float(np.min(hypersurface_scalars)),
        "hypersurface_scalar_mean": float(np.mean(hypersurface_scalars)),
        "passed": nonpositive == 0 and float(np.min(min_eigenvalues)) > 0.0,
    }


def characteristic_form_euler_summary(
    points_df: pd.DataFrame,
    *,
    geometry_name: str,
    parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return lightweight characteristic-form and Euler hooks for downstream analysis.

    These are baseline proxies, not full characteristic-form integrations.
    """

    ambient = fubini_study_scalar(points_df)
    hypersurface = hypersurface_fubini_study_scalar(
        points_df,
        geometry_name=geometry_name,
        parameters=parameters,
    )
    return {
        "ambient_fs_scalar_mean": float(np.mean(ambient)),
        "ambient_fs_scalar_std": float(np.std(ambient)),
        "hypersurface_fs_scalar_mean": float(np.mean(hypersurface)),
        "hypersurface_fs_scalar_std": float(np.std(hypersurface)),
        "euler_characteristic_reference": 24,
        "euler_summary_kind": "quartic_k3_reference",
        "note": (
            "This is a lightweight export hook for downstream GlobalCY evaluation. "
            "It records baseline scalar summaries plus the quartic K3 Euler reference, "
            "not a full numerical characteristic-form integration."
        ),
    }


def build_evaluation_summary(
    *,
    points: np.ndarray,
    points_df: pd.DataFrame,
    geometry_name: str,
    parameters: dict[str, object] | None,
    seed: int | None,
) -> dict[str, object]:
    """Build lightweight geometry-aware evaluation hooks for a bundle export."""

    resolved_parameters = parameters or {}
    rng = make_rng(None if seed is None else seed + 17)
    projective_drift = summarize_invariant_drift(points, rng=rng)
    symmetry = None
    if geometry_name == "cefalu_quartic" and resolved_parameters.get("lambda") is not None:
        symmetry = build_symmetry_report(points, lambda_value=float(resolved_parameters["lambda"]))
    return {
        "geometry": geometry_name,
        "parameters": resolved_parameters,
        "chart_consistency": chart_consistency_summary(points),
        "projective_invariance_drift": projective_drift,
        "symmetry_consistency": symmetry,
        "positivity_eigenvalue_summary": positivity_eigenvalue_summary(
            points_df,
            geometry_name=geometry_name,
            parameters=resolved_parameters,
        ),
        "characteristic_form_euler_summary": characteristic_form_euler_summary(
            points_df,
            geometry_name=geometry_name,
            parameters=resolved_parameters,
        ),
    }
