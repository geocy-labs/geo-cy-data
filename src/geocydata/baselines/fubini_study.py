"""Lightweight Fubini-Study baseline scalars for experiment targets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def affine_coordinate_matrix(points_df: pd.DataFrame) -> np.ndarray:
    """Extract complex affine coordinates from the exported points table."""

    real_columns = sorted(
        column for column in points_df.columns if column.startswith("affine_") and column.endswith("_re")
    )
    if not real_columns:
        raise ValueError("Points table does not contain expected affine coordinate columns.")
    coords: list[np.ndarray] = []
    for real_column in real_columns:
        base = real_column.removesuffix("_re")
        imag_column = f"{base}_im"
        if imag_column not in points_df.columns:
            raise ValueError(f"Points table is missing affine coordinate column: {imag_column}")
        coords.append(
            points_df[real_column].to_numpy(dtype=np.float64)
            + 1j * points_df[imag_column].to_numpy(dtype=np.float64)
        )
    return np.column_stack(coords)


def ambient_fubini_study_metric(affine_row: np.ndarray) -> np.ndarray:
    """Compute the ambient Fubini-Study metric matrix in one affine chart row."""

    norm_sq = float(np.sum(np.abs(affine_row) ** 2))
    identity = np.eye(affine_row.shape[0], dtype=np.complex128)
    outer = np.outer(affine_row, np.conjugate(affine_row))
    return (((1.0 + norm_sq) * identity) - outer) / (1.0 + norm_sq) ** 2


def hypersurface_gradient_affine(
    affine: np.ndarray,
    *,
    geometry_name: str,
    parameters: dict[str, object] | None = None,
) -> np.ndarray:
    """Compute the holomorphic affine-chart gradient of the hypersurface polynomial."""

    resolved_parameters = parameters or {}
    if geometry_name == "fermat_quartic":
        return 4.0 * np.power(affine, 3)
    if geometry_name == "cefalu_quartic":
        lambda_value = float(resolved_parameters.get("lambda", 0.0))
        square_sum = 1.0 + np.sum(np.power(affine, 2), axis=1, keepdims=True)
        return 4.0 * np.power(affine, 3) - ((4.0 * lambda_value) / 3.0) * square_sum * affine
    raise ValueError(f"Unsupported geometry for hypersurface target: {geometry_name}")


def fubini_study_scalar(points_df: pd.DataFrame) -> np.ndarray:
    """Compute an ambient Fubini-Study determinant proxy from affine chart coordinates.

    For an affine chart in projective space P^3 with local coordinates w, the determinant of
    the ambient Fubini-Study metric scales like (1 + ||w||^2)^-(n+1) with n = 3. We use this
    simple scalar proxy as the older geometry-derived supervised target.
    """

    affine = affine_coordinate_matrix(points_df)
    norm_sq = np.sum(np.abs(affine) ** 2, axis=1)
    return np.power(1.0 + norm_sq, -4.0, dtype=np.float64)


def hypersurface_fubini_study_scalar(
    points_df: pd.DataFrame,
    *,
    geometry_name: str,
    parameters: dict[str, object] | None = None,
    singular_tol: float = 1e-12,
) -> np.ndarray:
    """Compute a hypersurface-aware tangent-restricted Fubini-Study scalar proxy.

    This strengthens the older ambient ``fs_scalar`` target by combining the affine
    Fubini-Study metric with the local hypersurface gradient. For a smooth codimension-one
    hypersurface, the tangent-restricted determinant proxy can be written in basis-free form:

    ``det(G|_T) = det(G) * (g G^{-1} g*) / (g g*)``

    where ``G`` is the ambient affine Fubini-Study metric and ``g`` is the holomorphic chart
    gradient of the hypersurface polynomial. When the local gradient is numerically singular,
    we fall back to the ambient proxy for stability.
    """

    affine = affine_coordinate_matrix(points_df)
    gradients = hypersurface_gradient_affine(
        affine,
        geometry_name=geometry_name,
        parameters=parameters,
    )
    ambient_scalar = fubini_study_scalar(points_df)
    values = np.empty(affine.shape[0], dtype=np.float64)

    for idx, (row, gradient_row) in enumerate(zip(affine, gradients, strict=True)):
        gradient_norm_sq = float(np.vdot(gradient_row, gradient_row).real)
        if gradient_norm_sq <= singular_tol:
            values[idx] = ambient_scalar[idx]
            continue

        metric = ambient_fubini_study_metric(row)
        metric_inv = np.linalg.inv(metric)
        correction = np.vdot(gradient_row, metric_inv @ gradient_row) / gradient_norm_sq
        restricted_det = np.linalg.det(metric) * correction
        values[idx] = max(float(np.real_if_close(restricted_det).real), 1e-18)

    return values
