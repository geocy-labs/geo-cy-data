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
        coords.append(points_df[real_column].to_numpy(dtype=np.float64) + 1j * points_df[imag_column].to_numpy(dtype=np.float64))
    return np.column_stack(coords)


def fubini_study_scalar(points_df: pd.DataFrame) -> np.ndarray:
    """Compute an ambient Fubini-Study determinant proxy from affine chart coordinates.

    For an affine chart in projective space P^3 with local coordinates w, the determinant of
    the ambient Fubini-Study metric scales like (1 + ||w||^2)^-(n+1) with n = 3. We use this
    simple scalar proxy as a geometry-derived supervised target for Phase 5 experiments.
    """

    affine = affine_coordinate_matrix(points_df)
    norm_sq = np.sum(np.abs(affine) ** 2, axis=1)
    return np.power(1.0 + norm_sq, -4.0, dtype=np.float64)
