"""Projective invariant feature generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from geocydata.geometry.projective import hermitian_outer


def invariant_matrix(point: np.ndarray) -> np.ndarray:
    """Compute the projective invariant matrix for one point."""

    return hermitian_outer(point)


def flatten_invariant_matrix(matrix: np.ndarray) -> dict[str, float]:
    """Flatten real and imaginary parts of a square complex matrix."""

    row: dict[str, float] = {}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            row[f"m_{i}_{j}_re"] = float(matrix[i, j].real)
            row[f"m_{i}_{j}_im"] = float(matrix[i, j].imag)
    return row


def build_invariants_dataframe(points: np.ndarray) -> pd.DataFrame:
    """Build the invariant feature table for a batch of points."""

    rows: list[dict[str, float | int]] = []
    for point_id, point in enumerate(points):
        row: dict[str, float | int] = {"point_id": point_id}
        row.update(flatten_invariant_matrix(invariant_matrix(point)))
        rows.append(row)
    return pd.DataFrame(rows)

