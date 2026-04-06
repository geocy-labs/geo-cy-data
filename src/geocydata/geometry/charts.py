"""Affine chart extraction utilities."""

from __future__ import annotations

import numpy as np


def select_affine_charts(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Choose the chart with maximal coordinate magnitude and return affine coordinates."""

    chart_ids = np.argmax(np.abs(points), axis=1)
    affine = np.zeros((points.shape[0], points.shape[1] - 1), dtype=np.complex128)
    for row_index, chart_id in enumerate(chart_ids):
        pivot = points[row_index, chart_id]
        remaining = [idx for idx in range(points.shape[1]) if idx != chart_id]
        affine[row_index] = points[row_index, remaining] / pivot
    return chart_ids, affine
