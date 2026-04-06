"""Hypersurface evaluation helpers."""

from __future__ import annotations

import numpy as np


def fermat_quartic_polynomial(points: np.ndarray) -> np.ndarray:
    """Evaluate z0^4 + z1^4 + z2^4 + z3^4 on a batch of points."""

    return np.sum(points**4, axis=1)


def cefalu_quartic_polynomial(points: np.ndarray, lambda_value: float) -> np.ndarray:
    """Evaluate the parameterized Cefalu quartic family on a batch of points."""

    squared_sum = np.sum(points**2, axis=1)
    return fermat_quartic_polynomial(points) - (lambda_value / 3.0) * (squared_sum**2)


def evaluate_hypersurface(
    points: np.ndarray,
    *,
    geometry_name: str,
    parameters: dict[str, object] | None = None,
) -> np.ndarray:
    """Evaluate a supported hypersurface family on a batch of points."""

    parameters = parameters or {}
    if geometry_name == "fermat_quartic":
        return fermat_quartic_polynomial(points)
    if geometry_name == "cefalu_quartic":
        lambda_value = parameters.get("lambda")
        if lambda_value is None:
            raise ValueError("Cefalu quartic evaluation requires a 'lambda' parameter.")
        return cefalu_quartic_polynomial(points, float(lambda_value))
    raise ValueError(f"Unsupported geometry '{geometry_name}'.")


def hypersurface_residuals(
    points: np.ndarray,
    *,
    geometry_name: str,
    parameters: dict[str, object] | None = None,
) -> np.ndarray:
    """Compute absolute residuals for a supported hypersurface family."""

    return np.abs(evaluate_hypersurface(points, geometry_name=geometry_name, parameters=parameters))

