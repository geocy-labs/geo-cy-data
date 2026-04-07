"""Cefalu quartic family benchmark geometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geocydata.geometry.hypersurfaces import hypersurface_residuals
from geocydata.geometry.projective import normalize_homogeneous


@dataclass(frozen=True)
class CefaluQuarticGeometry:
    """Parameterized Cefalu quartic family in projective space P^3."""

    name: str = "cefalu_quartic"
    ambient_dimension: int = 3
    equation: str = (
        "z0^4 + z1^4 + z2^4 + z3^4 - (lambda / 3) * (z0^2 + z1^2 + z2^2 + z3^2)^2 = 0"
    )
    description: str = (
        "Initial branch-based smoke-test sampler for the parameterized Cefalu quartic family in P^3."
    )
    parameter_schema: dict[str, str] = field(
        default_factory=lambda: {
            "lambda": "Required float parameter for the Cefalu quartic family.",
        }
    )

    def validate_parameters(self, parameters: dict[str, object] | None) -> dict[str, float]:
        """Validate and normalize geometry parameters."""

        parameters = parameters or {}
        if parameters.get("lambda") is None:
            raise ValueError("Geometry 'cefalu_quartic' requires '--lambda'.")
        return {"lambda": float(parameters["lambda"])}

    def metadata(self) -> dict[str, object]:
        """Return public geometry metadata for CLI display."""

        return {
            "name": self.name,
            "ambient_dimension": self.ambient_dimension,
            "equation": self.equation,
            "description": self.description,
            "parameter_schema": self.parameter_schema,
        }

    def sample_points(
        self,
        n: int,
        rng: np.random.Generator,
        parameters: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Sample points by solving the induced quartic in z3 for each sampled base point."""

        if n <= 0:
            raise ValueError("n must be positive.")

        resolved = self.validate_parameters(parameters)
        lambda_value = resolved["lambda"]

        z0 = rng.normal(size=n) + 1j * rng.normal(size=n)
        z1 = rng.normal(size=n) + 1j * rng.normal(size=n)
        z2 = rng.normal(size=n) + 1j * rng.normal(size=n)

        roots = np.zeros(n, dtype=np.complex128)
        branches = rng.integers(0, 4, size=n)
        leading = 1.0 - lambda_value / 3.0

        for idx in range(n):
            quartic_sum = z0[idx] ** 4 + z1[idx] ** 4 + z2[idx] ** 4
            square_sum = z0[idx] ** 2 + z1[idx] ** 2 + z2[idx] ** 2
            coeffs = [
                leading,
                0.0,
                -(2.0 * lambda_value / 3.0) * square_sum,
                0.0,
                quartic_sum - (lambda_value / 3.0) * (square_sum**2),
            ]
            trimmed = np.array(coeffs, dtype=np.complex128)
            while len(trimmed) > 1 and np.isclose(trimmed[0], 0.0):
                trimmed = trimmed[1:]
            candidate_roots = np.roots(trimmed)
            ordered = candidate_roots[np.argsort(np.angle(candidate_roots))]
            roots[idx] = ordered[int(branches[idx]) % len(ordered)]

        points = np.column_stack([z0, z1, z2, roots]).astype(np.complex128)
        return normalize_homogeneous(points)

    def residuals(
        self,
        points: np.ndarray,
        parameters: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Evaluate absolute hypersurface residuals for sampled points."""

        resolved = self.validate_parameters(parameters)
        return hypersurface_residuals(
            points,
            geometry_name=self.name,
            parameters=resolved,
        )
