"""Fermat quartic benchmark geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geocydata.geometry.hypersurfaces import hypersurface_residuals
from geocydata.geometry.projective import normalize_homogeneous


@dataclass(frozen=True)
class FermatQuarticGeometry:
    """Fermat quartic hypersurface in projective space P^3."""

    name: str = "fermat_quartic"
    ambient_dimension: int = 3
    equation: str = "z0^4 + z1^4 + z2^4 + z3^4 = 0"
    description: str = (
        "Initial branch-based smoke-test sampler for the Fermat quartic hypersurface in P^3."
    )
    parameter_schema: dict[str, str] = None

    def validate_parameters(self, parameters: dict[str, object] | None = None) -> dict[str, object]:
        """Validate that no family parameters were supplied."""

        parameters = parameters or {}
        if parameters.get("lambda") is not None:
            raise ValueError("Geometry 'fermat_quartic' does not accept '--lambda'.")
        return {}

    def metadata(self) -> dict[str, object]:
        """Return public geometry metadata for CLI display."""

        return {
            "name": self.name,
            "ambient_dimension": self.ambient_dimension,
            "equation": self.equation,
            "description": self.description,
            "parameter_schema": {},
        }

    def sample_points(
        self,
        n: int,
        rng: np.random.Generator,
        parameters: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Sample points on the Fermat quartic by solving for one branch of z3."""

        if n <= 0:
            raise ValueError("n must be positive.")
        self.validate_parameters(parameters)

        z0 = rng.normal(size=n) + 1j * rng.normal(size=n)
        z1 = rng.normal(size=n) + 1j * rng.normal(size=n)
        z2 = rng.normal(size=n) + 1j * rng.normal(size=n)

        rhs = -(z0**4 + z1**4 + z2**4)
        branches = rng.integers(0, 4, size=n)
        magnitudes = np.abs(rhs) ** 0.25
        arguments = (np.angle(rhs) + 2.0 * np.pi * branches) / 4.0
        z3 = magnitudes * np.exp(1j * arguments)

        points = np.column_stack([z0, z1, z2, z3]).astype(np.complex128)
        return normalize_homogeneous(points)

    def residuals(
        self,
        points: np.ndarray,
        parameters: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Evaluate absolute hypersurface residuals for sampled points."""

        self.validate_parameters(parameters)
        return hypersurface_residuals(points, geometry_name=self.name, parameters={})

