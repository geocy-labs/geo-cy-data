"""Symmetry-orbit validation helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from geocydata.features.invariants import flatten_invariant_matrix
from geocydata.geometry.hypersurfaces import hypersurface_residuals
from geocydata.symmetry.actions import generate_orbit
from geocydata.symmetry.canonicalize import (
    canonical_invariant_matrix,
    canonical_key,
    canonical_key_string,
    choose_canonical_representative,
    phase_normalize,
)
from geocydata.symmetry.groups import cefalu_symmetry_actions


def build_orbits_dataframe(points: np.ndarray, lambda_value: float) -> pd.DataFrame:
    """Build an orbit table for Cefalu quartic sample points."""

    actions = cefalu_symmetry_actions()
    rows: list[dict[str, object]] = []

    for point_id, point in enumerate(points):
        canonical_action, canonical_point = choose_canonical_representative(point, actions)
        canonical_label = canonical_key_string(canonical_point)
        seen_keys: set[tuple[float, ...]] = set()
        orbit_rows: list[dict[str, object]] = []

        for action, transformed in generate_orbit(point, actions):
            key = canonical_key(transformed)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            row: dict[str, object] = {
                "point_id": point_id,
                "action_id": action.action_id,
                "permutation": json.dumps(list(action.permutation)),
                "signs": json.dumps(list(action.signs)),
                "canonical_key": canonical_label,
                "is_canonical": action.action_id == canonical_action.action_id,
                "family_lambda": lambda_value,
            }
            normalized = phase_normalize(transformed)
            for coord_idx in range(normalized.shape[0]):
                row[f"z{coord_idx}_re"] = float(normalized[coord_idx].real)
                row[f"z{coord_idx}_im"] = float(normalized[coord_idx].imag)
            orbit_rows.append(row)

        orbit_size = len(orbit_rows)
        for row in orbit_rows:
            row["orbit_size"] = orbit_size
        rows.extend(orbit_rows)

    return pd.DataFrame(rows)


def build_canonical_invariants_dataframe(points: np.ndarray, lambda_value: float) -> pd.DataFrame:
    """Build invariant features for canonical orbit representatives."""

    actions = cefalu_symmetry_actions()
    rows: list[dict[str, object]] = []
    for point_id, point in enumerate(points):
        row: dict[str, object] = {"point_id": point_id, "family_lambda": lambda_value}
        row.update(flatten_invariant_matrix(canonical_invariant_matrix(point, actions)))
        rows.append(row)
    return pd.DataFrame(rows)


def build_orbit_metadata_dataframe(points: np.ndarray, lambda_value: float) -> pd.DataFrame:
    """Build lightweight orbit metadata for one batch of Cefalu points."""

    actions = cefalu_symmetry_actions()
    rows: list[dict[str, object]] = []
    for point_id, point in enumerate(points):
        _, canonical_point = choose_canonical_representative(point, actions)
        orbit_size = len({canonical_key(image) for _, image in generate_orbit(point, actions)})
        rows.append(
            {
                "point_id": point_id,
                "canonical_key": canonical_key_string(canonical_point),
                "orbit_size": orbit_size,
                "group_size": len(actions),
                "family_lambda": lambda_value,
            }
        )
    return pd.DataFrame(rows)


def build_canonical_representatives_dataframe(points: np.ndarray, lambda_value: float) -> pd.DataFrame:
    """Build a canonical representative table for one batch of Cefalu points."""

    actions = cefalu_symmetry_actions()
    rows: list[dict[str, object]] = []
    for point_id, point in enumerate(points):
        _, canonical_point = choose_canonical_representative(point, actions)
        canonical = phase_normalize(canonical_point)
        orbit_size = len({canonical_key(image) for _, image in generate_orbit(point, actions)})
        row: dict[str, object] = {
            "point_id": point_id,
            "canonical_key": canonical_key_string(canonical),
            "orbit_size": orbit_size,
            "family_lambda": lambda_value,
        }
        for coord_idx in range(canonical.shape[0]):
            row[f"z{coord_idx}_re"] = float(canonical[coord_idx].real)
            row[f"z{coord_idx}_im"] = float(canonical[coord_idx].imag)
        rows.append(row)
    return pd.DataFrame(rows)


def build_symmetry_report(
    points: np.ndarray,
    *,
    lambda_value: float,
    residual_tol: float = 1e-8,
    canonical_tol: float = 1e-10,
    invariant_tol: float = 1e-10,
) -> dict[str, object]:
    """Validate residual preservation, orbit size, and canonical consistency for Cefalu orbits."""

    actions = cefalu_symmetry_actions()
    residual_deltas: list[float] = []
    canonical_deltas: list[float] = []
    invariant_deltas: list[float] = []
    orbit_sizes: list[int] = []

    for point in points:
        orbit = generate_orbit(point, actions)
        orbit_points = np.array([image for _, image in orbit], dtype=np.complex128)
        base_residual = hypersurface_residuals(
            point[np.newaxis, :],
            geometry_name="cefalu_quartic",
            parameters={"lambda": lambda_value},
        )[0]
        transformed_residuals = hypersurface_residuals(
            orbit_points,
            geometry_name="cefalu_quartic",
            parameters={"lambda": lambda_value},
        )
        residual_deltas.extend(float(abs(base_residual - value)) for value in transformed_residuals)

        keys = [canonical_key(image) for image in orbit_points]
        orbit_sizes.append(len(set(keys)))
        minimum_key = min(keys)
        canonical_images = [phase_normalize(image) for image, key in zip(orbit_points, keys) if key == minimum_key]
        reference_image = canonical_images[0]
        canonical_deltas.extend(
            float(np.max(np.abs(reference_image - candidate)))
            for candidate in canonical_images
        )

        canonical_matrix = canonical_invariant_matrix(point, actions)
        invariant_deltas.extend(
            float(np.max(np.abs(canonical_matrix - canonical_invariant_matrix(image, actions))))
            for image in canonical_images
        )

    report = {
        "geometry": "cefalu_quartic",
        "parameters": {"lambda": lambda_value},
        "group_size": len(actions),
        "n_points": int(points.shape[0]),
        "orbit_size": {
            "min": int(min(orbit_sizes)),
            "max": int(max(orbit_sizes)),
            "mean": float(np.mean(orbit_sizes)),
        },
        "residual_preservation": {
            "max": float(max(residual_deltas, default=0.0)),
            "mean": float(np.mean(residual_deltas)) if residual_deltas else 0.0,
        },
        "canonicalization_drift": {
            "max": float(max(canonical_deltas, default=0.0)),
            "mean": float(np.mean(canonical_deltas)) if canonical_deltas else 0.0,
        },
        "canonical_invariant_drift": {
            "max": float(max(invariant_deltas, default=0.0)),
            "mean": float(np.mean(invariant_deltas)) if invariant_deltas else 0.0,
        },
        "tolerances": {
            "residual_preservation_max": residual_tol,
            "canonicalization_max": canonical_tol,
            "canonical_invariant_max": invariant_tol,
        },
        "warnings": [],
    }
    if report["residual_preservation"]["max"] > residual_tol:
        report["warnings"].append("Symmetry action changed the polynomial residual beyond tolerance.")
    if report["canonicalization_drift"]["max"] > canonical_tol:
        report["warnings"].append("Canonical representative selection is not deterministic enough.")
    if report["canonical_invariant_drift"]["max"] > invariant_tol:
        report["warnings"].append("Canonical invariants drift across symmetry-related representatives.")
    report["passed"] = not report["warnings"]
    return report
