"""Point sampling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd

from geocydata.features.invariants import build_invariants_dataframe
from geocydata.geometry.charts import select_affine_charts
from geocydata.registry.cases import derive_case_id
from geocydata.registry.geometries import get_geometry
from geocydata.utils.seeds import make_rng
from geocydata.validation.symmetry_checks import (
    build_canonical_invariants_dataframe,
    build_canonical_representatives_dataframe,
    build_orbit_metadata_dataframe,
)


@dataclass
class SampleBatch:
    """Container for generated point arrays and exported tables."""

    points: np.ndarray
    points_df: pd.DataFrame
    invariants_df: pd.DataFrame
    sample_weights_df: pd.DataFrame
    canonical_representatives_df: pd.DataFrame | None
    orbits_df: pd.DataFrame | None
    canonical_invariants_df: pd.DataFrame | None


def _normalize_weights(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    if np.isclose(mean, 0.0):
        return np.ones_like(values, dtype=np.float64)
    return values / mean


def build_sample_weights_dataframe(
    *,
    n: int,
    chart_ids: np.ndarray,
    orbit_sizes: np.ndarray | None,
    geometry_name: str,
    case_id: str,
    seed: int | None,
    parameters: dict[str, object],
) -> pd.DataFrame:
    """Build model-facing sample weights for one generated bundle."""

    counts = Counter(int(value) for value in chart_ids.tolist())
    chart_balance = np.array([1.0 / counts[int(value)] for value in chart_ids], dtype=np.float64)
    chart_balance = _normalize_weights(chart_balance)
    uniform = np.ones(n, dtype=np.float64)
    symmetry = np.ones(n, dtype=np.float64)
    if orbit_sizes is not None:
        symmetry = _normalize_weights(1.0 / orbit_sizes.astype(np.float64))
    combined = _normalize_weights(chart_balance * symmetry)
    rows: dict[str, object] = {
        "point_id": list(range(n)),
        "geometry": [geometry_name] * n,
        "case_id": [case_id] * n,
        "seed": [seed] * n,
        "uniform_weight": uniform.tolist(),
        "chart_balance_weight": chart_balance.tolist(),
        "symmetry_orbit_weight": symmetry.tolist(),
        "combined_weight": combined.tolist(),
    }
    if "lambda" in parameters:
        rows["family_lambda"] = [float(parameters["lambda"])] * n
    return pd.DataFrame(rows)


def generate_sample_batch(
    geometry_name: str,
    n: int,
    seed: int | None,
    parameters: dict[str, object] | None = None,
    include_symmetry_exports: bool = False,
) -> SampleBatch:
    """Generate points and derived tables for a registered geometry."""

    geometry = get_geometry(geometry_name)
    rng = make_rng(seed)
    resolved_parameters = geometry.validate_parameters(parameters)
    case_id = derive_case_id(geometry_name, resolved_parameters)
    points = geometry.sample_points(n=n, rng=rng, parameters=resolved_parameters)
    chart_ids, affine = select_affine_charts(points)

    point_rows: dict[str, list[float] | list[int]] = {
        "point_id": list(range(n)),
        "chart_id": chart_ids.tolist(),
        "geometry": [geometry_name] * n,
        "case_id": [case_id] * n,
        "seed": [seed] * n,
    }
    for coord_idx in range(points.shape[1]):
        point_rows[f"z{coord_idx}_re"] = points[:, coord_idx].real.tolist()
        point_rows[f"z{coord_idx}_im"] = points[:, coord_idx].imag.tolist()
    for affine_idx in range(affine.shape[1]):
        point_rows[f"affine_{affine_idx}_re"] = affine[:, affine_idx].real.tolist()
        point_rows[f"affine_{affine_idx}_im"] = affine[:, affine_idx].imag.tolist()
    if "lambda" in resolved_parameters:
        point_rows["family_lambda"] = [float(resolved_parameters["lambda"])] * n

    points_df = pd.DataFrame(point_rows)
    invariants_df = build_invariants_dataframe(points)
    invariants_df["geometry"] = geometry_name
    invariants_df["case_id"] = case_id
    invariants_df["seed"] = seed
    orbits_df = None
    canonical_representatives_df = None
    canonical_invariants_df = None
    orbit_sizes = None
    if geometry_name == "cefalu_quartic" and include_symmetry_exports:
        lambda_value = float(resolved_parameters["lambda"])
        orbits_df = build_orbit_metadata_dataframe(points, lambda_value=lambda_value)
        canonical_representatives_df = build_canonical_representatives_dataframe(points, lambda_value=lambda_value)
        canonical_representatives_df["geometry"] = geometry_name
        canonical_representatives_df["case_id"] = case_id
        canonical_representatives_df["seed"] = seed
        canonical_invariants_df = build_canonical_invariants_dataframe(points, lambda_value=lambda_value)
        canonical_invariants_df["geometry"] = geometry_name
        canonical_invariants_df["case_id"] = case_id
        canonical_invariants_df["seed"] = seed
        orbit_sizes = canonical_representatives_df["orbit_size"].to_numpy(dtype=np.int64)
        orbits_df["geometry"] = geometry_name
        orbits_df["case_id"] = case_id
        orbits_df["seed"] = seed
    if "lambda" in resolved_parameters:
        invariants_df["family_lambda"] = float(resolved_parameters["lambda"])
    sample_weights_df = build_sample_weights_dataframe(
        n=n,
        chart_ids=chart_ids,
        orbit_sizes=orbit_sizes,
        geometry_name=geometry_name,
        case_id=case_id,
        seed=seed,
        parameters=resolved_parameters,
    )
    return SampleBatch(
        points=points,
        points_df=points_df,
        invariants_df=invariants_df,
        sample_weights_df=sample_weights_df,
        canonical_representatives_df=canonical_representatives_df,
        orbits_df=orbits_df,
        canonical_invariants_df=canonical_invariants_df,
    )
