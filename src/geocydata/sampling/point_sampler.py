"""Point sampling pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from geocydata.features.invariants import build_invariants_dataframe
from geocydata.geometry.charts import select_affine_charts
from geocydata.registry.geometries import get_geometry
from geocydata.utils.seeds import make_rng


@dataclass
class SampleBatch:
    """Container for generated point arrays and exported tables."""

    points: np.ndarray
    points_df: pd.DataFrame
    invariants_df: pd.DataFrame


def generate_sample_batch(
    geometry_name: str,
    n: int,
    seed: int | None,
    parameters: dict[str, object] | None = None,
) -> SampleBatch:
    """Generate points and derived tables for a registered geometry."""

    geometry = get_geometry(geometry_name)
    rng = make_rng(seed)
    resolved_parameters = geometry.validate_parameters(parameters)
    points = geometry.sample_points(n=n, rng=rng, parameters=resolved_parameters)
    chart_ids, affine = select_affine_charts(points)

    point_rows: dict[str, list[float] | list[int]] = {
        "point_id": list(range(n)),
        "chart_id": chart_ids.tolist(),
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
    if "lambda" in resolved_parameters:
        invariants_df["family_lambda"] = float(resolved_parameters["lambda"])
    return SampleBatch(points=points, points_df=points_df, invariants_df=invariants_df)
