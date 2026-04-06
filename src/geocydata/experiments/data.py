"""Bundle loading and feature preparation for experiment runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from geocydata.baselines.fubini_study import fubini_study_scalar, hypersurface_fubini_study_scalar


TARGET_METADATA = {
    "hypersurface_fs_scalar": {
        "description": (
            "Tangent-restricted Fubini-Study scalar proxy using the local hypersurface gradient. "
            "Preferred current benchmark target."
        ),
        "status": "preferred",
        "kind": "hypersurface-aware",
    },
    "fs_scalar": {
        "description": "Ambient Fubini-Study determinant proxy from affine chart coordinates. Legacy proxy target.",
        "status": "legacy_proxy",
        "kind": "ambient-proxy",
    },
    "invariant_weighted_sum": {
        "description": "Convenience/debug target from a weighted sum of invariant features.",
        "status": "debug",
        "kind": "invariant-debug",
    },
}
TARGET_LABELS = {name: payload["description"] for name, payload in TARGET_METADATA.items()}


@dataclass(frozen=True)
class BundleDataset:
    """Loaded bundle tables and metadata."""

    bundle_dir: Path
    manifest: dict[str, object]
    points_df: pd.DataFrame
    invariants_df: pd.DataFrame


@dataclass(frozen=True)
class ExperimentMatrix:
    """Prepared supervised learning arrays."""

    X: np.ndarray
    y: np.ndarray
    point_ids: np.ndarray
    feature_names: list[str]
    target_name: str


def load_bundle_dataset(bundle_dir: str | Path) -> BundleDataset:
    """Load a GeoCYData bundle for experiments."""

    bundle_path = Path(bundle_dir)
    manifest_path = bundle_path / "manifest.json"
    points_path = bundle_path / "points.parquet"
    invariants_path = bundle_path / "invariants.parquet"
    for required in (manifest_path, points_path, invariants_path):
        if not required.exists():
            raise FileNotFoundError(f"Bundle is missing required file: {required.name}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    points_df = pd.read_parquet(points_path)
    invariants_df = pd.read_parquet(invariants_path)
    return BundleDataset(
        bundle_dir=bundle_path,
        manifest=manifest,
        points_df=points_df,
        invariants_df=invariants_df,
    )


def target_from_invariants(invariants_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Build a deterministic scalar regression target from invariant features."""

    feature_columns = [
        column
        for column in invariants_df.columns
        if column.startswith("m_") and (column.endswith("_re") or column.endswith("_im"))
    ]
    if not feature_columns:
        raise ValueError("Invariant feature table does not contain expected m_* columns.")
    weights = np.linspace(0.5, 1.5, num=len(feature_columns), dtype=np.float64)
    matrix = invariants_df[feature_columns].to_numpy(dtype=np.float64)
    target = (matrix @ weights) / float(len(feature_columns))
    return target.astype(np.float64), "invariant_weighted_sum"


def geometry_target_from_points(points_df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Build the older ambient-geometry scalar target from affine chart coordinates."""

    return fubini_study_scalar(points_df).astype(np.float64), "fs_scalar"


def hypersurface_geometry_target(dataset: BundleDataset) -> tuple[np.ndarray, str]:
    """Build the preferred hypersurface-aware scalar target from local chart data."""

    parameters = dict(dataset.manifest.get("parameters", {}))
    geometry_name = str(dataset.manifest.get("geometry", ""))
    target = hypersurface_fubini_study_scalar(
        dataset.points_df,
        geometry_name=geometry_name,
        parameters=parameters,
    )
    return target.astype(np.float64), "hypersurface_fs_scalar"


def local_feature_matrix(points_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build local affine-coordinate features with one-hot chart indicators."""

    affine_columns = [
        column
        for column in points_df.columns
        if column.startswith("affine_") and (column.endswith("_re") or column.endswith("_im"))
    ]
    if not affine_columns:
        raise ValueError("Points table does not contain expected affine_* columns.")
    features = points_df[affine_columns].copy()
    chart_dummies = pd.get_dummies(points_df["chart_id"], prefix="chart", dtype=float)
    features = pd.concat([features, chart_dummies], axis=1)
    return features.to_numpy(dtype=np.float64), list(features.columns)


def global_feature_matrix(invariants_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build global invariant-input features."""

    feature_columns = [
        column
        for column in invariants_df.columns
        if column.startswith("m_") and (column.endswith("_re") or column.endswith("_im"))
    ]
    if not feature_columns:
        raise ValueError("Invariant feature table does not contain expected m_* columns.")
    return invariants_df[feature_columns].to_numpy(dtype=np.float64), feature_columns


def build_target(dataset: BundleDataset, target_name: str) -> tuple[np.ndarray, str]:
    """Build a supervised target array for the selected experiment target."""

    if target_name == "hypersurface_fs_scalar":
        return hypersurface_geometry_target(dataset)
    if target_name == "fs_scalar":
        return geometry_target_from_points(dataset.points_df)
    if target_name == "invariant_weighted_sum":
        return target_from_invariants(dataset.invariants_df)
    raise ValueError(f"Unsupported experiment target '{target_name}'.")


def prepare_experiment_matrix(
    dataset: BundleDataset,
    model_name: str,
    *,
    target_name: str,
) -> ExperimentMatrix:
    """Prepare features and target arrays for the chosen model family and target."""

    y, resolved_target_name = build_target(dataset, target_name)
    if model_name == "local":
        X, feature_names = local_feature_matrix(dataset.points_df)
    elif model_name == "global":
        X, feature_names = global_feature_matrix(dataset.invariants_df)
    else:
        raise ValueError(f"Unsupported experiment model '{model_name}'.")

    point_ids = dataset.points_df["point_id"].to_numpy(dtype=np.int64)
    return ExperimentMatrix(
        X=X,
        y=y,
        point_ids=point_ids,
        feature_names=feature_names,
        target_name=resolved_target_name,
    )
