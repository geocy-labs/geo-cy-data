"""Experiment runner orchestration for GeoCYData."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from geocydata.experiments.data import (
    BundleDataset,
    TARGET_LABELS,
    TARGET_METADATA,
    load_bundle_dataset,
    prepare_experiment_matrix,
)
from geocydata.experiments.models import MODEL_LABELS, build_regressor
from geocydata.utils.paths import ensure_directory


def _summary_markdown(config: dict[str, object], metrics: dict[str, object]) -> str:
    return f"""# GeoCYData Experiment Summary

## Run

- model: `{config["model"]}`
- bundle: `{config["bundle"]}`
- benchmark case: `{config["benchmark_case_id"]}`
- target: `{metrics["target_name"]}`
- target description: `{metrics["target_description"]}`
- target status: `{metrics["target_status"]}`
- target kind: `{metrics["target_kind"]}`
- geometry: `{metrics["geometry"]}`
- lambda: `{metrics["lambda"]}`
- split strategy: `{metrics["split_strategy"]}`
- split seed: `{metrics["split_seed"]}`

## Metrics

- train score: `{metrics["train_score"]:.6f}`
- validation score: `{metrics["validation_score"]:.6f}`
- train mse: `{metrics["train_mse"]:.6e}`
- validation mse: `{metrics["validation_mse"]:.6e}`
- train mae: `{metrics["train_mae"]:.6e}`
- validation mae: `{metrics["validation_mae"]:.6e}`
- runtime seconds: `{metrics["runtime_seconds"]:.3f}`
- samples: `{metrics["n_samples"]}`
- feature dimension: `{metrics["feature_dim"]}`
- feature mode: `{metrics["feature_mode"]}`
- train samples: `{metrics["train_samples"]}`
- validation samples: `{metrics["validation_samples"]}`
"""


def _comparison_markdown(comparison: dict[str, object]) -> str:
    local_metrics = comparison["local"]
    global_metrics = comparison["global"]
    return f"""# GeoCYData Experiment Comparison

## Bundle

- bundle: `{comparison["bundle"]}`
- benchmark case: `{comparison["benchmark_case_id"]}`
- geometry: `{comparison["geometry"]}`
- lambda: `{comparison["lambda"]}`
- target: `{comparison["target_name"]}`
- target status: `{comparison["target_status"]}`

## Validation

- local score: `{local_metrics["validation_score"]:.6f}`
- global score: `{global_metrics["validation_score"]:.6f}`
- validation score delta (global - local): `{comparison["metric_deltas"]["validation_score"]:.6f}`
- local mse: `{local_metrics["validation_mse"]:.6e}`
- global mse: `{global_metrics["validation_mse"]:.6e}`
- validation mse delta (global - local): `{comparison["metric_deltas"]["validation_mse"]:.6e}`
- local mae: `{local_metrics["validation_mae"]:.6e}`
- global mae: `{global_metrics["validation_mae"]:.6e}`
- validation mae delta (global - local): `{comparison["metric_deltas"]["validation_mae"]:.6e}`
"""


def _metrics_dict(
    *,
    dataset: BundleDataset,
    target_name: str,
    split_seed: int,
    benchmark_case_id: str | None,
    feature_dim: int,
    n_samples: int,
    runtime_seconds: float,
    train_samples: int,
    validation_samples: int,
    y_train,
    y_val,
    train_pred,
    val_pred,
) -> dict[str, object]:
    parameters = dict(dataset.manifest.get("parameters", {}))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "phase9",
        "bundle_geometry": dataset.manifest.get("geometry"),
        "geometry": dataset.manifest.get("geometry"),
        "bundle_path": str(dataset.bundle_dir),
        "lambda": parameters.get("lambda"),
        "target_name": target_name,
        "target_description": TARGET_LABELS[target_name],
        "target_status": TARGET_METADATA[target_name]["status"],
        "target_kind": TARGET_METADATA[target_name]["kind"],
        "n_samples": n_samples,
        "feature_dim": feature_dim,
        "runtime_seconds": runtime_seconds,
        "seed": split_seed,
        "bundle_seed": parameters.get("seed"),
        "split_strategy": "deterministic_random_train_validation_split",
        "split_seed": split_seed,
        "benchmark_case_id": benchmark_case_id,
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "train_size": train_samples,
        "validation_size": validation_samples,
        "train_score": float(r2_score(y_train, train_pred)),
        "validation_score": float(r2_score(y_val, val_pred)),
        "train_mse": float(mean_squared_error(y_train, train_pred)),
        "validation_mse": float(mean_squared_error(y_val, val_pred)),
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "validation_mae": float(mean_absolute_error(y_val, val_pred)),
    }


def run_experiment(
    *,
    bundle_dir: str | Path,
    model_name: str,
    target_name: str,
    out_dir: str | Path,
    seed: int = 7,
    test_size: float = 0.2,
    benchmark_case_id: str | None = None,
) -> dict[str, object]:
    """Run one experiment and write reproducible artifacts."""

    dataset = load_bundle_dataset(bundle_dir)
    matrix = prepare_experiment_matrix(dataset, model_name=model_name, target_name=target_name)
    output_dir = ensure_directory(out_dir)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "phase9",
        "bundle": str(Path(bundle_dir)),
        "geometry": dataset.manifest.get("geometry"),
        "lambda": dict(dataset.manifest.get("parameters", {})).get("lambda"),
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "feature_mode": model_name,
        "seed": seed,
        "bundle_seed": dict(dataset.manifest.get("parameters", {})).get("seed"),
        "test_size": test_size,
        "split_strategy": "deterministic_random_train_validation_split",
        "split_seed": seed,
        "benchmark_case_id": benchmark_case_id,
        "target_name": matrix.target_name,
        "target_description": TARGET_LABELS[matrix.target_name],
        "target_status": TARGET_METADATA[matrix.target_name]["status"],
        "target_kind": TARGET_METADATA[matrix.target_name]["kind"],
        "run_path": str(output_dir),
    }

    start = time.perf_counter()
    indices = matrix.point_ids
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        matrix.X,
        matrix.y,
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    regressor = build_regressor(model_name)
    regressor.fit(X_train, y_train)
    train_pred = regressor.predict(X_train)
    val_pred = regressor.predict(X_val)
    runtime_seconds = time.perf_counter() - start

    metrics = _metrics_dict(
        dataset=dataset,
        target_name=matrix.target_name,
        split_seed=seed,
        benchmark_case_id=benchmark_case_id,
        feature_dim=matrix.X.shape[1],
        n_samples=matrix.X.shape[0],
        runtime_seconds=runtime_seconds,
        train_samples=len(idx_train),
        validation_samples=len(idx_val),
        y_train=y_train,
        y_val=y_val,
        train_pred=train_pred,
        val_pred=val_pred,
    )
    metrics["model"] = model_name
    metrics["model_label"] = MODEL_LABELS[model_name]
    metrics["feature_mode"] = model_name
    metrics["run_path"] = str(output_dir)

    run_manifest = {
        "created_at": metrics["created_at"],
        "protocol_version": "phase9",
        "bundle_path": str(dataset.bundle_dir),
        "run_path": str(output_dir),
        "benchmark_case_id": benchmark_case_id,
        "geometry": metrics["geometry"],
        "lambda": metrics["lambda"],
        "target_name": metrics["target_name"],
        "target_status": metrics["target_status"],
        "model": model_name,
        "feature_mode": model_name,
        "split_strategy": metrics["split_strategy"],
        "split_seed": metrics["split_seed"],
        "n_samples": metrics["n_samples"],
        "train_size": metrics["train_size"],
        "validation_size": metrics["validation_size"],
        "artifacts": {
            "config": "config.json",
            "metrics": "metrics.json",
            "predictions": "predictions.parquet",
            "summary": "summary.md",
            "run_manifest": "run_manifest.json",
        },
    }

    predictions = pd.DataFrame(
        {
            "point_id": list(idx_train) + list(idx_val),
            "split": ["train"] * len(idx_train) + ["validation"] * len(idx_val),
            "y_true": list(y_train) + list(y_val),
            "y_pred": list(train_pred) + list(val_pred),
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "summary.md").write_text(_summary_markdown(config, metrics), encoding="utf-8")
    return metrics


def compare_experiments(
    *,
    bundle_dir: str | Path,
    out_dir: str | Path,
    target_name: str,
    seed: int = 7,
    test_size: float = 0.2,
) -> dict[str, object]:
    """Run local and global experiment modes and write a comparison summary."""

    output_dir = ensure_directory(out_dir)
    local_metrics = run_experiment(
        bundle_dir=bundle_dir,
        model_name="local",
        target_name=target_name,
        out_dir=output_dir / "local",
        seed=seed,
        test_size=test_size,
        benchmark_case_id=None,
    )
    global_metrics = run_experiment(
        bundle_dir=bundle_dir,
        model_name="global",
        target_name=target_name,
        out_dir=output_dir / "global",
        seed=seed,
        test_size=test_size,
        benchmark_case_id=None,
    )
    dataset = load_bundle_dataset(bundle_dir)
    comparison = {
        "bundle": str(Path(bundle_dir)),
        "benchmark_case_id": None,
        "geometry": dataset.manifest.get("geometry"),
        "lambda": dict(dataset.manifest.get("parameters", {})).get("lambda"),
        "target_name": local_metrics["target_name"],
        "target_description": local_metrics["target_description"],
        "target_status": local_metrics["target_status"],
        "local": local_metrics,
        "global": global_metrics,
        "metric_deltas": {
            "validation_score": global_metrics["validation_score"] - local_metrics["validation_score"],
            "validation_mse": global_metrics["validation_mse"] - local_metrics["validation_mse"],
            "validation_mae": global_metrics["validation_mae"] - local_metrics["validation_mae"],
        },
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (output_dir / "comparison.md").write_text(_comparison_markdown(comparison), encoding="utf-8")
    return comparison
