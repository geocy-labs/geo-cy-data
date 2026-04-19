"""Benchmark sweep orchestration for GeoCYData experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from geocydata.experiments.data import TARGET_METADATA
from geocydata.experiments.protocols import resolve_protocol_preset
from geocydata.experiments.reporting import (
    aggregate_benchmark_results,
    build_benchmark_aggregated_summary_markdown,
    build_benchmark_manifest,
    build_benchmark_robustness_summary_markdown,
    build_benchmark_summary_markdown,
    build_robustness_table,
    write_benchmark_aggregated_results,
    write_benchmark_robustness_outputs,
    write_benchmark_results,
)
from geocydata.experiments.runner import run_experiment
from geocydata.export.manifest import build_manifest, write_manifest
from geocydata.export.parquet_io import write_parquet
from geocydata.registry.cases import GEOMETRY_CASES, GeometryCase, build_benchmark_case_entry, model_facing_views_for_case
from geocydata.registry.geometries import get_geometry
from geocydata.sampling.point_sampler import generate_sample_batch
from geocydata.utils.paths import ensure_directory
from geocydata.validation.reports import build_validation_report


BENCHMARK_CASES: tuple[GeometryCase, ...] = GEOMETRY_CASES


def _bundle_summary_markdown(
    *,
    geometry_name: str,
    parameters: dict[str, object],
    n_points: int,
    artifact_paths: dict[str, str],
    report: dict[str, object],
) -> str:
    chart_distribution = ", ".join(
        f"chart {chart}: {count}" for chart, count in report["chart_distribution"].items()
    )
    artifacts = "\n".join(f"- `{name}`: `{path}`" for name, path in artifact_paths.items())
    warnings = report["warnings"] or ["none"]
    warning_text = "\n".join(f"- {warning}" for warning in warnings)
    parameter_lines = "\n".join(f"- {name}: `{value}`" for name, value in parameters.items()) or "- none"
    return f"""# GeoCYData Bundle Summary

## Geometry

- name: `{geometry_name}`
- points: `{n_points}`

## Parameters

{parameter_lines}

## Validation

- max residual: `{report["residual"]["max"]:.3e}`
- mean residual: `{report["residual"]["mean"]:.3e}`
- max invariant drift: `{report["invariant_drift"]["max"]:.3e}`
- mean invariant drift: `{report["invariant_drift"]["mean"]:.3e}`
- passed: `{report["passed"]}`

## Chart distribution

- {chart_distribution}

## Warnings

{warning_text}

## Artifacts

{artifacts}
"""


def _resolve_cases(include: list[str] | None) -> list[GeometryCase]:
    if not include:
        return list(BENCHMARK_CASES)

    available = {case.case_id: case for case in BENCHMARK_CASES}
    missing = [case_id for case_id in include if case_id not in available]
    if missing:
        choices = ", ".join(sorted(available))
        raise ValueError(f"Unknown benchmark case(s): {', '.join(missing)}. Available cases: {choices}.")
    return [available[case_id] for case_id in include]


def _bundle_matches_request(
    manifest: dict[str, object],
    *,
    case: GeometryCase,
    n: int,
    seed: int,
) -> bool:
    parameters = dict(manifest.get("parameters", {}))
    if manifest.get("geometry") != case.geometry:
        return False
    if manifest.get("n_points") != n or manifest.get("seed") != seed:
        return False
    for key, value in case.parameters.items():
        if parameters.get(key) != value:
            return False
    return True


def ensure_benchmark_bundle(
    *,
    case: GeometryCase,
    bundle_dir: Path,
    n: int,
    seed: int,
) -> Path:
    """Reuse or generate one bundle needed by the benchmark sweep."""

    output_dir = ensure_directory(bundle_dir)
    manifest_path = output_dir / "manifest.json"
    points_path = output_dir / "points.parquet"
    invariants_path = output_dir / "invariants.parquet"
    sample_weights_path = output_dir / "sample_weights.parquet"
    case_metadata_path = output_dir / "case_metadata.json"
    report_path = output_dir / "validation_report.json"
    evaluation_summary_path = output_dir / "evaluation_summary.json"
    summary_path = output_dir / "summary.md"
    required = (
        manifest_path,
        points_path,
        invariants_path,
        sample_weights_path,
        case_metadata_path,
        report_path,
        evaluation_summary_path,
        summary_path,
    )
    if all(path.exists() for path in required):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _bundle_matches_request(manifest, case=case, n=n, seed=seed):
            return output_dir

    geometry = get_geometry(case.geometry)
    batch = generate_sample_batch(
        geometry_name=case.geometry,
        n=n,
        seed=seed,
        parameters=case.parameters,
        include_symmetry_exports=case.geometry == "cefalu_quartic",
    )
    report = build_validation_report(
        batch.points,
        geometry_name=case.geometry,
        parameters=case.parameters,
        n_points=n,
        seed=seed,
        points_df=batch.points_df,
    )
    artifact_paths = {
        "manifest": "manifest.json",
        "case_metadata": "case_metadata.json",
        "points": "points.parquet",
        "invariants": "invariants.parquet",
        "sample_weights": "sample_weights.parquet",
        "validation_report": "validation_report.json",
        "evaluation_summary": "evaluation_summary.json",
        "summary": "summary.md",
    }
    if batch.canonical_representatives_df is not None:
        artifact_paths["canonical_representatives"] = "canonical_representatives.parquet"
    if batch.canonical_invariants_df is not None:
        artifact_paths["canonical_invariants"] = "canonical_invariants.parquet"
    if batch.orbits_df is not None:
        artifact_paths["orbits"] = "orbits.parquet"
        artifact_paths["symmetry_report"] = "symmetry_report.json"
    write_parquet(batch.points_df, points_path)
    write_parquet(batch.invariants_df, invariants_path)
    write_parquet(batch.sample_weights_df, sample_weights_path)
    if batch.canonical_representatives_df is not None:
        write_parquet(batch.canonical_representatives_df, output_dir / artifact_paths["canonical_representatives"])
    if batch.canonical_invariants_df is not None:
        write_parquet(batch.canonical_invariants_df, output_dir / artifact_paths["canonical_invariants"])
    if batch.orbits_df is not None:
        write_parquet(batch.orbits_df, output_dir / artifact_paths["orbits"])
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    evaluation_summary_path.write_text(
        json.dumps(report.get("geometry_evaluation_hooks", {}), indent=2),
        encoding="utf-8",
    )
    case_metadata_path.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "geometry": case.geometry,
                "lambda": case.parameters.get("lambda"),
                "seed": seed,
                "n": n,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if batch.orbits_df is not None:
        symmetry_report = report.get("geometry_evaluation_hooks", {}).get("symmetry_consistency", {})
        (output_dir / artifact_paths["symmetry_report"]).write_text(
            json.dumps(symmetry_report, indent=2),
            encoding="utf-8",
        )
    summary_path.write_text(
        _bundle_summary_markdown(
            geometry_name=case.geometry,
            parameters={**case.parameters, "case_id": case.case_id},
            n_points=n,
            artifact_paths=artifact_paths,
            report=report,
        ),
        encoding="utf-8",
    )
    manifest = build_manifest(
        geometry=case.geometry,
        n_points=n,
        seed=seed,
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        parameters={
            "geometry": case.geometry,
            "geometry_description": geometry.description,
            **case.parameters,
            "case_id": case.case_id,
            "n": n,
            "seed": seed,
        },
        case_id=case.case_id,
        protocol_metadata={
            "export_profile": "benchmark_sweep_model_ready",
            "case_label": case.label,
            "paper1_core_case": case.case_id in {"fermat_quartic", "cefalu_lambda_0_0", "cefalu_lambda_0_75", "cefalu_lambda_1_0", "cefalu_lambda_1_5", "cefalu_lambda_3_0"},
            "available_model_facing_views": model_facing_views_for_case(case),
        },
    )
    write_manifest(manifest, manifest_path)
    return output_dir


def _write_case_manifest(
    *,
    case: GeometryCase,
    target_name: str,
    seed: int,
    n: int,
    test_size: float,
    bundle_dir: Path,
    case_dir: Path,
    model_run_dirs: dict[str, str],
    preset_name: str | None,
) -> None:
    case_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "phase9",
        "benchmark_case_id": case.case_id,
        "case_label": case.label,
        "geometry": case.geometry,
        "parameters": case.parameters,
        "preset_name": preset_name,
        "target_name": target_name,
        "target_description": TARGET_METADATA[target_name]["description"],
        "target_status": TARGET_METADATA[target_name]["status"],
        "seed": seed,
        "n_samples": n,
        "test_size": test_size,
        "split_strategy": "deterministic_random_train_validation_split",
        "bundle_path": str(bundle_dir),
        "run_paths": model_run_dirs,
        "available_model_facing_views": model_facing_views_for_case(case),
        "artifacts": {
            "case_manifest": "case_manifest.json",
        },
    }
    (case_dir / "case_manifest.json").write_text(json.dumps(case_manifest, indent=2), encoding="utf-8")


def sweep_experiments(
    *,
    out_dir: str | Path,
    target_name: str | None = None,
    seeds: list[int] | None = None,
    n: int = 200,
    include: list[str] | None = None,
    test_size: float = 0.2,
    preset_name: str | None = None,
) -> dict[str, object]:
    """Run the standardized benchmark sweep and write per-seed plus aggregated results."""

    resolved_preset = resolve_protocol_preset(preset_name) if preset_name else None
    benchmark_version = resolved_preset.benchmark_version if resolved_preset else "ad_hoc_benchmark_v1"
    resolved_target_name = target_name or (resolved_preset.target_name if resolved_preset else "hypersurface_fs_scalar")
    resolved_seeds = sorted(set(seeds or (list(resolved_preset.seeds) if resolved_preset else [7])))
    resolved_n = n if n != 200 or not resolved_preset else resolved_preset.n_samples
    resolved_test_size = test_size if test_size != 0.2 or not resolved_preset else resolved_preset.test_size
    resolved_include = include or (list(resolved_preset.include) if resolved_preset else None)
    cases = _resolve_cases(resolved_include)
    output_dir = ensure_directory(out_dir)
    bundles_root = ensure_directory(output_dir / "bundles")
    runs_root = ensure_directory(output_dir / "runs")
    records: list[dict[str, object]] = []

    benchmark_protocol = {
        "protocol_version": "phase9",
        "benchmark_version": benchmark_version,
        "preset_name": preset_name,
        "preset": resolved_preset.metadata() if resolved_preset else None,
        "resolved": {
            "target_name": resolved_target_name,
            "target_description": TARGET_METADATA[resolved_target_name]["description"],
            "target_status": TARGET_METADATA[resolved_target_name]["status"],
            "seeds": resolved_seeds,
            "n_samples": resolved_n,
            "include": [case.case_id for case in cases],
            "test_size": resolved_test_size,
            "split_strategy": "deterministic_random_train_validation_split",
        },
    }
    benchmark_preset_manifest = {
        "preset_name": preset_name,
        "benchmark_version": benchmark_version,
        "geometry_family": "cefalu_quartic" if all(case.geometry == "cefalu_quartic" for case in cases) else "mixed",
        "target_name": resolved_target_name,
        "seeds": resolved_seeds,
        "n_samples": resolved_n,
        "cases": [build_benchmark_case_entry(case, benchmark_version=benchmark_version) for case in cases],
    }

    for case in cases:
        for seed in resolved_seeds:
            case_dir = ensure_directory(output_dir / "cases" / case.case_id / f"seed_{seed}")
            bundle_dir = ensure_benchmark_bundle(
                case=case,
                bundle_dir=bundles_root / case.case_id / f"seed_{seed}",
                n=resolved_n,
                seed=seed,
            )
            model_run_dirs: dict[str, str] = {}
            for model_name in ("local", "global"):
                run_dir = runs_root / case.case_id / f"seed_{seed}" / model_name
                metrics = run_experiment(
                    bundle_dir=bundle_dir,
                    model_name=model_name,
                    target_name=resolved_target_name,
                    out_dir=run_dir,
                    seed=seed,
                    test_size=resolved_test_size,
                    benchmark_case_id=case.case_id,
                )
                model_run_dirs[model_name] = str(run_dir)
                record = {
                    "benchmark_case": case.case_id,
                    "case_label": case.label,
                    "geometry": metrics["geometry"],
                    "lambda": metrics["lambda"],
                    "target": metrics["target_name"],
                    "target_status": metrics["target_status"],
                    "model": model_name,
                    "feature_mode": model_name,
                    "seed": seed,
                    "n_samples": metrics["n_samples"],
                    "train_size": metrics["train_size"],
                    "validation_size": metrics["validation_size"],
                    "split_strategy": metrics["split_strategy"],
                    "split_seed": metrics["split_seed"],
                    "validation_score": metrics["validation_score"],
                    "validation_mse": metrics["validation_mse"],
                    "validation_mae": metrics["validation_mae"],
                    "runtime_seconds": metrics["runtime_seconds"],
                    "bundle_path": str(bundle_dir),
                    "run_path": str(run_dir),
                    "timestamp": metrics["created_at"],
                }
                records.append(record)

            _write_case_manifest(
                case=case,
                target_name=resolved_target_name,
                seed=seed,
                n=resolved_n,
                test_size=resolved_test_size,
                bundle_dir=bundle_dir,
                case_dir=case_dir,
                model_run_dirs=model_run_dirs,
                preset_name=preset_name,
            )

    results_df = pd.DataFrame(records).sort_values(["benchmark_case", "seed", "model"]).reset_index(drop=True)
    aggregated_df = aggregate_benchmark_results(results_df)
    robustness_df = build_robustness_table(results_df, aggregated_df)
    write_benchmark_results(results_df, output_dir=output_dir)
    write_benchmark_aggregated_results(aggregated_df, output_dir=output_dir)
    write_benchmark_robustness_outputs(robustness_df, output_dir=output_dir)
    (output_dir / "benchmark_protocol.json").write_text(json.dumps(benchmark_protocol, indent=2), encoding="utf-8")
    (output_dir / "benchmark_preset_manifest.json").write_text(json.dumps(benchmark_preset_manifest, indent=2), encoding="utf-8")
    manifest = build_benchmark_manifest(
        output_dir=output_dir,
        target_name=resolved_target_name,
        seeds=resolved_seeds,
        n_samples=resolved_n,
        test_size=resolved_test_size,
        cases=[build_benchmark_case_entry(case, benchmark_version=benchmark_version) for case in cases],
        result_count=len(results_df),
        benchmark_version=benchmark_version,
        protocol=benchmark_protocol,
    )
    (output_dir / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "benchmark_summary.md").write_text(
        build_benchmark_summary_markdown(results_df, target_name=resolved_target_name, seeds=resolved_seeds),
        encoding="utf-8",
    )
    (output_dir / "benchmark_aggregated_summary.md").write_text(
        build_benchmark_aggregated_summary_markdown(
            aggregated_df,
            results_df,
            target_name=resolved_target_name,
            seeds=resolved_seeds,
        ),
        encoding="utf-8",
    )
    (output_dir / "benchmark_robustness_summary.md").write_text(
        build_benchmark_robustness_summary_markdown(
            robustness_df,
            target_name=resolved_target_name,
            preset_name=preset_name,
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "results": results_df,
        "aggregated_results": aggregated_df,
        "robustness": robustness_df,
        "protocol": benchmark_protocol,
        "preset_manifest": benchmark_preset_manifest,
        "out_dir": str(output_dir),
    }
