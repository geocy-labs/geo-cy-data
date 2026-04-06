import json
from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from geocydata.cli.main import app
from geocydata.experiments.data import build_target, load_bundle_dataset, prepare_experiment_matrix
from geocydata.experiments.paper_assets import build_paper_assets
from geocydata.experiments.protocols import PROTOCOL_PRESETS, resolve_protocol_preset
from geocydata.experiments.release import create_benchmark_release
from geocydata.experiments.runner import compare_experiments, run_experiment
from geocydata.experiments.sweep import BENCHMARK_CASES, sweep_experiments
from geocydata.experiments.validate_paper_assets import validate_paper_assets
from geocydata.experiments.validate_release import validate_benchmark_release
from geocydata.export.manifest import build_manifest, write_manifest
from geocydata.export.parquet_io import write_parquet
from geocydata.sampling.point_sampler import generate_sample_batch


def create_bundle(tmp_path: Path, *, geometry: str = "cefalu_quartic", lambda_value: float = 1.0) -> Path:
    parameters = {"lambda": lambda_value} if geometry == "cefalu_quartic" else {}
    batch = generate_sample_batch(geometry_name=geometry, n=48, seed=7, parameters=parameters)
    bundle_dir = tmp_path / f"{geometry}-bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(batch.points_df, bundle_dir / "points.parquet")
    write_parquet(batch.invariants_df, bundle_dir / "invariants.parquet")
    manifest = build_manifest(
        geometry=geometry,
        n_points=len(batch.points_df),
        seed=7,
        output_dir=bundle_dir,
        artifact_paths={"points": "points.parquet", "invariants": "invariants.parquet"},
        parameters={
            "geometry": geometry,
            **parameters,
            "n": len(batch.points_df),
            "seed": 7,
        },
    )
    write_manifest(manifest, bundle_dir / "manifest.json")
    return bundle_dir


def test_experiment_bundle_loading(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path)
    dataset = load_bundle_dataset(bundle_dir)
    matrix = prepare_experiment_matrix(dataset, model_name="local", target_name="fs_scalar")
    assert dataset.manifest["geometry"] == "cefalu_quartic"
    assert matrix.X.shape[0] == dataset.points_df.shape[0]


def test_protocol_preset_resolution() -> None:
    preset = resolve_protocol_preset("paper_v1_default")
    assert preset.name == "paper_v1_default"
    assert preset.target_name == "hypersurface_fs_scalar"
    assert list(preset.seeds) == [7, 11, 19]
    assert "paper_v1_multiseed" in PROTOCOL_PRESETS


def test_geometry_target_computation_smoke(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path)
    dataset = load_bundle_dataset(bundle_dir)
    target, target_name = build_target(dataset, "fs_scalar")
    assert target_name == "fs_scalar"
    assert target.shape[0] == dataset.points_df.shape[0]
    assert float(target.min()) > 0.0


def test_hypersurface_target_computation_smoke(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path, geometry="fermat_quartic")
    dataset = load_bundle_dataset(bundle_dir)
    target, target_name = build_target(dataset, "hypersurface_fs_scalar")
    ambient_target, _ = build_target(dataset, "fs_scalar")
    assert target_name == "hypersurface_fs_scalar"
    assert target.shape[0] == dataset.points_df.shape[0]
    assert float(target.min()) > 0.0
    assert not np.allclose(target, ambient_target)


def test_local_model_run_smoke(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path)
    run_dir = tmp_path / "local-run"
    metrics = run_experiment(
        bundle_dir=bundle_dir,
        model_name="local",
        target_name="fs_scalar",
        out_dir=run_dir,
        seed=7,
    )
    assert (run_dir / "metrics.json").exists()
    assert metrics["feature_dim"] > 0
    assert metrics["target_name"] == "fs_scalar"


def test_global_model_run_smoke(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path)
    run_dir = tmp_path / "global-run"
    metrics = run_experiment(
        bundle_dir=bundle_dir,
        model_name="global",
        target_name="fs_scalar",
        out_dir=run_dir,
        seed=7,
    )
    assert (run_dir / "predictions.parquet").exists()
    assert metrics["feature_dim"] > 0
    assert metrics["target_name"] == "fs_scalar"


def test_local_model_run_hypersurface_target(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path, geometry="fermat_quartic")
    run_dir = tmp_path / "local-hfs-run"
    metrics = run_experiment(
        bundle_dir=bundle_dir,
        model_name="local",
        target_name="hypersurface_fs_scalar",
        out_dir=run_dir,
        seed=7,
    )
    assert metrics["target_name"] == "hypersurface_fs_scalar"
    assert metrics["target_status"] == "preferred"
    assert (run_dir / "run_manifest.json").exists()


def test_global_model_run_hypersurface_target(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path, geometry="fermat_quartic")
    run_dir = tmp_path / "global-hfs-run"
    metrics = run_experiment(
        bundle_dir=bundle_dir,
        model_name="global",
        target_name="hypersurface_fs_scalar",
        out_dir=run_dir,
        seed=7,
    )
    assert metrics["target_name"] == "hypersurface_fs_scalar"
    assert metrics["target_status"] == "preferred"
    assert (run_dir / "predictions.parquet").exists()


def test_compare_writes_comparison_file(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path)
    comparison_dir = tmp_path / "compare-run"
    comparison = compare_experiments(
        bundle_dir=bundle_dir,
        out_dir=comparison_dir,
        target_name="fs_scalar",
        seed=7,
    )
    assert (comparison_dir / "comparison.json").exists()
    assert comparison["local"]["target_name"] == comparison["global"]["target_name"]
    assert comparison["target_name"] == "fs_scalar"
    json.loads((comparison_dir / "comparison.json").read_text(encoding="utf-8"))


def test_cli_experiment_run_smoke(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path)
    run_dir = tmp_path / "cli-run"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "run",
            "--bundle",
            str(bundle_dir),
            "--model",
            "global",
            "--target",
            "fs_scalar",
            "--out",
            str(run_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "metrics.json").exists()


def test_sweep_smoke_subset(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "phase6-sweep"
    result = sweep_experiments(
        out_dir=sweep_dir,
        target_name="fs_scalar",
        seeds=[7],
        n=24,
        include=["fermat_quartic", "cefalu_lambda_1_0"],
        test_size=0.25,
    )
    results_df = result["results"]
    assert len(results_df) == 4
    assert set(results_df["model"]) == {"local", "global"}
    assert (sweep_dir / "benchmark_manifest.json").exists()
    assert (sweep_dir / "benchmark_results.csv").exists()
    assert (sweep_dir / "benchmark_results.json").exists()
    assert (sweep_dir / "benchmark_summary.md").exists()
    assert (sweep_dir / "benchmark_aggregated.csv").exists()
    assert (sweep_dir / "benchmark_aggregated.json").exists()
    assert (sweep_dir / "benchmark_aggregated_summary.md").exists()


def test_sweep_results_report_generation(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "phase6-report"
    sweep_experiments(
        out_dir=sweep_dir,
        target_name="fs_scalar",
        seeds=[7],
        n=24,
        include=[BENCHMARK_CASES[0].case_id],
        test_size=0.25,
    )
    results_df = pd.read_csv(sweep_dir / "benchmark_results.csv")
    summary_text = (sweep_dir / "benchmark_summary.md").read_text(encoding="utf-8")
    manifest = json.loads((sweep_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert not results_df.empty
    assert "validation score delta (global - local)" in summary_text
    assert manifest["target"] == "fs_scalar"


def test_hypersurface_target_sweep_subset(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "phase8-sweep"
    result = sweep_experiments(
        out_dir=sweep_dir,
        target_name="hypersurface_fs_scalar",
        seeds=[7, 11],
        n=24,
        include=["fermat_quartic"],
        test_size=0.25,
    )
    assert len(result["results"]) == 4
    assert len(result["aggregated_results"]) == 2
    assert set(result["results"]["target"]) == {"hypersurface_fs_scalar"}


def test_preset_sweep_smoke_with_overrides(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "preset-sweep"
    result = sweep_experiments(
        out_dir=sweep_dir,
        preset_name="paper_v1_default",
        seeds=[7, 11],
        n=24,
        include=["fermat_quartic"],
    )
    assert len(result["results"]) == 4
    protocol = json.loads((sweep_dir / "benchmark_protocol.json").read_text(encoding="utf-8"))
    assert protocol["preset_name"] == "paper_v1_default"
    assert protocol["resolved"]["include"] == ["fermat_quartic"]
    assert protocol["resolved"]["n_samples"] == 24


def test_run_bookkeeping_records_split_and_case_metadata(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path, geometry="fermat_quartic")
    run_dir = tmp_path / "bookkeeping-run"
    metrics = run_experiment(
        bundle_dir=bundle_dir,
        model_name="local",
        target_name="hypersurface_fs_scalar",
        out_dir=run_dir,
        seed=11,
        benchmark_case_id="fermat_quartic",
    )
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics_json = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert config["split_strategy"] == "deterministic_random_train_validation_split"
    assert config["split_seed"] == 11
    assert config["benchmark_case_id"] == "fermat_quartic"
    assert metrics_json["benchmark_case_id"] == "fermat_quartic"
    assert metrics_json["feature_mode"] == "local"
    assert run_manifest["benchmark_case_id"] == "fermat_quartic"
    assert run_manifest["split_seed"] == 11
    assert metrics["target_status"] == "preferred"


def test_multiseed_sweep_aggregation_subset(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "phase7-multiseed"
    result = sweep_experiments(
        out_dir=sweep_dir,
        target_name="fs_scalar",
        seeds=[7, 11],
        n=24,
        include=["fermat_quartic"],
        test_size=0.25,
    )
    aggregated_df = result["aggregated_results"]
    assert len(result["results"]) == 4
    assert len(aggregated_df) == 2
    assert set(aggregated_df["run_count"]) == {2}
    assert "validation_score_mean" in aggregated_df.columns
    assert "validation_score_std" in aggregated_df.columns


def test_aggregated_outputs_and_summary_content(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "phase7-outputs"
    sweep_experiments(
        out_dir=sweep_dir,
        target_name="fs_scalar",
        seeds=[7, 11],
        n=24,
        include=["cefalu_lambda_1_0"],
        test_size=0.25,
    )
    aggregated_df = pd.read_csv(sweep_dir / "benchmark_aggregated.csv")
    aggregated_json = json.loads((sweep_dir / "benchmark_aggregated.json").read_text(encoding="utf-8"))
    aggregated_summary = (sweep_dir / "benchmark_aggregated_summary.md").read_text(encoding="utf-8")
    assert not aggregated_df.empty
    assert len(aggregated_json) == len(aggregated_df)
    assert "best model by mean validation score" in aggregated_summary
    assert "global beat local on all seeds" in aggregated_summary


def test_robustness_outputs_and_schema(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "phase9-robustness"
    result = sweep_experiments(
        out_dir=sweep_dir,
        preset_name="paper_v1_default",
        seeds=[7, 11],
        n=24,
        include=["fermat_quartic", "cefalu_lambda_1_0"],
    )
    robustness_df = pd.read_csv(sweep_dir / "benchmark_robustness.csv")
    robustness_json = json.loads((sweep_dir / "benchmark_robustness.json").read_text(encoding="utf-8"))
    robustness_summary = (sweep_dir / "benchmark_robustness_summary.md").read_text(encoding="utf-8")
    assert len(result["robustness"]) == 2
    assert not robustness_df.empty
    assert len(robustness_json) == len(robustness_df)
    assert "score_gap_mean" in robustness_df.columns
    assert "global_beats_local_all_seeds" in robustness_df.columns
    assert "hardest case by variance" in robustness_summary
    assert (sweep_dir / "paper_table.csv").exists()


def test_cli_experiment_sweep_smoke(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "cli-sweep"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "sweep",
            "--out",
            str(sweep_dir),
            "--target",
            "fs_scalar",
            "--seed",
            "7",
            "--n",
            "24",
            "--include",
            "cefalu_lambda_1_0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (sweep_dir / "benchmark_results.csv").exists()


def test_cli_experiment_multiseed_sweep_smoke(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "cli-multiseed-sweep"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "sweep",
            "--out",
            str(sweep_dir),
            "--target",
            "fs_scalar",
            "--seeds",
            "7",
            "11",
            "--n",
            "24",
            "--include",
            "fermat_quartic",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (sweep_dir / "benchmark_aggregated.csv").exists()


def test_cli_hypersurface_target_sweep_smoke(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "cli-hfs-sweep"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "sweep",
            "--out",
            str(sweep_dir),
            "--target",
            "hypersurface_fs_scalar",
            "--seeds",
            "7",
            "11",
            "--n",
            "24",
            "--include",
            "fermat_quartic",
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((sweep_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["target"] == "hypersurface_fs_scalar"
    assert (sweep_dir / "benchmark_aggregated_summary.md").exists()


def test_cli_preset_sweep_smoke(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "cli-preset-sweep"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "sweep",
            "--preset",
            "paper_v1_default",
            "--out",
            str(sweep_dir),
            "--n",
            "24",
            "--include",
            "fermat_quartic",
        ],
    )
    assert result.exit_code == 0, result.output
    protocol = json.loads((sweep_dir / "benchmark_protocol.json").read_text(encoding="utf-8"))
    assert protocol["preset_name"] == "paper_v1_default"
    assert (sweep_dir / "benchmark_robustness.csv").exists()


def test_benchmark_release_smoke(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-release"
    release = create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=False,
    )
    assert release["manifest"]["preset_name"] == "paper_v1_default"
    assert (release_dir / "release_manifest.json").exists()
    assert (release_dir / "release_protocol.json").exists()
    assert (release_dir / "final_results.csv").exists()
    assert (release_dir / "final_robustness.csv").exists()
    assert (release_dir / "results_memo.md").exists()


def test_release_hard_slice_outputs(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-release-hard"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=True,
        hard_slice_name="cefalu_hard_v1",
    )
    release_manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
    assert release_manifest["include_hard_slice"] is True
    assert (release_dir / "harder_slice" / "benchmark_manifest.json").exists()
    assert (release_dir / "harder_slice" / "benchmark_robustness.csv").exists()


def test_results_memo_creation(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-release-memo"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=True,
    )
    memo = (release_dir / "results_memo.md").read_text(encoding="utf-8")
    assert "## Main findings" in memo
    assert "## Limitations" in memo
    assert "hardest core case" in memo


def test_cli_release_smoke(tmp_path: Path) -> None:
    release_dir = tmp_path / "cli-release"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "release",
            "--preset",
            "paper_v1_default",
            "--out",
            str(release_dir),
            "--include-hard-slice",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (release_dir / "release_manifest.json").exists()
    assert (release_dir / "final_summary.md").exists()


def test_release_validation_smoke(tmp_path: Path) -> None:
    release_dir = tmp_path / "validated-release"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=True,
    )
    report = validate_benchmark_release(release_dir)
    assert report["passed"] is True
    assert (release_dir / "release_validation_report.json").exists()
    assert (release_dir / "release_validation_summary.md").exists()


def test_release_validation_missing_file_failure(tmp_path: Path) -> None:
    release_dir = tmp_path / "broken-release"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=False,
    )
    (release_dir / "final_results.csv").unlink()
    report = validate_benchmark_release(release_dir)
    assert report["passed"] is False
    assert any("final_results.csv" in failure for failure in report["failures"])


def test_release_manifest_protocol_consistency_and_versions(tmp_path: Path) -> None:
    release_dir = tmp_path / "versioned-release"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=True,
    )
    manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((release_dir / "release_protocol.json").read_text(encoding="utf-8"))
    assert manifest["release_version"] == protocol["release_version"]
    assert manifest["benchmark_contract_version"] == protocol["benchmark_contract_version"]
    assert manifest["preset_name"] == protocol["preset_name"]
    assert manifest["target_name"] == protocol["preset"]["target_name"]


def test_cli_validate_release_smoke(tmp_path: Path) -> None:
    release_dir = tmp_path / "cli-validated-release"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=False,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "validate-release",
            "--input",
            str(release_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((release_dir / "release_validation_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True


def test_paper_assets_build_smoke(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-assets-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=True,
    )
    built = build_paper_assets(input_dir=release_dir, out_dir=paper_dir)
    assert built["target_name"] == "hypersurface_fs_scalar"
    assert (paper_dir / "outline.md").exists()
    assert (paper_dir / "assets" / "core_results_table.csv").exists()
    assert (paper_dir / "assets" / "fig_core_scores.png").exists()


def test_paper_table_files_are_generated(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-table-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=False,
    )
    build_paper_assets(input_dir=release_dir, out_dir=paper_dir)
    core_table = pd.read_csv(paper_dir / "assets" / "core_results_table.csv")
    robustness_table = pd.read_csv(paper_dir / "assets" / "robustness_table.csv")
    assert "validation_score_mean" in core_table.columns
    assert "score_gap_mean" in robustness_table.columns
    assert "global_beats_local_all_seeds" in robustness_table.columns


def test_paper_figure_files_are_generated(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-figure-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=True,
    )
    build_paper_assets(input_dir=release_dir, out_dir=paper_dir)
    assert (paper_dir / "assets" / "fig_core_scores.png").stat().st_size > 0
    assert (paper_dir / "assets" / "fig_hardest_case.png").stat().st_size > 0


def test_manuscript_scaffold_content(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-notes-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=True,
    )
    build_paper_assets(input_dir=release_dir, out_dir=paper_dir)
    results_notes = (paper_dir / "results_notes.md").read_text(encoding="utf-8")
    reproducibility_notes = (paper_dir / "reproducibility_notes.md").read_text(encoding="utf-8")
    assert "Claim boundaries" in results_notes
    assert "hypersurface_fs_scalar" in results_notes
    assert "validate-release" in reproducibility_notes


def test_cli_build_paper_assets_smoke(tmp_path: Path) -> None:
    release_dir = tmp_path / "cli-paper-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(
        out_dir=release_dir,
        preset_name="paper_v1_default",
        include_hard_slice=True,
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "build-paper-assets",
            "--input",
            str(release_dir),
            "--out",
            str(paper_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (paper_dir / "abstract_draft.md").exists()
    assert (paper_dir / "assets" / "robustness_table.md").exists()


def test_validate_paper_assets_smoke(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-validation-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(out_dir=release_dir, preset_name="paper_v1_default", include_hard_slice=True)
    build_paper_assets(input_dir=release_dir, out_dir=paper_dir)
    report = validate_paper_assets(release_dir=release_dir, paper_dir=paper_dir)
    assert report["passed"] is True
    assert (paper_dir / "paper_validation_report.json").exists()
    assert (paper_dir / "paper_validation_summary.md").exists()


def test_validate_paper_assets_missing_file_failure(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-validation-broken-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(out_dir=release_dir, preset_name="paper_v1_default", include_hard_slice=True)
    build_paper_assets(input_dir=release_dir, out_dir=paper_dir)
    (paper_dir / "assets" / "core_results_table.csv").unlink()
    report = validate_paper_assets(release_dir=release_dir, paper_dir=paper_dir)
    assert report["passed"] is False
    assert any("core_results_table.csv" in failure for failure in report["failures"])


def test_validate_paper_assets_table_consistency_failure(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-validation-table-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(out_dir=release_dir, preset_name="paper_v1_default", include_hard_slice=True)
    build_paper_assets(input_dir=release_dir, out_dir=paper_dir)
    frame = pd.read_csv(paper_dir / "assets" / "core_results_table.csv")
    frame.loc[0, "validation_score_mean"] = frame.loc[0, "validation_score_mean"] + 1.0
    frame.to_csv(paper_dir / "assets" / "core_results_table.csv", index=False)
    report = validate_paper_assets(release_dir=release_dir, paper_dir=paper_dir)
    assert report["passed"] is False
    assert any("core table" in failure for failure in report["failures"])


def test_validate_paper_assets_finding_consistency_failure(tmp_path: Path) -> None:
    release_dir = tmp_path / "paper-validation-finding-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(out_dir=release_dir, preset_name="paper_v1_default", include_hard_slice=True)
    build_paper_assets(input_dir=release_dir, out_dir=paper_dir)
    results_notes_path = paper_dir / "results_notes.md"
    results_notes = results_notes_path.read_text(encoding="utf-8")
    results_notes_path.write_text(results_notes.replace("cefalu_lambda_1_0", "fermat_quartic", 1), encoding="utf-8")
    report = validate_paper_assets(release_dir=release_dir, paper_dir=paper_dir)
    assert report["passed"] is False
    assert any("hardest core case" in failure for failure in report["failures"])


def test_cli_validate_paper_assets_smoke(tmp_path: Path) -> None:
    release_dir = tmp_path / "cli-paper-validation-release"
    paper_dir = tmp_path / "paper"
    create_benchmark_release(out_dir=release_dir, preset_name="paper_v1_default", include_hard_slice=True)
    build_paper_assets(input_dir=release_dir, out_dir=paper_dir)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "experiments",
            "validate-paper-assets",
            "--release",
            str(release_dir),
            "--paper",
            str(paper_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((paper_dir / "paper_validation_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
