import json
from pathlib import Path

from typer.testing import CliRunner

from geocydata.cli.main import app
from geocydata.experiments.data import build_target, load_bundle_dataset, prepare_experiment_matrix
from geocydata.experiments.runner import compare_experiments, run_experiment
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


def test_geometry_target_computation_smoke(tmp_path: Path) -> None:
    bundle_dir = create_bundle(tmp_path)
    dataset = load_bundle_dataset(bundle_dir)
    target, target_name = build_target(dataset, "fs_scalar")
    assert target_name == "fs_scalar"
    assert target.shape[0] == dataset.points_df.shape[0]
    assert float(target.min()) > 0.0


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
