import json
from pathlib import Path

from geocydata.export.manifest import build_manifest


def test_manifest_contains_required_keys(tmp_path: Path) -> None:
    manifest = build_manifest(
        geometry="fermat_quartic",
        n_points=5,
        seed=7,
        output_dir=tmp_path,
        artifact_paths={"points": "points.parquet"},
        parameters={"geometry": "fermat_quartic", "n": 5, "seed": 7},
    )
    required = {
        "app_name",
        "app_version",
        "bundle_name",
        "bundle_path",
        "geometry",
        "parameters",
        "n_points",
        "seed",
        "created_at",
        "git_commit",
        "schema_version",
        "artifacts",
    }
    assert required.issubset(manifest)
    assert manifest["bundle_name"] == tmp_path.name
    assert manifest["bundle_path"] == str(tmp_path)
    json.dumps(manifest)


def test_manifest_stores_cefalu_lambda(tmp_path: Path) -> None:
    manifest = build_manifest(
        geometry="cefalu_quartic",
        n_points=5,
        seed=7,
        output_dir=tmp_path,
        artifact_paths={"points": "points.parquet"},
        parameters={"geometry": "cefalu_quartic", "lambda": 1.0, "n": 5, "seed": 7},
    )
    assert manifest["parameters"]["lambda"] == 1.0
