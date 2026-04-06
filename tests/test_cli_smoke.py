from pathlib import Path

from typer.testing import CliRunner

from geocydata.cli.main import app

runner = CliRunner()

def test_cli_generates_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "demo"
    result = runner.invoke(
        app,
        [
            "generate",
            "bundle",
            "--geometry",
            "fermat_quartic",
            "--n",
            "16",
            "--seed",
            "7",
            "--out",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "points.parquet").exists()
    assert (output_dir / "invariants.parquet").exists()
    assert (output_dir / "validation_report.json").exists()
    assert (output_dir / "summary.md").exists()
    assert "GeoCYData bundle written to" in result.stdout


def test_cli_generates_cefalu_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "cefalu-demo"
    result = runner.invoke(
        app,
        [
            "generate",
            "bundle",
            "--geometry",
            "cefalu_quartic",
            "--lambda",
            "1.0",
            "--n",
            "200",
            "--seed",
            "7",
            "--out",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "points.parquet").exists()
    assert (output_dir / "invariants.parquet").exists()


def test_validate_bundle_reports_missing_points_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", "bundle", "--input", str(tmp_path)])
    assert result.exit_code == 2
    assert "missing required file: points.parquet" in result.output


def test_cli_generates_cefalu_orbit_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "cefalu-orbits"
    result = runner.invoke(
        app,
        [
            "generate",
            "orbits",
            "--geometry",
            "cefalu_quartic",
            "--lambda",
            "1.0",
            "--n",
            "12",
            "--seed",
            "7",
            "--out",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "points.parquet").exists()
    assert (output_dir / "invariants.parquet").exists()
    assert (output_dir / "orbits.parquet").exists()
    assert (output_dir / "symmetry_report.json").exists()
