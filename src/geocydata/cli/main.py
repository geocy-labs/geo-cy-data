"""Typer CLI for GeoCYData."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from geocydata.experiments.runner import compare_experiments, run_experiment
from geocydata.export.manifest import build_manifest, write_manifest
from geocydata.export.parquet_io import write_parquet
from geocydata.registry.geometries import GEOMETRIES, get_geometry, list_geometries
from geocydata.sampling.point_sampler import generate_sample_batch
from geocydata.symmetry.canonicalize import canonical_key_string
from geocydata.utils.logging import configure_logging
from geocydata.utils.paths import ensure_directory
from geocydata.validation.reports import build_validation_report
from geocydata.validation.symmetry_checks import (
    build_canonical_invariants_dataframe,
    build_orbits_dataframe,
    build_symmetry_report,
)

app = typer.Typer(
    help=(
        "Generate and validate projective hypersurface benchmark bundles.\n\n"
        "Start with `geocydata geometry list`, then generate a reproducible bundle with "
        "`geocydata generate bundle`."
    ),
    no_args_is_help=True,
)
geometry_app = typer.Typer(help="List and inspect registered benchmark geometries.", no_args_is_help=True)
generate_app = typer.Typer(
    help="Generate reproducible dataset bundles for registered geometries.",
    no_args_is_help=True,
)
validate_app = typer.Typer(
    help="Validate an existing GeoCYData bundle directory.",
    no_args_is_help=True,
)
experiments_app = typer.Typer(
    help="Run lightweight representation-comparison experiments on GeoCYData bundles.",
    no_args_is_help=True,
)

app.add_typer(geometry_app, name="geometry")
app.add_typer(generate_app, name="generate")
app.add_typer(validate_app, name="validate")
app.add_typer(experiments_app, name="experiments")


def _bundle_summary(
    *,
    geometry_name: str,
    parameters: dict[str, object],
    n_points: int,
    artifact_paths: dict[str, str],
    report: dict,
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


def _orbit_summary(
    *,
    geometry_name: str,
    parameters: dict[str, object],
    n_points: int,
    artifact_paths: dict[str, str],
    report: dict[str, object],
) -> str:
    artifacts = "\n".join(f"- `{name}`: `{path}`" for name, path in artifact_paths.items())
    parameter_lines = "\n".join(f"- {name}: `{value}`" for name, value in parameters.items()) or "- none"
    warnings = report["warnings"] or ["none"]
    warning_text = "\n".join(f"- {warning}" for warning in warnings)
    return f"""# GeoCYData Orbit Summary

## Geometry

- name: `{geometry_name}`
- points: `{n_points}`

## Parameters

{parameter_lines}

## Orbit statistics

- group size: `{report["group_size"]}`
- min orbit size: `{report["orbit_size"]["min"]}`
- max orbit size: `{report["orbit_size"]["max"]}`
- mean orbit size: `{report["orbit_size"]["mean"]:.2f}`

## Symmetry validation

- max residual preservation delta: `{report["residual_preservation"]["max"]:.3e}`
- max canonicalization drift: `{report["canonicalization_drift"]["max"]:.3e}`
- max canonical invariant drift: `{report["canonical_invariant_drift"]["max"]:.3e}`
- passed: `{report["passed"]}`

## Warnings

{warning_text}

## Artifacts

{artifacts}
"""


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging.")) -> None:
    """GeoCYData command-line interface."""

    configure_logging(verbose=verbose)


@geometry_app.command("list")
def geometry_list() -> None:
    """List the geometries available in this installation."""

    for name in list_geometries():
        parameter_names = ", ".join(GEOMETRIES[name].metadata()["parameter_schema"]) or "none"
        typer.echo(f"{name}: {GEOMETRIES[name].description} (parameters: {parameter_names})")


@geometry_app.command("show")
def geometry_show(
    geometry: str = typer.Option(..., "--geometry", help="Registered geometry name."),
    lambda_value: float | None = typer.Option(
        None,
        "--lambda",
        help="Required family parameter for `cefalu_quartic`; omit for `fermat_quartic`.",
    ),
) -> None:
    """Show metadata for one registered geometry."""

    try:
        resolved_geometry = get_geometry(geometry)
        parameters = resolved_geometry.validate_parameters({"lambda": lambda_value})
    except (KeyError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    payload = resolved_geometry.metadata()
    payload["parameters"] = parameters
    typer.echo(json.dumps(payload, indent=2))


@generate_app.command("bundle")
def generate_bundle(
    geometry: str = typer.Option(
        ...,
        "--geometry",
        help="Registered geometry name, for example `fermat_quartic`.",
    ),
    n: int = typer.Option(..., "--n", min=1, help="Number of projective points to sample."),
    seed: int = typer.Option(0, "--seed", help="Random seed for reproducibility."),
    lambda_value: float | None = typer.Option(
        None,
        "--lambda",
        help="Required family parameter for `cefalu_quartic`; omit for `fermat_quartic`.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output bundle directory. It will be created if it does not exist.",
    ),
) -> None:
    """Generate a benchmark bundle with points, invariants, validation, and metadata."""

    try:
        resolved_geometry = get_geometry(geometry)
        resolved_parameters = resolved_geometry.validate_parameters({"lambda": lambda_value})
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    output_dir = ensure_directory(out)
    batch = generate_sample_batch(
        geometry_name=geometry,
        n=n,
        seed=seed,
        parameters=resolved_parameters,
    )
    report = build_validation_report(
        batch.points,
        geometry_name=geometry,
        parameters=resolved_parameters,
        n_points=n,
        seed=seed,
    )

    artifact_paths = {
        "manifest": "manifest.json",
        "points": "points.parquet",
        "invariants": "invariants.parquet",
        "validation_report": "validation_report.json",
        "summary": "summary.md",
    }
    write_parquet(batch.points_df, output_dir / artifact_paths["points"])
    write_parquet(batch.invariants_df, output_dir / artifact_paths["invariants"])
    (output_dir / artifact_paths["validation_report"]).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output_dir / artifact_paths["summary"]).write_text(
        _bundle_summary(
            geometry_name=geometry,
            parameters=resolved_parameters,
            n_points=n,
            artifact_paths=artifact_paths,
            report=report,
        ),
        encoding="utf-8",
    )
    manifest = build_manifest(
        geometry=geometry,
        n_points=n,
        seed=seed,
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        parameters={
            "geometry": geometry,
            "geometry_description": resolved_geometry.description,
            **resolved_parameters,
            "n": n,
            "seed": seed,
        },
    )
    write_manifest(manifest, output_dir / artifact_paths["manifest"])
    typer.echo(f"GeoCYData bundle written to {output_dir}")
    typer.echo("Artifacts: manifest.json, points.parquet, invariants.parquet, validation_report.json, summary.md")


@generate_app.command("orbits")
def generate_orbits(
    geometry: str = typer.Option(
        ...,
        "--geometry",
        help="Registered geometry name. Phase 3 orbit generation currently supports `cefalu_quartic`.",
    ),
    n: int = typer.Option(..., "--n", min=1, help="Number of base projective points to sample."),
    seed: int = typer.Option(0, "--seed", help="Random seed for reproducibility."),
    lambda_value: float | None = typer.Option(
        None,
        "--lambda",
        help="Required family parameter for `cefalu_quartic`.",
    ),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output orbit bundle directory. It will be created if it does not exist.",
    ),
) -> None:
    """Generate symmetry-orbit data for the Cefalu quartic family."""

    if geometry != "cefalu_quartic":
        typer.secho(
            "Phase 3 orbit generation currently supports only 'cefalu_quartic'.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        resolved_geometry = get_geometry(geometry)
        resolved_parameters = resolved_geometry.validate_parameters({"lambda": lambda_value})
    except (KeyError, ValueError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    output_dir = ensure_directory(out)
    batch = generate_sample_batch(
        geometry_name=geometry,
        n=n,
        seed=seed,
        parameters=resolved_parameters,
    )
    lambda_float = float(resolved_parameters["lambda"])
    orbits_df = build_orbits_dataframe(batch.points, lambda_value=lambda_float)
    canonical_invariants_df = build_canonical_invariants_dataframe(batch.points, lambda_value=lambda_float)
    symmetry_report = build_symmetry_report(batch.points, lambda_value=lambda_float)

    artifact_paths = {
        "manifest": "manifest.json",
        "points": "points.parquet",
        "invariants": "invariants.parquet",
        "orbits": "orbits.parquet",
        "symmetry_report": "symmetry_report.json",
        "summary": "summary.md",
    }
    write_parquet(batch.points_df, output_dir / artifact_paths["points"])
    write_parquet(canonical_invariants_df, output_dir / artifact_paths["invariants"])
    write_parquet(orbits_df, output_dir / artifact_paths["orbits"])
    (output_dir / artifact_paths["symmetry_report"]).write_text(
        json.dumps(symmetry_report, indent=2),
        encoding="utf-8",
    )
    (output_dir / artifact_paths["summary"]).write_text(
        _orbit_summary(
            geometry_name=geometry,
            parameters=resolved_parameters,
            n_points=n,
            artifact_paths=artifact_paths,
            report=symmetry_report,
        ),
        encoding="utf-8",
    )
    manifest = build_manifest(
        geometry=geometry,
        n_points=n,
        seed=seed,
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        parameters={
            "geometry": geometry,
            "geometry_description": resolved_geometry.description,
            **resolved_parameters,
            "n": n,
            "seed": seed,
            "orbit_mode": True,
        },
    )
    write_manifest(manifest, output_dir / artifact_paths["manifest"])
    typer.echo(f"GeoCYData orbit bundle written to {output_dir}")
    typer.echo("Artifacts: manifest.json, points.parquet, invariants.parquet, orbits.parquet, symmetry_report.json, summary.md")


@validate_app.command("bundle")
def validate_bundle(
    input: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        file_okay=False,
        help="Path to an existing GeoCYData bundle directory.",
    )
) -> None:
    """Validate an existing bundle from its exported points table."""

    points_path = input / "points.parquet"
    if not points_path.exists():
        typer.secho(
            f"Bundle directory is missing required file: {points_path.name}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        points_df = pd.read_parquet(points_path)
    except Exception as exc:
        typer.secho(
            f"Could not read {points_path.name}: {exc}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from exc

    manifest_path = input / "manifest.json"
    geometry_name = "fermat_quartic"
    parameters: dict[str, object] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            geometry_name = str(manifest.get("geometry", geometry_name))
            parameters = dict(manifest.get("parameters", {}))
        except Exception as exc:
            typer.secho(
                f"Could not read manifest.json: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1) from exc

    coords = []
    for idx in range(4):
        real_col = f"z{idx}_re"
        imag_col = f"z{idx}_im"
        if real_col not in points_df.columns or imag_col not in points_df.columns:
            typer.secho(
                f"Bundle points table is missing required columns: {real_col}, {imag_col}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        coords.append(points_df[real_col].to_numpy() + 1j * points_df[imag_col].to_numpy())
    points = np.column_stack(coords)
    report = build_validation_report(
        points,
        geometry_name=geometry_name,
        parameters=parameters,
        n_points=len(points_df),
        seed=None,
    )
    typer.echo(json.dumps(report, indent=2))


@validate_app.command("symmetry")
def validate_symmetry(
    input: Path = typer.Option(
        ...,
        "--input",
        exists=True,
        file_okay=False,
        help="Path to an existing GeoCYData orbit bundle directory.",
    )
) -> None:
    """Validate a symmetry-orbit bundle for the Cefalu quartic family."""

    points_path = input / "points.parquet"
    orbits_path = input / "orbits.parquet"
    manifest_path = input / "manifest.json"
    for required_path in (points_path, orbits_path, manifest_path):
        if not required_path.exists():
            typer.secho(
                f"Orbit bundle is missing required file: {required_path.name}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

    try:
        points_df = pd.read_parquet(points_path)
        pd.read_parquet(orbits_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        typer.secho(f"Could not read orbit bundle: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    parameters = dict(manifest.get("parameters", {}))
    geometry_name = str(manifest.get("geometry", ""))
    if geometry_name != "cefalu_quartic":
        typer.secho(
            "Symmetry validation currently supports only 'cefalu_quartic' orbit bundles.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    if parameters.get("lambda") is None:
        typer.secho("Orbit bundle manifest is missing the Cefalu 'lambda' parameter.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)

    coords = []
    for idx in range(4):
        coords.append(points_df[f"z{idx}_re"].to_numpy() + 1j * points_df[f"z{idx}_im"].to_numpy())
    points = np.column_stack(coords)
    report = build_symmetry_report(points, lambda_value=float(parameters["lambda"]))
    typer.echo(json.dumps(report, indent=2))


def run() -> None:
    """Console script entrypoint."""

    app()


@experiments_app.command("run")
def experiments_run(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=False, help="Input bundle directory."),
    model: str = typer.Option(..., "--model", help="Experiment model: `local` or `global`."),
    target: str = typer.Option(
        "fs_scalar",
        "--target",
        help="Experiment target: `fs_scalar` (preferred) or `invariant_weighted_sum` (debug/convenience).",
    ),
    out: Path = typer.Option(..., "--out", help="Output run directory."),
    seed: int = typer.Option(7, "--seed", help="Deterministic train/validation split seed."),
    test_size: float = typer.Option(0.2, "--test-size", min=0.05, max=0.5, help="Validation fraction."),
) -> None:
    """Run one lightweight experiment against a GeoCYData bundle."""

    if model not in {"local", "global"}:
        typer.secho("Experiment model must be either 'local' or 'global'.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    try:
        metrics = run_experiment(
            bundle_dir=bundle,
            model_name=model,
            target_name=target,
            out_dir=out,
            seed=seed,
            test_size=test_size,
        )
    except Exception as exc:
        typer.secho(f"Experiment run failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Experiment run written to {out}")
    typer.echo(
        f"Validation score: {metrics['validation_score']:.6f}, "
        f"validation MSE: {metrics['validation_mse']:.6e}"
    )


@experiments_app.command("compare")
def experiments_compare(
    bundle: Path = typer.Option(..., "--bundle", exists=True, file_okay=False, help="Input bundle directory."),
    target: str = typer.Option(
        "fs_scalar",
        "--target",
        help="Experiment target: `fs_scalar` (preferred) or `invariant_weighted_sum` (debug/convenience).",
    ),
    out: Path = typer.Option(..., "--out", help="Output comparison directory."),
    seed: int = typer.Option(7, "--seed", help="Deterministic train/validation split seed."),
    test_size: float = typer.Option(0.2, "--test-size", min=0.05, max=0.5, help="Validation fraction."),
) -> None:
    """Run local and global experiment modes and compare their results."""

    try:
        comparison = compare_experiments(
            bundle_dir=bundle,
            out_dir=out,
            target_name=target,
            seed=seed,
            test_size=test_size,
        )
    except Exception as exc:
        typer.secho(f"Experiment comparison failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Experiment comparison written to {out}")
    typer.echo(
        f"Local val score: {comparison['local']['validation_score']:.6f}, "
        f"Global val score: {comparison['global']['validation_score']:.6f}"
    )
