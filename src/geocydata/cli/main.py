"""Typer CLI for GeoCYData."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from geocydata.export.manifest import build_manifest, write_manifest
from geocydata.export.parquet_io import write_parquet
from geocydata.registry.geometries import GEOMETRIES, get_geometry, list_geometries
from geocydata.sampling.point_sampler import generate_sample_batch
from geocydata.utils.logging import configure_logging
from geocydata.utils.paths import ensure_directory
from geocydata.validation.reports import build_validation_report

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

app.add_typer(geometry_app, name="geometry")
app.add_typer(generate_app, name="generate")
app.add_typer(validate_app, name="validate")


def _bundle_summary(
    *,
    geometry_name: str,
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
    return f"""# GeoCYData Bundle Summary

## Geometry

- name: `{geometry_name}`
- points: `{n_points}`

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


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging.")) -> None:
    """GeoCYData command-line interface."""

    configure_logging(verbose=verbose)


@geometry_app.command("list")
def geometry_list() -> None:
    """List the geometries available in this installation."""

    for name in list_geometries():
        typer.echo(f"{name}: {GEOMETRIES[name].description}")


@generate_app.command("bundle")
def generate_bundle(
    geometry: str = typer.Option(
        ...,
        "--geometry",
        help="Registered geometry name, for example `fermat_quartic`.",
    ),
    n: int = typer.Option(..., "--n", min=1, help="Number of projective points to sample."),
    seed: int = typer.Option(0, "--seed", help="Random seed for reproducibility."),
    out: Path = typer.Option(
        ...,
        "--out",
        help="Output bundle directory. It will be created if it does not exist.",
    ),
) -> None:
    """Generate a benchmark bundle with points, invariants, validation, and metadata."""

    try:
        resolved_geometry = get_geometry(geometry)
    except KeyError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc

    output_dir = ensure_directory(out)
    batch = generate_sample_batch(geometry_name=geometry, n=n, seed=seed)
    report = build_validation_report(batch.points, n_points=n, seed=seed)

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
            "n": n,
            "seed": seed,
        },
    )
    write_manifest(manifest, output_dir / artifact_paths["manifest"])
    typer.echo(f"GeoCYData bundle written to {output_dir}")
    typer.echo("Artifacts: manifest.json, points.parquet, invariants.parquet, validation_report.json, summary.md")


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
    report = build_validation_report(points, n_points=len(points_df), seed=None)
    typer.echo(json.dumps(report, indent=2))


def run() -> None:
    """Console script entrypoint."""

    app()
