"""Bundle manifest helpers."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from geocydata.registry.cases import derive_case_id
from geocydata.utils.version import __version__


def get_git_commit(cwd: str | Path | None = None) -> str | None:
    """Return the current git commit hash if available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_manifest(
    *,
    geometry: str,
    n_points: int,
    seed: int | None,
    output_dir: Path,
    artifact_paths: dict[str, str],
    parameters: dict[str, object],
    case_id: str | None = None,
    protocol_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build bundle metadata for one generation run."""

    resolved_case_id = case_id
    if resolved_case_id is None:
        resolved_case_id = derive_case_id(geometry, parameters)
    return {
        "app_name": "GeoCYData",
        "app_version": __version__,
        "schema_version": "0.2",
        "bundle_name": output_dir.name,
        "bundle_path": str(output_dir),
        "geometry": geometry,
        "case_id": resolved_case_id,
        "parameters": parameters,
        "protocol_metadata": protocol_metadata or {},
        "n_points": n_points,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(output_dir.parent),
        "artifacts": artifact_paths,
    }


def write_manifest(manifest: dict[str, object], path: str | Path) -> Path:
    """Write manifest metadata to JSON."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target
