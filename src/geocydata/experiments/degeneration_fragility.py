from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from geocydata.export.parquet_io import write_parquet


REQUIRED_CASE_IDS = (
    "cefalu_lambda_0_50",
    "cefalu_lambda_0_75",
    "cefalu_lambda_0_90",
    "cefalu_lambda_1_0",
    "cefalu_lambda_1_10",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_sweep_dir(sweep_dir: str | Path | None) -> Path:
    repo_root = _resolve_repo_root()
    if sweep_dir is None:
        return (repo_root / "runs" / "cefalu_hard_regime_sweep_v1").resolve()
    return repo_root.joinpath(sweep_dir).resolve() if not Path(sweep_dir).is_absolute() else Path(sweep_dir).resolve()


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    repo_root = _resolve_repo_root()
    if output_dir is None:
        return (repo_root / "artifacts").resolve()
    return repo_root.joinpath(output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()


def _cefalu_gradient_norm(points: np.ndarray, lambda_value: float) -> np.ndarray:
    sum_sq = np.sum(points**2, axis=1)
    gradient = 4.0 * (points**3) - (4.0 * float(lambda_value) / 3.0) * points * sum_sq[:, None]
    return np.sqrt(np.sum(np.abs(gradient) ** 2, axis=1))


def _bundle_pointwise_frame(bundle_dir: Path) -> pd.DataFrame:
    manifest = _load_json(bundle_dir / "manifest.json")
    points = pd.read_parquet(bundle_dir / "points.parquet")
    lambda_value = float(manifest["parameters"]["lambda"])
    complex_points = np.column_stack(
        [
            points["z0_re"].to_numpy(dtype=float) + 1j * points["z0_im"].to_numpy(dtype=float),
            points["z1_re"].to_numpy(dtype=float) + 1j * points["z1_im"].to_numpy(dtype=float),
            points["z2_re"].to_numpy(dtype=float) + 1j * points["z2_im"].to_numpy(dtype=float),
            points["z3_re"].to_numpy(dtype=float) + 1j * points["z3_im"].to_numpy(dtype=float),
        ]
    )
    fragility_score = _cefalu_gradient_norm(complex_points, lambda_value)
    frame = pd.DataFrame(
        {
            "point_id": points["point_id"].astype(int),
            "case_id": str(manifest["case_id"]),
            "seed": int(manifest["seed"]),
            "lambda": lambda_value,
            "chart_id": points["chart_id"].astype(int),
            "fragility_score": fragility_score.astype(float),
        }
    )
    frame["fragile_cluster_id"] = pd.Series([pd.NA] * len(frame), dtype="Int64")
    return frame


def _pointwise_fragility(sweep_dir: Path, required_case_ids: tuple[str, ...] = REQUIRED_CASE_IDS) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for case_id in required_case_ids:
        case_root = sweep_dir / "bundles" / case_id
        if not case_root.exists():
            raise FileNotFoundError(f"Missing bundle directory for case '{case_id}': {case_root}")
        for seed_dir in sorted(path for path in case_root.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            frames.append(_bundle_pointwise_frame(seed_dir))
    if not frames:
        raise ValueError("No bundle pointwise data found for fragility export.")
    pointwise = pd.concat(frames, ignore_index=True)
    global_eps = float(pointwise["fragility_score"].quantile(0.10))
    pointwise["fragile_flag"] = pointwise["fragility_score"].astype(float) <= global_eps
    pointwise["fragility_threshold_eps"] = global_eps
    return pointwise.sort_values(["case_id", "seed", "point_id"]).reset_index(drop=True)


def _casewise_summary(pointwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, group in pointwise.groupby("case_id", dropna=False):
        ordered = group.sort_values(["seed", "point_id"]).reset_index(drop=True)
        rows.append(
            {
                "case_id": str(case_id),
                "lambda": float(ordered["lambda"].iloc[0]),
                "seed_count": int(ordered["seed"].nunique()),
                "point_count": int(len(ordered)),
                "fragility_eps": float(ordered["fragility_threshold_eps"].iloc[0]),
                "fragility_q05": float(ordered["fragility_score"].quantile(0.05)),
                "fragility_q10": float(ordered["fragility_score"].quantile(0.10)),
                "fragility_mean": float(ordered["fragility_score"].mean()),
                "fragility_frac_below_eps": float(ordered["fragile_flag"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["lambda", "case_id"]).reset_index(drop=True)


def _json_payload(*, sweep_dir: Path, pointwise: pd.DataFrame, casewise: pd.DataFrame) -> dict[str, Any]:
    eps = float(pointwise["fragility_threshold_eps"].iloc[0])
    return {
        "title": "GCY2-BLOB-01_geocy_fragility_proxy_export",
        "proxy_name": "cefalu_gradient_norm",
        "definition": {
            "score": "s(x) = ||grad F(x)||_2 on normalized sampled homogeneous quartic points",
            "meaning": "Lower values indicate more fragile / more degeneration-adjacent sampled geometry regions under this proxy.",
            "note": "This is a degeneration-sensitive geometry-side proxy, not a singularity invariant.",
        },
        "thresholding": {
            "fragile_flag_rule": "fragility_score <= eps",
            "eps_definition": "global 10th percentile of fragility_score over all sampled points in the required hard-regime sweep cases and seeds",
            "eps_value": eps,
            "fragile_cluster_id": "not assigned in this first export; column is present but null",
        },
        "source_sweep_dir": str(sweep_dir),
        "required_cases": list(REQUIRED_CASE_IDS),
        "case_records": casewise.to_dict(orient="records"),
    }


def export_degeneration_fragility(
    sweep_dir: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    resolved_sweep_dir = _resolve_sweep_dir(sweep_dir)
    resolved_output_dir = _resolve_output_dir(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    pointwise = _pointwise_fragility(resolved_sweep_dir)
    casewise = _casewise_summary(pointwise)

    csv_path = resolved_output_dir / "paper2_degeneration_sweep.csv"
    json_path = resolved_output_dir / "paper2_degeneration_sweep.json"
    parquet_path = resolved_output_dir / "paper2_pointwise_fragility.parquet"

    casewise.to_csv(csv_path, index=False)
    write_parquet(pointwise, parquet_path)
    json_path.write_text(
        json.dumps(_json_payload(sweep_dir=resolved_sweep_dir, pointwise=pointwise, casewise=casewise), indent=2),
        encoding="utf-8",
    )

    return {
        "sweep_dir": str(resolved_sweep_dir),
        "output_dir": str(resolved_output_dir),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "parquet_path": str(parquet_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = export_degeneration_fragility(args.sweep_dir, output_dir=args.out)
    print(f"Wrote degeneration fragility artifacts to {result['output_dir']}")


if __name__ == "__main__":
    main()
