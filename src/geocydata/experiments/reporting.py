"""Standardized reporting helpers for experiment sweeps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from geocydata.experiments.data import TARGET_METADATA
from geocydata.export.manifest import get_git_commit
from geocydata.utils.version import __version__


def build_benchmark_manifest(
    *,
    output_dir: Path,
    target_name: str,
    seeds: list[int],
    n_samples: int,
    test_size: float,
    cases: list[dict[str, object]],
    result_count: int,
    protocol: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build standardized metadata for one benchmark sweep."""

    return {
        "app_name": "GeoCYData",
        "app_version": __version__,
        "schema_version": "0.1",
        "protocol_version": "phase9",
        "benchmark_name": output_dir.name,
        "benchmark_path": str(output_dir),
        "target": target_name,
        "target_description": TARGET_METADATA[target_name]["description"],
        "target_status": TARGET_METADATA[target_name]["status"],
        "seeds": seeds,
        "n_samples_per_bundle": n_samples,
        "test_size": test_size,
        "split_strategy": "deterministic_random_train_validation_split",
        "cases": cases,
        "n_results": result_count,
        "protocol": protocol,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(output_dir.parent),
        "artifacts": {
            "benchmark_protocol": "benchmark_protocol.json",
            "benchmark_manifest": "benchmark_manifest.json",
            "benchmark_results_csv": "benchmark_results.csv",
            "benchmark_results_json": "benchmark_results.json",
            "benchmark_summary": "benchmark_summary.md",
            "benchmark_aggregated_csv": "benchmark_aggregated.csv",
            "benchmark_aggregated_json": "benchmark_aggregated.json",
            "benchmark_aggregated_summary": "benchmark_aggregated_summary.md",
            "benchmark_robustness_csv": "benchmark_robustness.csv",
            "benchmark_robustness_json": "benchmark_robustness.json",
            "benchmark_robustness_summary": "benchmark_robustness_summary.md",
            "paper_table_csv": "paper_table.csv",
        },
    }


def _json_ready_records(results_df: pd.DataFrame) -> list[dict[str, Any]]:
    records = results_df.to_dict(orient="records")
    normalized: list[dict[str, Any]] = []
    for record in records:
        normalized.append({key: (None if pd.isna(value) else value) for key, value in record.items()})
    return normalized


def write_benchmark_results(
    results_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    """Write standardized machine-readable benchmark outputs."""

    results_df.to_csv(output_dir / "benchmark_results.csv", index=False)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(_json_ready_records(results_df), indent=2),
        encoding="utf-8",
    )


def aggregate_benchmark_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate benchmark metrics across seeds for each case/model pair."""

    if results_df.empty:
        return pd.DataFrame(
            columns=[
                "benchmark_case",
                "case_label",
                "geometry",
                "lambda",
                "target",
                "target_status",
                "model",
                "n_samples",
                "train_size",
                "validation_size",
                "run_count",
                "validation_score_mean",
                "validation_score_std",
                "validation_mse_mean",
                "validation_mse_std",
                "validation_mae_mean",
                "validation_mae_std",
                "runtime_seconds_mean",
                "runtime_seconds_std",
            ]
        )

    grouped = (
        results_df.groupby(
            ["benchmark_case", "case_label", "geometry", "lambda", "target", "target_status", "model"],
            dropna=False,
        )
        .agg(
            n_samples=("n_samples", "first"),
            train_size=("train_size", "first"),
            validation_size=("validation_size", "first"),
            run_count=("seed", "nunique"),
            validation_score_mean=("validation_score", "mean"),
            validation_score_std=("validation_score", "std"),
            validation_mse_mean=("validation_mse", "mean"),
            validation_mse_std=("validation_mse", "std"),
            validation_mae_mean=("validation_mae", "mean"),
            validation_mae_std=("validation_mae", "std"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            runtime_seconds_std=("runtime_seconds", "std"),
        )
        .reset_index()
    )
    std_columns = [column for column in grouped.columns if column.endswith("_std")]
    grouped[std_columns] = grouped[std_columns].fillna(0.0)
    return grouped.sort_values(["benchmark_case", "model"]).reset_index(drop=True)


def write_benchmark_aggregated_results(
    aggregated_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    """Write aggregated benchmark metrics across seeds."""

    aggregated_df.to_csv(output_dir / "benchmark_aggregated.csv", index=False)
    (output_dir / "benchmark_aggregated.json").write_text(
        json.dumps(_json_ready_records(aggregated_df), indent=2),
        encoding="utf-8",
    )


def build_robustness_table(
    results_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build case-level robustness metrics comparing global and local models."""

    if results_df.empty or aggregated_df.empty:
        return pd.DataFrame(
            columns=[
                "benchmark_case",
                "case_label",
                "geometry",
                "lambda",
                "target",
                "target_status",
                "run_count",
                "local_validation_score_mean",
                "local_validation_score_std",
                "global_validation_score_mean",
                "global_validation_score_std",
                "score_gap_mean",
                "score_gap_std",
                "local_validation_mse_mean",
                "global_validation_mse_mean",
                "mse_gap_mean",
                "mse_gap_std",
                "global_beats_local_mean",
                "global_beats_local_all_seeds",
                "max_validation_score_std",
                "absolute_score_gap_mean",
            ]
        )

    rows: list[dict[str, object]] = []
    for case_id, case_group in results_df.groupby("benchmark_case", sort=True):
        local_runs = case_group[case_group["model"] == "local"].sort_values("seed")
        global_runs = case_group[case_group["model"] == "global"].sort_values("seed")
        local_agg = aggregated_df[(aggregated_df["benchmark_case"] == case_id) & (aggregated_df["model"] == "local")].iloc[0]
        global_agg = aggregated_df[(aggregated_df["benchmark_case"] == case_id) & (aggregated_df["model"] == "global")].iloc[0]
        score_gaps = (
            global_runs.set_index("seed")["validation_score"]
            - local_runs.set_index("seed")["validation_score"]
        )
        mse_gaps = (
            global_runs.set_index("seed")["validation_mse"]
            - local_runs.set_index("seed")["validation_mse"]
        )
        rows.append(
            {
                "benchmark_case": case_id,
                "case_label": local_agg["case_label"],
                "geometry": local_agg["geometry"],
                "lambda": local_agg["lambda"],
                "target": local_agg["target"],
                "target_status": local_agg["target_status"],
                "run_count": int(local_agg["run_count"]),
                "local_validation_score_mean": float(local_agg["validation_score_mean"]),
                "local_validation_score_std": float(local_agg["validation_score_std"]),
                "global_validation_score_mean": float(global_agg["validation_score_mean"]),
                "global_validation_score_std": float(global_agg["validation_score_std"]),
                "score_gap_mean": float(score_gaps.mean()),
                "score_gap_std": float(score_gaps.std(ddof=1) if len(score_gaps) > 1 else 0.0),
                "local_validation_mse_mean": float(local_agg["validation_mse_mean"]),
                "global_validation_mse_mean": float(global_agg["validation_mse_mean"]),
                "mse_gap_mean": float(mse_gaps.mean()),
                "mse_gap_std": float(mse_gaps.std(ddof=1) if len(mse_gaps) > 1 else 0.0),
                "global_beats_local_mean": bool(global_agg["validation_score_mean"] > local_agg["validation_score_mean"]),
                "global_beats_local_all_seeds": bool((score_gaps > 0.0).all()),
                "max_validation_score_std": float(
                    max(local_agg["validation_score_std"], global_agg["validation_score_std"])
                ),
                "absolute_score_gap_mean": float(abs(score_gaps.mean())),
            }
        )

    robustness_df = pd.DataFrame(rows).sort_values("benchmark_case").reset_index(drop=True)
    if not robustness_df.empty:
        hardest_variance = robustness_df["max_validation_score_std"].idxmax()
        hardest_separation = robustness_df["absolute_score_gap_mean"].idxmin()
        robustness_df["hardest_by_variance"] = False
        robustness_df["hardest_by_smallest_separation"] = False
        robustness_df.loc[hardest_variance, "hardest_by_variance"] = True
        robustness_df.loc[hardest_separation, "hardest_by_smallest_separation"] = True
    return robustness_df


def write_benchmark_robustness_outputs(
    robustness_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:
    """Write robustness outputs and a compact paper-style table."""

    robustness_df.to_csv(output_dir / "benchmark_robustness.csv", index=False)
    (output_dir / "benchmark_robustness.json").write_text(
        json.dumps(_json_ready_records(robustness_df), indent=2),
        encoding="utf-8",
    )
    paper_table = robustness_df[
        [
            "benchmark_case",
            "case_label",
            "local_validation_score_mean",
            "local_validation_score_std",
            "global_validation_score_mean",
            "global_validation_score_std",
            "score_gap_mean",
            "global_beats_local_all_seeds",
        ]
    ]
    paper_table.to_csv(output_dir / "paper_table.csv", index=False)


def build_benchmark_summary_markdown(
    results_df: pd.DataFrame,
    *,
    target_name: str,
    seeds: list[int],
) -> str:
    """Render a markdown summary for the per-seed benchmark sweep."""

    if results_df.empty:
        return "# GeoCYData Benchmark Summary\n\nNo benchmark results were recorded.\n"

    lines = [
        "# GeoCYData Benchmark Summary",
        "",
        "## Protocol",
        "",
        f"- target: `{target_name}`",
        f"- target description: `{TARGET_METADATA[target_name]['description']}`",
        f"- target status: `{TARGET_METADATA[target_name]['status']}`",
        f"- seeds: `{', '.join(str(seed) for seed in seeds)}`",
        "",
        "## Case summary",
        "",
    ]

    for (_, seed), group in results_df.groupby(["benchmark_case", "seed"], sort=True):
        local_row = group[group["model"] == "local"].iloc[0]
        global_row = group[group["model"] == "global"].iloc[0]
        best_model = "global" if global_row["validation_score"] >= local_row["validation_score"] else "local"
        score_delta = float(global_row["validation_score"] - local_row["validation_score"])
        mse_delta = float(global_row["validation_mse"] - local_row["validation_mse"])
        if score_delta > 0.0:
            narrative = "Global invariant features outperformed the local baseline on validation score."
        elif score_delta < 0.0:
            narrative = "The local affine baseline outperformed the global invariant model on validation score."
        else:
            narrative = "The local and global models matched on validation score for this case."

        case_label = str(local_row["case_label"])
        lambda_value = local_row["lambda"]
        lambda_text = "none" if pd.isna(lambda_value) else str(lambda_value)
        lines.extend(
            [
                f"### {case_label} (seed {seed})",
                "",
                f"- benchmark case id: `{local_row['benchmark_case']}`",
                f"- best model: `{best_model}`",
                f"- local validation score: `{local_row['validation_score']:.6f}`",
                f"- global validation score: `{global_row['validation_score']:.6f}`",
                f"- validation score delta (global - local): `{score_delta:.6f}`",
                f"- local validation MSE: `{local_row['validation_mse']:.6e}`",
                f"- global validation MSE: `{global_row['validation_mse']:.6e}`",
                f"- validation MSE delta (global - local): `{mse_delta:.6e}`",
                f"- lambda: `{lambda_text}`",
                f"- note: {narrative}",
                "",
            ]
        )

    lines.extend(
        [
            "## Tidy results",
            "",
            "| case | seed | model | validation_score | validation_mse | validation_mae | runtime_seconds |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in results_df.sort_values(["benchmark_case", "seed", "model"]).to_dict(orient="records"):
        lines.append(
            "| "
            f"{row['benchmark_case']} | {row['seed']} | {row['model']} | "
            f"{row['validation_score']:.6f} | {row['validation_mse']:.6e} | "
            f"{row['validation_mae']:.6e} | {row['runtime_seconds']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_benchmark_aggregated_summary_markdown(
    aggregated_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    target_name: str,
    seeds: list[int],
) -> str:
    """Render a markdown summary of multi-seed aggregate benchmark results."""

    if aggregated_df.empty:
        return "# GeoCYData Aggregated Benchmark Summary\n\nNo aggregated benchmark results were recorded.\n"

    lines = [
        "# GeoCYData Aggregated Benchmark Summary",
        "",
        "## Protocol",
        "",
        f"- target: `{target_name}`",
        f"- target description: `{TARGET_METADATA[target_name]['description']}`",
        f"- target status: `{TARGET_METADATA[target_name]['status']}`",
        f"- seeds: `{', '.join(str(seed) for seed in seeds)}`",
        f"- seed count: `{len(seeds)}`",
        "",
        "## Aggregated case summary",
        "",
    ]

    for case_id, case_group in aggregated_df.groupby("benchmark_case", sort=True):
        local_row = case_group[case_group["model"] == "local"].iloc[0]
        global_row = case_group[case_group["model"] == "global"].iloc[0]
        best_model = "global" if global_row["validation_score_mean"] >= local_row["validation_score_mean"] else "local"
        score_gap = float(global_row["validation_score_mean"] - local_row["validation_score_mean"])
        mse_gap = float(global_row["validation_mse_mean"] - local_row["validation_mse_mean"])
        case_results = results_df[results_df["benchmark_case"] == case_id]
        paired = case_results.pivot(index="seed", columns="model", values="validation_score")
        global_won_all = bool((paired["global"] > paired["local"]).all())
        lambda_value = local_row["lambda"]
        lambda_text = "none" if pd.isna(lambda_value) else str(lambda_value)
        if global_won_all:
            narrative = "Global invariant features beat the local baseline on every recorded seed."
        elif score_gap > 0.0:
            narrative = "Global invariant features led on mean validation score, but not on every seed."
        elif score_gap < 0.0:
            narrative = "The local affine baseline led on mean validation score."
        else:
            narrative = "The local and global models matched on mean validation score."

        lines.extend(
            [
                f"### {local_row['case_label']}",
                "",
                f"- benchmark case id: `{case_id}`",
                f"- best model by mean validation score: `{best_model}`",
                f"- mean local validation score: `{local_row['validation_score_mean']:.6f}` +/- `{local_row['validation_score_std']:.6f}`",
                f"- mean global validation score: `{global_row['validation_score_mean']:.6f}` +/- `{global_row['validation_score_std']:.6f}`",
                f"- validation score gap (global - local): `{score_gap:.6f}`",
                f"- mean local validation MSE: `{local_row['validation_mse_mean']:.6e}` +/- `{local_row['validation_mse_std']:.6e}`",
                f"- mean global validation MSE: `{global_row['validation_mse_mean']:.6e}` +/- `{global_row['validation_mse_std']:.6e}`",
                f"- validation MSE gap (global - local): `{mse_gap:.6e}`",
                f"- global beat local on all seeds: `{global_won_all}`",
                f"- lambda: `{lambda_text}`",
                "- target note: This target is more hypersurface-aware than the older ambient `fs_scalar` proxy because it uses the local hypersurface gradient.",
                f"- note: {narrative}",
                "",
            ]
        )

    lines.extend(
        [
            "## Aggregated results",
            "",
            "| case | model | run_count | validation_score_mean | validation_score_std | validation_mse_mean | validation_mse_std |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in aggregated_df.to_dict(orient="records"):
        lines.append(
            "| "
            f"{row['benchmark_case']} | {row['model']} | {row['run_count']} | "
            f"{row['validation_score_mean']:.6f} | {row['validation_score_std']:.6f} | "
            f"{row['validation_mse_mean']:.6e} | {row['validation_mse_std']:.6e} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_benchmark_robustness_summary_markdown(
    robustness_df: pd.DataFrame,
    *,
    target_name: str,
    preset_name: str | None,
) -> str:
    """Render a paper-style robustness summary across benchmark cases."""

    if robustness_df.empty:
        return "# GeoCYData Robustness Summary\n\nNo robustness results were recorded.\n"

    hardest_variance_row = robustness_df.loc[robustness_df["max_validation_score_std"].idxmax()]
    hardest_gap_row = robustness_df.loc[robustness_df["absolute_score_gap_mean"].idxmin()]
    global_mean_wins = int(robustness_df["global_beats_local_mean"].sum())
    global_seed_wins = int(robustness_df["global_beats_local_all_seeds"].sum())

    lines = [
        "# GeoCYData Robustness Summary",
        "",
        "## Protocol",
        "",
        f"- target: `{target_name}`",
        f"- target description: `{TARGET_METADATA[target_name]['description']}`",
        f"- preset: `{preset_name}`",
        "",
        "## Overall",
        "",
        f"- cases where global beats local on mean validation score: `{global_mean_wins}` / `{len(robustness_df)}`",
        f"- cases where global beats local on all seeds: `{global_seed_wins}` / `{len(robustness_df)}`",
        f"- hardest case by variance: `{hardest_variance_row['benchmark_case']}`",
        f"- hardest case by smallest local/global separation: `{hardest_gap_row['benchmark_case']}`",
        "",
        "## Case table",
        "",
        "| case | global_beats_local_mean | global_beats_local_all_seeds | score_gap_mean | score_gap_std | mse_gap_mean |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in robustness_df.to_dict(orient="records"):
        lines.append(
            "| "
            f"{row['benchmark_case']} | {row['global_beats_local_mean']} | {row['global_beats_local_all_seeds']} | "
            f"{row['score_gap_mean']:.6f} | {row['score_gap_std']:.6f} | {row['mse_gap_mean']:.6e} |"
        )
    lines.append("")
    return "\n".join(lines)
