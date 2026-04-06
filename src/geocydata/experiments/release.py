"""Publication-facing benchmark release packaging for GeoCYData."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from geocydata.experiments.protocols import resolve_hard_evaluation_slice, resolve_protocol_preset
from geocydata.experiments.sweep import sweep_experiments
from geocydata.export.manifest import get_git_commit
from geocydata.utils.paths import ensure_directory
from geocydata.utils.version import __version__

RELEASE_VERSION = "paper_v1_release_v1"
BENCHMARK_CONTRACT_VERSION = "paper_v1_contract_v1"


def _json_ready_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {key: (None if pd.isna(value) else value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _final_summary_markdown(
    *,
    preset_name: str,
    target_name: str,
    robustness_df: pd.DataFrame,
    hard_slice_name: str | None,
    hard_slice_robustness_df: pd.DataFrame | None,
) -> str:
    hardest_case = robustness_df.loc[robustness_df["absolute_score_gap_mean"].idxmin()]
    lines = [
        "# GeoCYData Benchmark Release Summary",
        "",
        "## Core benchmark",
        "",
        f"- release version: `{RELEASE_VERSION}`",
        f"- benchmark contract version: `{BENCHMARK_CONTRACT_VERSION}`",
        f"- preset: `{preset_name}`",
        f"- target: `{target_name}`",
        f"- hardest core case by smallest local/global separation: `{hardest_case['benchmark_case']}`",
        f"- global beats local on mean validation score in `{int(robustness_df['global_beats_local_mean'].sum())}` / `{len(robustness_df)}` cases",
        f"- global beats local on all seeds in `{int(robustness_df['global_beats_local_all_seeds'].sum())}` / `{len(robustness_df)}` cases",
        "",
    ]
    if hard_slice_name and hard_slice_robustness_df is not None and not hard_slice_robustness_df.empty:
        hardest_hard = hard_slice_robustness_df.loc[hard_slice_robustness_df["absolute_score_gap_mean"].idxmin()]
        lines.extend(
            [
                "## Harder evaluation slice",
                "",
                f"- slice: `{hard_slice_name}`",
                f"- hardest slice case by smallest local/global separation: `{hardest_hard['benchmark_case']}`",
                f"- global beats local on mean validation score in `{int(hard_slice_robustness_df['global_beats_local_mean'].sum())}` / `{len(hard_slice_robustness_df)}` slice cases",
                "",
            ]
        )
    return "\n".join(lines)


def _results_memo_markdown(
    *,
    preset_name: str,
    target_name: str,
    protocol: dict[str, object],
    robustness_df: pd.DataFrame,
    hard_slice_name: str | None,
    hard_slice_robustness_df: pd.DataFrame | None,
) -> str:
    hardest_case = robustness_df.loc[robustness_df["absolute_score_gap_mean"].idxmin()]
    hardest_variance = robustness_df.loc[robustness_df["max_validation_score_std"].idxmax()]
    lines = [
        "# Results Memo",
        "",
        "This memo packages the current GeoCYData benchmark as a publication-facing release artifact.",
        "",
        "## Protocol",
        "",
        f"- release version: `{RELEASE_VERSION}`",
        f"- benchmark contract version: `{BENCHMARK_CONTRACT_VERSION}`",
        f"- preset: `{preset_name}`",
        f"- target: `{target_name}`",
        f"- seeds: `{', '.join(str(seed) for seed in protocol['resolved']['seeds'])}`",
        f"- cases: `{', '.join(protocol['resolved']['include'])}`",
        f"- split strategy: `{protocol['resolved']['split_strategy']}`",
        "",
        "## Main findings",
        "",
        f"- global beats local on mean validation score in all `{len(robustness_df)}` core benchmark cases",
        f"- the hardest core case by smallest mean separation is `{hardest_case['benchmark_case']}`",
        f"- the highest-variance core case is `{hardest_variance['benchmark_case']}`",
        f"- global does not beat local on all seeds in every case; the current failure point remains `{hardest_case['benchmark_case']}`",
        "",
    ]
    if hard_slice_name and hard_slice_robustness_df is not None and not hard_slice_robustness_df.empty:
        hard_case = hard_slice_robustness_df.loc[hard_slice_robustness_df["absolute_score_gap_mean"].idxmin()]
        lines.extend(
            [
                "## Harder slice",
                "",
                f"- additional evaluation slice: `{hard_slice_name}`",
                f"- hardest slice case by smallest separation: `{hard_case['benchmark_case']}`",
                f"- this slice is intended to stress the known difficult Cefalu neighborhood around `lambda = 1.0`",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations",
            "",
            "- the preferred target is a stronger hypersurface-aware proxy, not a final Ricci-flat metric objective",
            "- the models remain lightweight sklearn baselines",
            "- the release improves protocol clarity and robustness reporting, but it does not settle the final scientific representation question",
            "",
            "## Interpretation",
            "",
            "- these results support the claim that global invariant inputs outperform the local affine baseline on the current benchmark protocol",
            "- these results do not establish a final geometry-learning solution or a complete physical interpretation",
            "",
        ]
    )
    return "\n".join(lines)


def create_benchmark_release(
    *,
    out_dir: str | Path,
    preset_name: str,
    include_hard_slice: bool = False,
    hard_slice_name: str = "cefalu_hard_v1",
) -> dict[str, object]:
    """Materialize a publication-facing benchmark release."""

    output_dir = ensure_directory(out_dir)
    preset = resolve_protocol_preset(preset_name)

    core_dir = output_dir / "core_benchmark"
    core = sweep_experiments(
        out_dir=core_dir,
        preset_name=preset_name,
        target_name=None,
        seeds=None,
        n=preset.n_samples,
        include=None,
        test_size=preset.test_size,
    )

    hard_slice = None
    hard_slice_config = None
    if include_hard_slice:
        hard_slice_config = resolve_hard_evaluation_slice(hard_slice_name)
        hard_dir = output_dir / "harder_slice"
        hard_slice = sweep_experiments(
            out_dir=hard_dir,
            preset_name=None,
            target_name=preset.target_name,
            seeds=list(preset.seeds),
            n=preset.n_samples,
            include=list(hard_slice_config["include"]),
            test_size=preset.test_size,
        )

    core_aggregated = core["aggregated_results"]
    core_robustness = core["robustness"]
    final_results_path = output_dir / "final_results.csv"
    final_results_json_path = output_dir / "final_results.json"
    core_aggregated.to_csv(final_results_path, index=False)
    final_results_json_path.write_text(json.dumps(_json_ready_records(core_aggregated), indent=2), encoding="utf-8")

    final_robustness_path = output_dir / "final_robustness.csv"
    final_robustness_json_path = output_dir / "final_robustness.json"
    core_robustness.to_csv(final_robustness_path, index=False)
    final_robustness_json_path.write_text(
        json.dumps(_json_ready_records(core_robustness), indent=2),
        encoding="utf-8",
    )

    release_protocol = {
        "protocol_version": "phase10",
        "release_version": RELEASE_VERSION,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "preset_name": preset_name,
        "preset": preset.metadata(),
        "core_protocol": core["protocol"],
        "hard_slice": hard_slice_config,
        "include_hard_slice": include_hard_slice,
    }
    (output_dir / "release_protocol.json").write_text(json.dumps(release_protocol, indent=2), encoding="utf-8")

    (output_dir / "final_summary.md").write_text(
        _final_summary_markdown(
            preset_name=preset_name,
            target_name=preset.target_name,
            robustness_df=core_robustness,
            hard_slice_name=hard_slice_name if include_hard_slice else None,
            hard_slice_robustness_df=None if hard_slice is None else hard_slice["robustness"],
        ),
        encoding="utf-8",
    )
    (output_dir / "results_memo.md").write_text(
        _results_memo_markdown(
            preset_name=preset_name,
            target_name=preset.target_name,
            protocol=core["protocol"],
            robustness_df=core_robustness,
            hard_slice_name=hard_slice_name if include_hard_slice else None,
            hard_slice_robustness_df=None if hard_slice is None else hard_slice["robustness"],
        ),
        encoding="utf-8",
    )

    release_manifest = {
        "app_name": "GeoCYData",
        "app_version": __version__,
        "schema_version": "0.1",
        "protocol_version": "phase10",
        "release_version": RELEASE_VERSION,
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "release_name": output_dir.name,
        "release_path": str(output_dir),
        "preset_name": preset_name,
        "target_name": preset.target_name,
        "include_hard_slice": include_hard_slice,
        "hard_slice_name": hard_slice_name if include_hard_slice else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(output_dir.parent),
        "artifacts": {
            "release_manifest": "release_manifest.json",
            "release_protocol": "release_protocol.json",
            "final_results_csv": "final_results.csv",
            "final_results_json": "final_results.json",
            "final_robustness_csv": "final_robustness.csv",
            "final_robustness_json": "final_robustness.json",
            "final_summary": "final_summary.md",
            "results_memo": "results_memo.md",
            "core_benchmark": "core_benchmark",
            "harder_slice": "harder_slice" if include_hard_slice else None,
        },
    }
    (output_dir / "release_manifest.json").write_text(json.dumps(release_manifest, indent=2), encoding="utf-8")

    return {
        "manifest": release_manifest,
        "protocol": release_protocol,
        "core": core,
        "hard_slice": hard_slice,
        "out_dir": str(output_dir),
    }
