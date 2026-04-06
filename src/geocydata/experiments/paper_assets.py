"""Paper-facing assets built from frozen GeoCYData release outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

from geocydata.utils.paths import ensure_directory

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CORE_RESULTS_COLUMNS = (
    "case_label",
    "geometry",
    "lambda",
    "model",
    "validation_score_mean",
    "validation_score_std",
    "validation_mse_mean",
    "run_count",
)
ROBUSTNESS_COLUMNS = (
    "case_label",
    "geometry",
    "lambda",
    "score_gap_mean",
    "score_gap_std",
    "global_beats_local_mean",
    "global_beats_local_all_seeds",
    "run_count",
)


def _markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.fillna("").itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _format_table_for_markdown(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: f"{value:.3f}")
        elif pd.api.types.is_bool_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "yes" if value else "no")
    if "model" in formatted.columns:
        formatted["model"] = formatted["model"].map(lambda value: str(value).title())
    return formatted


def _ordered_results(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["model"] = pd.Categorical(ordered["model"], categories=["local", "global"], ordered=True)
    ordered = ordered.sort_values(["case_label", "model"]).reset_index(drop=True)
    ordered["model"] = ordered["model"].astype(str)
    return ordered


def _plot_score_bars(frame: pd.DataFrame, output_path: Path, *, title: str) -> None:
    ordered = _ordered_results(frame)
    case_labels = list(dict.fromkeys(ordered["case_label"].tolist()))
    local = ordered[ordered["model"] == "local"].set_index("case_label")
    global_ = ordered[ordered["model"] == "global"].set_index("case_label")

    x = list(range(len(case_labels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(
        [value - width / 2 for value in x],
        [float(local.loc[label, "validation_score_mean"]) for label in case_labels],
        width,
        yerr=[float(local.loc[label, "validation_score_std"]) for label in case_labels],
        capsize=4,
        label="Local",
        color="#c96b3b",
    )
    ax.bar(
        [value + width / 2 for value in x],
        [float(global_.loc[label, "validation_score_mean"]) for label in case_labels],
        width,
        yerr=[float(global_.loc[label, "validation_score_std"]) for label in case_labels],
        capsize=4,
        label="Global",
        color="#2c6e91",
    )
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=15, ha="right")
    ax.set_ylabel("Validation score (mean +/- std)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _paper_outline() -> str:
    return """# Paper Outline

1. Abstract
2. Introduction
3. Benchmark definition
4. Geometry cases and targets
5. Models and protocol presets
6. Core results
7. Robustness and harder-slice behavior
8. Reproducibility and release validation
9. Limitations and next steps
"""


def _abstract_draft(
    *,
    preset_name: str,
    target_name: str,
    hardest_case: str,
    robustness_df: pd.DataFrame,
) -> str:
    wins = int(robustness_df["global_beats_local_mean"].sum())
    return f"""# Abstract Draft

GeoCYData provides a reproducible benchmark pipeline for projective hypersurface Calabi-Yau data generation and lightweight representation comparison. In the current `{preset_name}` benchmark protocol, we evaluate local affine and global invariant baselines on the preferred `{target_name}` target across Fermat and Cefalu quartic cases. The current benchmark indicates that the global model outperforms the local baseline on mean validation score in `{wins}` of `{len(robustness_df)}` core benchmark cases, while the hardest regime remains `{hardest_case}` near the Cefalu `lambda = 1` neighborhood. These results support a meaningful but non-absolute advantage for global invariant inputs under the present benchmark contract.
"""


def _introduction_notes() -> str:
    return """# Introduction Notes

- Position GeoCYData as benchmark infrastructure for controlled geometry-learning experiments rather than as a final metric-learning system.
- Motivate the need for public, versioned, regenerable benchmark releases for Calabi-Yau-style data workflows.
- Frame the local-versus-global comparison as a representation benchmark with clear claim boundaries.
"""


def _methods_notes(protocol: dict[str, object]) -> str:
    resolved = protocol["core_protocol"]["resolved"]
    return f"""# Methods Notes

- Frozen protocol preset: `{protocol["preset_name"]}`
- Target: `{protocol["preset"]["target_name"]}`
- Seeds: `{', '.join(str(seed) for seed in resolved["seeds"])}`
- Cases: `{', '.join(resolved["include"])}`
- Samples per case: `{resolved["n_samples"]}`
- Split strategy: `{resolved["split_strategy"]}`
- Models: local affine baseline and global invariant baseline, both using the existing lightweight sklearn runner
- Source of truth: release outputs under `final_results.csv`, `final_robustness.csv`, and the preserved benchmark subdirectories
"""


def _results_notes(
    *,
    target_name: str,
    robustness_df: pd.DataFrame,
    hard_slice_df: pd.DataFrame | None,
) -> str:
    hardest_case = robustness_df.loc[robustness_df["absolute_score_gap_mean"].idxmin()]
    lines = [
        "# Results Notes",
        "",
        f"- Preferred target: `{target_name}`",
        f"- Core finding: global beats local on mean validation score in `{int(robustness_df['global_beats_local_mean'].sum())}` / `{len(robustness_df)}` core cases",
        f"- Hardest core case by smallest separation: `{hardest_case['benchmark_case']}`",
        f"- Hardest core case by variance: `{robustness_df.loc[robustness_df['max_validation_score_std'].idxmax(), 'benchmark_case']}`",
        f"- Global beats local on all seeds in hardest core case: `{bool(hardest_case['global_beats_local_all_seeds'])}`",
    ]
    if hard_slice_df is not None and not hard_slice_df.empty:
        hardest_slice = hard_slice_df.loc[hard_slice_df["absolute_score_gap_mean"].idxmin()]
        lines.extend(
            [
                f"- Harder-slice stress case by smallest separation: `{hardest_slice['benchmark_case']}`",
                f"- Harder-slice global mean-win count: `{int(hard_slice_df['global_beats_local_mean'].sum())}` / `{len(hard_slice_df)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundaries",
            "",
            "- The benchmark supports a reproducible advantage for the global representation on mean score across the current core benchmark.",
            "- The advantage is meaningful but not absolute, especially near the Cefalu `lambda = 1` neighborhood.",
            "- These results do not establish a final geometry-learning or physics-learning solution.",
        ]
    )
    return "\n".join(lines) + "\n"


def _reproducibility_notes(
    *,
    release_dir: Path,
    manifest: dict[str, object],
    protocol: dict[str, object],
) -> str:
    return f"""# Reproducibility Notes

- Release path: `{release_dir}`
- Release version: `{manifest["release_version"]}`
- Benchmark contract version: `{manifest["benchmark_contract_version"]}`
- Preset: `{manifest["preset_name"]}`
- Preferred target: `{manifest["target_name"]}`

## Regeneration

```bash
geocydata experiments regenerate-release --preset {manifest["preset_name"]} --out releases/{manifest["release_name"]}_regenerated{" --include-hard-slice" if manifest["include_hard_slice"] else ""}
```

## Validation

```bash
geocydata experiments validate-release --input {release_dir}
```

## Paper assets

```bash
geocydata experiments build-paper-assets --input {release_dir} --out paper
```

- The release outputs remain the source of truth for the paper tables, figures, and manuscript notes.
- Validation should succeed before paper-facing assets are circulated externally.
"""


def _load_release_inputs(input_dir: Path) -> tuple[dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    manifest = json.loads((input_dir / "release_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((input_dir / "release_protocol.json").read_text(encoding="utf-8"))
    final_results = pd.read_csv(input_dir / "final_results.csv")
    final_robustness = pd.read_csv(input_dir / "final_robustness.csv")
    harder_slice_results = None
    harder_slice_aggregated = input_dir / "harder_slice" / "benchmark_aggregated.csv"
    if harder_slice_aggregated.exists():
        harder_slice_results = pd.read_csv(harder_slice_aggregated)
    return manifest, protocol, final_results, final_robustness, harder_slice_results


def build_paper_assets(*, input_dir: str | Path, out_dir: str | Path = "paper") -> dict[str, object]:
    """Build manuscript-facing tables, figures, and notes from a frozen release."""

    release_dir = Path(input_dir)
    output_dir = ensure_directory(out_dir)
    assets_dir = ensure_directory(output_dir / "assets")

    manifest, protocol, final_results, final_robustness, harder_slice_results = _load_release_inputs(release_dir)
    ordered_results = _ordered_results(final_results)

    core_results_table = ordered_results.loc[:, CORE_RESULTS_COLUMNS].copy()
    robustness_table = final_robustness.loc[:, ROBUSTNESS_COLUMNS].copy().sort_values("case_label").reset_index(drop=True)

    core_results_csv = assets_dir / "core_results_table.csv"
    core_results_md = assets_dir / "core_results_table.md"
    robustness_csv = assets_dir / "robustness_table.csv"
    robustness_md = assets_dir / "robustness_table.md"

    core_results_table.to_csv(core_results_csv, index=False)
    core_results_md.write_text(_markdown_table(_format_table_for_markdown(core_results_table)), encoding="utf-8")
    robustness_table.to_csv(robustness_csv, index=False)
    robustness_md.write_text(_markdown_table(_format_table_for_markdown(robustness_table)), encoding="utf-8")

    core_fig = assets_dir / "fig_core_scores.png"
    _plot_score_bars(ordered_results, core_fig, title="Core benchmark mean validation score")

    hardest_fig = assets_dir / "fig_hardest_case.png"
    if harder_slice_results is not None and not harder_slice_results.empty:
        _plot_score_bars(
            harder_slice_results,
            hardest_fig,
            title="Harder Cefalu slice near lambda=1",
        )
    else:
        hardest_case = final_robustness.loc[final_robustness["absolute_score_gap_mean"].idxmin(), "benchmark_case"]
        _plot_score_bars(
            ordered_results[ordered_results["benchmark_case"] == hardest_case],
            hardest_fig,
            title=f"Hardest core case: {hardest_case}",
        )

    hardest_core_case = final_robustness.loc[final_robustness["absolute_score_gap_mean"].idxmin(), "benchmark_case"]
    harder_slice_robustness = None
    harder_slice_robustness_path = release_dir / "harder_slice" / "benchmark_robustness.csv"
    if harder_slice_robustness_path.exists():
        harder_slice_robustness = pd.read_csv(harder_slice_robustness_path)

    (output_dir / "outline.md").write_text(_paper_outline(), encoding="utf-8")
    (output_dir / "abstract_draft.md").write_text(
        _abstract_draft(
            preset_name=str(manifest["preset_name"]),
            target_name=str(manifest["target_name"]),
            hardest_case=str(hardest_core_case),
            robustness_df=final_robustness,
        ),
        encoding="utf-8",
    )
    (output_dir / "introduction_notes.md").write_text(_introduction_notes(), encoding="utf-8")
    (output_dir / "methods_notes.md").write_text(_methods_notes(protocol), encoding="utf-8")
    (output_dir / "results_notes.md").write_text(
        _results_notes(
            target_name=str(manifest["target_name"]),
            robustness_df=final_robustness,
            hard_slice_df=harder_slice_robustness,
        ),
        encoding="utf-8",
    )
    (output_dir / "reproducibility_notes.md").write_text(
        _reproducibility_notes(release_dir=release_dir, manifest=manifest, protocol=protocol),
        encoding="utf-8",
    )

    return {
        "input_dir": str(release_dir),
        "out_dir": str(output_dir),
        "assets_dir": str(assets_dir),
        "core_results_rows": len(core_results_table),
        "robustness_rows": len(robustness_table),
        "target_name": manifest["target_name"],
    }
