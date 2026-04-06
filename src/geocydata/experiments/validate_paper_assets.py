"""Consistency validation for paper-facing assets built from a frozen release."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geocydata.experiments.paper_assets import CORE_RESULTS_COLUMNS, ROBUSTNESS_COLUMNS


REQUIRED_PAPER_FILES = (
    "outline.md",
    "abstract_draft.md",
    "introduction_notes.md",
    "methods_notes.md",
    "results_notes.md",
    "reproducibility_notes.md",
    "assets/core_results_table.csv",
    "assets/core_results_table.md",
    "assets/robustness_table.csv",
    "assets/robustness_table.md",
    "assets/fig_core_scores.png",
    "assets/fig_hardest_case.png",
)
REQUIRED_RELEASE_FILES = (
    "release_manifest.json",
    "release_protocol.json",
    "final_results.csv",
    "final_robustness.csv",
)


def _paper_validation_summary(report: dict[str, object]) -> str:
    failures = report["failures"]
    warnings = report["warnings"] or ["none"]
    checks = report["checks"]
    failure_text = "\n".join(f"- {failure}" for failure in failures) or "- none"
    warning_text = "\n".join(f"- {warning}" for warning in warnings)
    structural_text = "\n".join(f"- {name}: `{value}`" for name, value in checks["structural"].items())
    table_text = "\n".join(f"- {name}: `{value}`" for name, value in checks["tables"].items())
    finding_text = "\n".join(f"- {name}: `{value}`" for name, value in checks["findings"].items())
    return f"""# GeoCYData Paper Asset Validation Summary

## Status

- passed: `{report["passed"]}`
- release: `{report["release_path"]}`
- paper: `{report["paper_path"]}`

## Structural checks

{structural_text}

## Table checks

{table_text}

## Finding checks

{finding_text}

## Failures

{failure_text}

## Warnings

{warning_text}
"""


def _load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def _same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    left_norm = left.sort_values(list(left.columns)).reset_index(drop=True)
    right_norm = right.sort_values(list(right.columns)).reset_index(drop=True)
    return left_norm.equals(right_norm)


def validate_paper_assets(*, release_dir: str | Path, paper_dir: str | Path) -> dict[str, object]:
    """Validate that paper-facing assets remain consistent with a frozen release."""

    release_path = Path(release_dir)
    paper_path = Path(paper_dir)
    failures: list[str] = []
    warnings: list[str] = []

    structural_checks = {
        "paper_files_present": True,
        "release_files_present": True,
        "figures_non_empty": True,
    }
    table_checks = {
        "core_results_table_matches_release": True,
        "robustness_table_matches_release": True,
        "paper_cases_cover_release_cases": True,
    }
    finding_checks = {
        "core_mean_win_claim_present": True,
        "hardest_core_case_claim_present": True,
        "hardest_harder_slice_claim_present": True,
    }

    for relative_path in REQUIRED_PAPER_FILES:
        path = paper_path / relative_path
        if not path.exists():
            structural_checks["paper_files_present"] = False
            failures.append(f"Missing required paper file: {relative_path}")
    for relative_path in REQUIRED_RELEASE_FILES:
        path = release_path / relative_path
        if not path.exists():
            structural_checks["release_files_present"] = False
            failures.append(f"Missing required release file: {relative_path}")

    for relative_path in ("assets/fig_core_scores.png", "assets/fig_hardest_case.png"):
        path = paper_path / relative_path
        if path.exists() and path.stat().st_size <= 0:
            structural_checks["figures_non_empty"] = False
            failures.append(f"Figure file is empty: {relative_path}")

    manifest: dict[str, object] = {}
    release_results = pd.DataFrame()
    release_robustness = pd.DataFrame()
    paper_results = pd.DataFrame()
    paper_robustness = pd.DataFrame()

    if not failures:
        manifest = json.loads((release_path / "release_manifest.json").read_text(encoding="utf-8"))
        release_results = _load_csv(release_path / "final_results.csv")
        release_robustness = _load_csv(release_path / "final_robustness.csv")
        paper_results = _load_csv(paper_path / "assets" / "core_results_table.csv")
        paper_robustness = _load_csv(paper_path / "assets" / "robustness_table.csv")

        expected_paper_results = release_results.loc[:, CORE_RESULTS_COLUMNS].copy().fillna("")
        expected_paper_robustness = release_robustness.loc[:, ROBUSTNESS_COLUMNS].copy().fillna("")
        if not _same_frame(paper_results, expected_paper_results):
            table_checks["core_results_table_matches_release"] = False
            failures.append("paper/assets/core_results_table.csv does not match the release-derived core table.")
        if not _same_frame(paper_robustness, expected_paper_robustness):
            table_checks["robustness_table_matches_release"] = False
            failures.append("paper/assets/robustness_table.csv does not match the release-derived robustness table.")

        release_cases = set(str(value) for value in release_results["case_label"].unique().tolist())
        paper_cases = set(str(value) for value in paper_results["case_label"].unique().tolist())
        if release_cases != paper_cases:
            table_checks["paper_cases_cover_release_cases"] = False
            failures.append("Paper core-results table does not cover the same cases as final_results.csv.")

        results_notes = (paper_path / "results_notes.md").read_text(encoding="utf-8")
        expected_core_claim = f"global beats local on mean validation score in `{int(release_robustness['global_beats_local_mean'].sum())}` / `{len(release_robustness)}` core cases"
        if expected_core_claim not in results_notes:
            finding_checks["core_mean_win_claim_present"] = False
            failures.append("results_notes.md is missing the expected core mean-win claim.")

        hardest_core = str(
            release_robustness.loc[release_robustness["absolute_score_gap_mean"].idxmin(), "benchmark_case"]
        )
        if f"Hardest core case by smallest separation: `{hardest_core}`" not in results_notes:
            finding_checks["hardest_core_case_claim_present"] = False
            failures.append("results_notes.md is missing the expected hardest core case claim.")

        harder_slice_robustness_path = release_path / "harder_slice" / "benchmark_robustness.csv"
        if manifest.get("include_hard_slice") and harder_slice_robustness_path.exists():
            harder_slice_robustness = _load_csv(harder_slice_robustness_path)
            hardest_harder = str(
                harder_slice_robustness.loc[
                    harder_slice_robustness["absolute_score_gap_mean"].idxmin(),
                    "benchmark_case",
                ]
            )
            if f"Harder-slice stress case by smallest separation: `{hardest_harder}`" not in results_notes:
                finding_checks["hardest_harder_slice_claim_present"] = False
                failures.append("results_notes.md is missing the expected harder-slice hardest-case claim.")
        else:
            finding_checks["hardest_harder_slice_claim_present"] = False
            warnings.append("No harder slice was present, so the harder-slice finding check was skipped.")

    report = {
        "release_path": str(release_path),
        "paper_path": str(paper_path),
        "checks": {
            "structural": structural_checks,
            "tables": table_checks,
            "findings": finding_checks,
        },
        "warnings": warnings,
        "failures": failures,
        "passed": not failures,
    }

    (paper_path / "paper_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (paper_path / "paper_validation_summary.md").write_text(
        _paper_validation_summary(report),
        encoding="utf-8",
    )
    return report
