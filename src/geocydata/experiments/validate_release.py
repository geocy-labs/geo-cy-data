"""Release validation helpers for GeoCYData benchmark releases."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geocydata.experiments.release import BENCHMARK_CONTRACT_VERSION, RELEASE_VERSION


REQUIRED_RELEASE_FILES = (
    "release_manifest.json",
    "release_protocol.json",
    "final_results.csv",
    "final_results.json",
    "final_robustness.csv",
    "final_robustness.json",
    "final_summary.md",
    "results_memo.md",
)

REQUIRED_RELEASE_DIRS = ("core_benchmark",)
REQUIRED_FINAL_RESULTS_COLUMNS = (
    "benchmark_case",
    "case_label",
    "model",
    "validation_score_mean",
    "validation_score_std",
    "validation_mse_mean",
)
REQUIRED_FINAL_ROBUSTNESS_COLUMNS = (
    "benchmark_case",
    "score_gap_mean",
    "score_gap_std",
    "global_beats_local_mean",
    "global_beats_local_all_seeds",
)


def _release_validation_summary(report: dict[str, object]) -> str:
    failures = report["failures"]
    warnings = report["warnings"] or ["none"]
    warning_text = "\n".join(f"- {warning}" for warning in warnings)
    failure_text = "\n".join(f"- {failure}" for failure in failures) or "- none"
    return f"""# GeoCYData Release Validation Summary

## Status

- passed: `{report["passed"]}`
- release version: `{report.get("release_version")}`
- benchmark contract version: `{report.get("benchmark_contract_version")}`

## Failures

{failure_text}

## Warnings

{warning_text}
"""


def validate_benchmark_release(input_dir: str | Path) -> dict[str, object]:
    """Validate a publication-facing GeoCYData release directory."""

    release_dir = Path(input_dir)
    failures: list[str] = []
    warnings: list[str] = []

    for file_name in REQUIRED_RELEASE_FILES:
        if not (release_dir / file_name).exists():
            failures.append(f"Missing required release file: {file_name}")
    for dir_name in REQUIRED_RELEASE_DIRS:
        if not (release_dir / dir_name).is_dir():
            failures.append(f"Missing required release directory: {dir_name}")

    manifest: dict[str, object] = {}
    protocol: dict[str, object] = {}
    final_results = pd.DataFrame()
    final_robustness = pd.DataFrame()
    if not failures:
        manifest = json.loads((release_dir / "release_manifest.json").read_text(encoding="utf-8"))
        protocol = json.loads((release_dir / "release_protocol.json").read_text(encoding="utf-8"))
        final_results = pd.read_csv(release_dir / "final_results.csv")
        final_robustness = pd.read_csv(release_dir / "final_robustness.csv")

    if manifest and protocol:
        for key in ("preset_name", "include_hard_slice"):
            if manifest.get(key) != protocol.get(key):
                failures.append(f"Manifest/protocol mismatch for field: {key}")
        protocol_hard_slice_name = None
        if isinstance(protocol.get("hard_slice"), dict):
            protocol_hard_slice_name = protocol["hard_slice"].get("name")
        if manifest.get("hard_slice_name") != protocol_hard_slice_name:
            failures.append("Manifest/protocol mismatch for field: hard_slice_name")
        if manifest.get("target_name") != protocol.get("preset", {}).get("target_name"):
            failures.append("Manifest target_name does not match release protocol preset target_name.")
        if manifest.get("release_version") != RELEASE_VERSION or protocol.get("release_version") != RELEASE_VERSION:
            failures.append("Release version metadata is missing or inconsistent.")
        if (
            manifest.get("benchmark_contract_version") != BENCHMARK_CONTRACT_VERSION
            or protocol.get("benchmark_contract_version") != BENCHMARK_CONTRACT_VERSION
        ):
            failures.append("Benchmark contract version metadata is missing or inconsistent.")
        if protocol.get("core_protocol") is None:
            failures.append("Release protocol is missing core_protocol metadata.")

    if not final_results.empty:
        missing = [column for column in REQUIRED_FINAL_RESULTS_COLUMNS if column not in final_results.columns]
        if missing:
            failures.append(f"final_results.csv is missing required columns: {', '.join(missing)}")
    if not final_robustness.empty:
        missing = [column for column in REQUIRED_FINAL_ROBUSTNESS_COLUMNS if column not in final_robustness.columns]
        if missing:
            failures.append(f"final_robustness.csv is missing required columns: {', '.join(missing)}")

    if manifest.get("include_hard_slice"):
        harder_slice_dir = release_dir / "harder_slice"
        if not harder_slice_dir.is_dir():
            failures.append("Release manifest declares a harder slice, but `harder_slice/` is missing.")
        else:
            for required in ("benchmark_manifest.json", "benchmark_robustness.csv"):
                if not (harder_slice_dir / required).exists():
                    failures.append(f"Harder slice is missing required file: {required}")

    report = {
        "release_path": str(release_dir),
        "release_version": manifest.get("release_version"),
        "benchmark_contract_version": manifest.get("benchmark_contract_version"),
        "preset_name": manifest.get("preset_name"),
        "target_name": manifest.get("target_name"),
        "include_hard_slice": manifest.get("include_hard_slice"),
        "warnings": warnings,
        "failures": failures,
        "passed": not failures,
    }

    (release_dir / "release_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (release_dir / "release_validation_summary.md").write_text(
        _release_validation_summary(report),
        encoding="utf-8",
    )
    return report
