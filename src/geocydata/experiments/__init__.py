"""Experiment runner package for GeoCYData."""

from geocydata.experiments.data import TARGET_METADATA
from geocydata.experiments.paper_assets import build_paper_assets
from geocydata.experiments.protocols import (
    HARD_EVALUATION_SLICES,
    PROTOCOL_PRESETS,
    list_hard_evaluation_slices,
    list_protocol_presets,
    resolve_hard_evaluation_slice,
    resolve_protocol_preset,
)
from geocydata.experiments.release import create_benchmark_release
from geocydata.experiments.runner import compare_experiments, run_experiment
from geocydata.experiments.sweep import BENCHMARK_CASES, sweep_experiments
from geocydata.experiments.validate_paper_assets import validate_paper_assets
from geocydata.experiments.validate_release import validate_benchmark_release

__all__ = [
    "BENCHMARK_CASES",
    "HARD_EVALUATION_SLICES",
    "PROTOCOL_PRESETS",
    "TARGET_METADATA",
    "build_paper_assets",
    "compare_experiments",
    "create_benchmark_release",
    "list_hard_evaluation_slices",
    "list_protocol_presets",
    "resolve_hard_evaluation_slice",
    "run_experiment",
    "resolve_protocol_preset",
    "sweep_experiments",
    "validate_paper_assets",
    "validate_benchmark_release",
]

