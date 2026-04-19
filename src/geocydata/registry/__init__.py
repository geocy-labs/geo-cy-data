"""Geometry and case registries."""

from geocydata.registry.cases import (
    GEOMETRY_CASES,
    NEAR_SINGULAR_SLICES,
    PAPER1_CORE_CASE_IDS,
    PAPER2_HARD_REGIME_CASE_IDS,
    build_benchmark_case_entry,
    canonicalize_cefalu_lambda_case_id,
    derive_case_id,
    get_case,
    list_case_ids,
    model_facing_views_for_case,
)

__all__ = [
    "GEOMETRY_CASES",
    "NEAR_SINGULAR_SLICES",
    "PAPER1_CORE_CASE_IDS",
    "PAPER2_HARD_REGIME_CASE_IDS",
    "build_benchmark_case_entry",
    "canonicalize_cefalu_lambda_case_id",
    "derive_case_id",
    "get_case",
    "list_case_ids",
    "model_facing_views_for_case",
]

