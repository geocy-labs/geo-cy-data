"""Geometry and case registries."""

from geocydata.registry.cases import (
    GEOMETRY_CASES,
    NEAR_SINGULAR_SLICES,
    PAPER1_CORE_CASE_IDS,
    canonicalize_cefalu_lambda_case_id,
    derive_case_id,
    get_case,
    list_case_ids,
)

__all__ = [
    "GEOMETRY_CASES",
    "NEAR_SINGULAR_SLICES",
    "PAPER1_CORE_CASE_IDS",
    "canonicalize_cefalu_lambda_case_id",
    "derive_case_id",
    "get_case",
    "list_case_ids",
]

