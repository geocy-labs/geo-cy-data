"""Named geometry cases and slices for GeoCYData benchmark workflows."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class GeometryCase:
    """A named geometry case with fixed parameters."""

    case_id: str
    geometry: str
    parameters: dict[str, object]
    label: str

    def metadata(self) -> dict[str, object]:
        """Return a JSON-serializable case definition."""

        return {
            "case_id": self.case_id,
            "geometry": self.geometry,
            "parameters": self.parameters,
            "label": self.label,
        }


GEOMETRY_CASES: tuple[GeometryCase, ...] = (
    GeometryCase("fermat_quartic", "fermat_quartic", {}, "Fermat quartic"),
    GeometryCase("cefalu_lambda_0_0", "cefalu_quartic", {"lambda": 0.0}, "Cefalu quartic (lambda=0.0)"),
    GeometryCase("cefalu_lambda_0_50", "cefalu_quartic", {"lambda": 0.5}, "Cefalu quartic (lambda=0.50)"),
    GeometryCase("cefalu_lambda_0_74", "cefalu_quartic", {"lambda": 0.74}, "Cefalu quartic (lambda=0.74)"),
    GeometryCase("cefalu_lambda_0_75", "cefalu_quartic", {"lambda": 0.75}, "Cefalu quartic (lambda=0.75)"),
    GeometryCase("cefalu_lambda_0_76", "cefalu_quartic", {"lambda": 0.76}, "Cefalu quartic (lambda=0.76)"),
    GeometryCase("cefalu_lambda_0_90", "cefalu_quartic", {"lambda": 0.9}, "Cefalu quartic (lambda=0.90)"),
    GeometryCase("cefalu_lambda_0_99", "cefalu_quartic", {"lambda": 0.99}, "Cefalu quartic (lambda=0.99)"),
    GeometryCase("cefalu_lambda_1_0", "cefalu_quartic", {"lambda": 1.0}, "Cefalu quartic (lambda=1.0)"),
    GeometryCase("cefalu_lambda_1_01", "cefalu_quartic", {"lambda": 1.01}, "Cefalu quartic (lambda=1.01)"),
    GeometryCase("cefalu_lambda_1_10", "cefalu_quartic", {"lambda": 1.1}, "Cefalu quartic (lambda=1.10)"),
    GeometryCase("cefalu_lambda_1_5", "cefalu_quartic", {"lambda": 1.5}, "Cefalu quartic (lambda=1.5)"),
    GeometryCase("cefalu_lambda_3_0", "cefalu_quartic", {"lambda": 3.0}, "Cefalu quartic (lambda=3.0)"),
)

CASE_BY_ID = {case.case_id: case for case in GEOMETRY_CASES}
CASE_ID_ALIASES: dict[str, str] = {
    "cefalu_lambda_0_5": "cefalu_lambda_0_50",
    "cefalu_lambda_0_9": "cefalu_lambda_0_90",
    "cefalu_lambda_1_00": "cefalu_lambda_1_0",
    "cefalu_lambda_1_1": "cefalu_lambda_1_10",
}

PAPER1_CORE_CASE_IDS: tuple[str, ...] = (
    "fermat_quartic",
    "cefalu_lambda_0_0",
    "cefalu_lambda_0_75",
    "cefalu_lambda_1_0",
    "cefalu_lambda_1_5",
    "cefalu_lambda_3_0",
)
PAPER2_HARD_REGIME_CASE_IDS: tuple[str, ...] = (
    "cefalu_lambda_0_50",
    "cefalu_lambda_0_75",
    "cefalu_lambda_0_90",
    "cefalu_lambda_1_0",
    "cefalu_lambda_1_10",
)
NEAR_SINGULAR_SLICES: dict[str, tuple[str, ...]] = {
    "cefalu_near_lambda_0_75": ("cefalu_lambda_0_74", "cefalu_lambda_0_75", "cefalu_lambda_0_76"),
    "cefalu_near_lambda_1_0": ("cefalu_lambda_0_99", "cefalu_lambda_1_0", "cefalu_lambda_1_01"),
}


def list_case_ids() -> list[str]:
    """Return the registered case ids."""

    return sorted(CASE_BY_ID)


def get_case(case_id: str) -> GeometryCase:
    """Resolve one named geometry case."""

    try:
        return CASE_BY_ID[CASE_ID_ALIASES.get(case_id, case_id)]
    except KeyError as exc:
        available = ", ".join(list_case_ids())
        raise ValueError(f"Unknown case_id '{case_id}'. Available cases: {available}.") from exc


def canonicalize_cefalu_lambda_case_id(lambda_value: float) -> str:
    """Return the canonical case id for one Cefalu family parameter."""

    normalized = format(Decimal(str(lambda_value)).normalize(), "f")
    if "." not in normalized:
        normalized = f"{normalized}.0"
    normalized = normalized.replace("-", "neg_").replace(".", "_")
    return f"cefalu_lambda_{normalized}"


def derive_case_id(geometry: str, parameters: dict[str, object] | None = None) -> str:
    """Return a stable case id for a geometry/parameter choice."""

    resolved_parameters = parameters or {}
    if geometry == "cefalu_quartic" and "lambda" in resolved_parameters:
        canonical_case_id = canonicalize_cefalu_lambda_case_id(float(resolved_parameters["lambda"]))
        if canonical_case_id in CASE_BY_ID:
            return canonical_case_id
    for case in GEOMETRY_CASES:
        if case.geometry != geometry:
            continue
        if case.parameters == resolved_parameters:
            return case.case_id
    if geometry == "cefalu_quartic" and "lambda" in resolved_parameters:
        return canonicalize_cefalu_lambda_case_id(float(resolved_parameters["lambda"]))
    return geometry


def model_facing_views_for_case(case: GeometryCase) -> list[str]:
    """Return the model-facing bundle views expected for a benchmark case."""

    views = [
        "local_chart_representation",
        "invariant_representation",
        "sampling_metadata",
    ]
    if case.geometry == "cefalu_quartic":
        views.extend(
            [
                "canonical_representatives",
                "canonical_invariants",
                "symmetry_orbit_metadata",
            ]
        )
    return views


def build_benchmark_case_entry(case: GeometryCase, *, benchmark_version: str) -> dict[str, object]:
    """Return richer machine-readable metadata for one benchmark case."""

    return {
        **case.metadata(),
        "lambda_value": case.parameters.get("lambda"),
        "geometry_family": case.geometry,
        "benchmark_version": benchmark_version,
        "available_model_facing_views": model_facing_views_for_case(case),
    }
