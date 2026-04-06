import json

import numpy as np

from geocydata.geometry.cefalu import CefaluQuarticGeometry
from geocydata.geometry.hypersurfaces import cefalu_quartic_polynomial
from geocydata.symmetry.actions import apply_action, generate_orbit
from geocydata.symmetry.canonicalize import (
    canonical_key,
    canonical_key_string,
    choose_canonical_representative,
)
from geocydata.symmetry.groups import cefalu_symmetry_actions
from geocydata.utils.seeds import make_rng
from geocydata.validation.symmetry_checks import build_orbits_dataframe, build_symmetry_report


def test_cefalu_symmetry_action_preserves_polynomial() -> None:
    geometry = CefaluQuarticGeometry()
    parameters = {"lambda": 1.0}
    point = geometry.sample_points(n=1, rng=make_rng(7), parameters=parameters)[0]
    action = cefalu_symmetry_actions()[37]
    transformed = apply_action(point, action)
    base_value = cefalu_quartic_polynomial(point[np.newaxis, :], 1.0)[0]
    transformed_value = cefalu_quartic_polynomial(transformed[np.newaxis, :], 1.0)[0]
    assert abs(base_value - transformed_value) < 1e-10


def test_orbit_generation_returns_multiple_valid_representatives() -> None:
    geometry = CefaluQuarticGeometry()
    point = geometry.sample_points(n=1, rng=make_rng(11), parameters={"lambda": 0.75})[0]
    orbit = generate_orbit(point, cefalu_symmetry_actions())
    unique_keys = {canonical_key(image) for _, image in orbit}
    assert len(unique_keys) > 4


def test_canonical_representative_selection_is_deterministic() -> None:
    geometry = CefaluQuarticGeometry()
    point = geometry.sample_points(n=1, rng=make_rng(19), parameters={"lambda": 1.0})[0]
    actions = cefalu_symmetry_actions()
    _, canonical_point = choose_canonical_representative(point, actions)
    transformed = apply_action(point, actions[121])
    _, transformed_canonical = choose_canonical_representative(transformed, actions)
    assert canonical_key(canonical_point) == canonical_key(transformed_canonical)
    assert canonical_key_string(canonical_point) == canonical_key_string(transformed_canonical)


def test_orbit_dataframe_and_report_integration() -> None:
    geometry = CefaluQuarticGeometry()
    points = geometry.sample_points(n=2, rng=make_rng(5), parameters={"lambda": 1.0})
    orbits_df = build_orbits_dataframe(points, lambda_value=1.0)
    assert {"point_id", "action_id", "canonical_key", "is_canonical", "orbit_size"}.issubset(orbits_df.columns)
    report = build_symmetry_report(points, lambda_value=1.0)
    assert report["passed"] is True
    json.dumps(report)
