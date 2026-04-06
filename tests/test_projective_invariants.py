import numpy as np

from geocydata.features.invariants import invariant_matrix
from geocydata.geometry.cefalu import CefaluQuarticGeometry
from geocydata.geometry.fermat import FermatQuarticGeometry
from geocydata.geometry.projective import complex_rescaling
from geocydata.utils.seeds import make_rng


def test_projective_invariants_are_stable_under_rescaling() -> None:
    geometry = FermatQuarticGeometry()
    rng = make_rng(11)
    point = geometry.sample_points(n=1, rng=rng)[0]
    scale = complex_rescaling(rng, size=1)[0]
    baseline = invariant_matrix(point)
    rescaled = invariant_matrix(point * scale)
    assert np.max(np.abs(baseline - rescaled)) < 1e-12


def test_cefalu_projective_invariants_are_stable_under_rescaling() -> None:
    geometry = CefaluQuarticGeometry()
    rng = make_rng(17)
    parameters = {"lambda": 0.75}
    point = geometry.sample_points(n=1, rng=rng, parameters=parameters)[0]
    scale = complex_rescaling(rng, size=1)[0]
    baseline = invariant_matrix(point)
    rescaled = invariant_matrix(point * scale)
    assert np.max(np.abs(baseline - rescaled)) < 1e-12

