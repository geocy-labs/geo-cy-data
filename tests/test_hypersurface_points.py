from geocydata.geometry.cefalu import CefaluQuarticGeometry
from geocydata.geometry.fermat import FermatQuarticGeometry
from geocydata.utils.seeds import make_rng


def test_fermat_quartic_points_satisfy_equation_within_tolerance() -> None:
    geometry = FermatQuarticGeometry()
    points = geometry.sample_points(n=32, rng=make_rng(7))
    residuals = geometry.residuals(points)
    assert residuals.max() < 1e-10


def test_cefalu_quartic_points_satisfy_equation_within_tolerance() -> None:
    geometry = CefaluQuarticGeometry()
    parameters = {"lambda": 1.0}
    points = geometry.sample_points(n=32, rng=make_rng(7), parameters=parameters)
    residuals = geometry.residuals(points, parameters=parameters)
    assert residuals.max() < 1e-10

