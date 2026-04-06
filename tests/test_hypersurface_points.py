from geocydata.geometry.fermat import FermatQuarticGeometry
from geocydata.utils.seeds import make_rng


def test_fermat_quartic_points_satisfy_equation_within_tolerance() -> None:
    geometry = FermatQuarticGeometry()
    points = geometry.sample_points(n=32, rng=make_rng(7))
    residuals = geometry.residuals(points)
    assert residuals.max() < 1e-10

