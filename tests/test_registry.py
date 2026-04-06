from geocydata.registry.geometries import get_geometry, list_geometries


def test_cefalu_quartic_is_registered() -> None:
    assert "cefalu_quartic" in list_geometries()
    geometry = get_geometry("cefalu_quartic")
    metadata = geometry.metadata()
    assert metadata["name"] == "cefalu_quartic"
    assert "lambda" in metadata["parameter_schema"]
