"""Registered benchmark geometries."""

from __future__ import annotations

from geocydata.geometry.fermat import FermatQuarticGeometry

GEOMETRIES = {
    "fermat_quartic": FermatQuarticGeometry(),
}


def list_geometries() -> list[str]:
    """Return the registered geometry names."""

    return sorted(GEOMETRIES)


def get_geometry(name: str) -> FermatQuarticGeometry:
    """Resolve a geometry from the registry."""

    try:
        return GEOMETRIES[name]
    except KeyError as exc:
        available = ", ".join(list_geometries())
        raise KeyError(f"Unknown geometry '{name}'. Available: {available}.") from exc

