# Architecture

GeoCYData uses a small `src/` layout with focused modules:

- `geocydata.cli`: Typer-based command-line interface
- `geocydata.registry`: benchmark geometry registry
- `geocydata.geometry`: projective utilities, hypersurface evaluation, Fermat quartic model, and affine chart logic
- `geocydata.sampling`: point generation for benchmark geometries
- `geocydata.features`: invariant feature construction from normalized outer products
- `geocydata.validation`: residual, invariance, and report aggregation logic
- `geocydata.export`: Parquet writing and manifest generation
- `geocydata.utils`: logging, seeds, paths, and version helpers

The current benchmark path is intentionally narrow: sample points on the Fermat quartic, compute affine and invariant representations, validate the resulting dataset, and export a documented bundle.

