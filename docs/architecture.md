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

The current benchmark path is intentionally narrow: sample points on the Fermat quartic or the parameterized Cefalu quartic family, compute affine and invariant representations, validate the resulting dataset, and export a documented bundle.

Phase 3 adds a small symmetry layer for the Cefalu family:

- `geocydata.symmetry.groups`: default signed permutation actions
- `geocydata.symmetry.actions`: application of actions and orbit generation
- `geocydata.symmetry.canonicalize`: deterministic canonical representatives
- `geocydata.validation.symmetry_checks`: orbit export tables and symmetry consistency reports

Phase 4 adds a lightweight experiment layer:

- `geocydata.experiments.data`: bundle loading plus local/global feature preparation
- `geocydata.experiments.models`: small sklearn baselines used for both views
- `geocydata.experiments.runner`: reproducible train/validation splits, artifact writing, and comparison runs

