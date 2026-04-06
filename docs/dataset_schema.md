# Dataset Schema

## `points.parquet`

One row per sampled point with:

- `point_id`
- `chart_id`
- `z{0..3}_re`, `z{0..3}_im`: normalized homogeneous coordinates
- `affine_{0..2}_re`, `affine_{0..2}_im`: affine coordinates in the chosen chart
- `family_lambda` when the geometry is `cefalu_quartic`

`affine_k` stores the coordinates after dividing by the selected pivot coordinate and omitting that pivot entry.

## `invariants.parquet`

One row per sampled point with:

- `point_id`
- `m_{i}_{j}_re`
- `m_{i}_{j}_im`
- `family_lambda` when the geometry is `cefalu_quartic`

These values come from the normalized Hermitian outer product

`M_ij = z_i conj(z_j) / ||z||^2`

which is invariant under nonzero complex projective rescaling.

## `manifest.json`

Includes application metadata, generation parameters, artifact paths, schema version, and bundle naming fields such as `bundle_name` and `bundle_path`. For the Cefalu family, `parameters.lambda` stores the selected family value.

## `orbits.parquet`

Produced by `geocydata generate orbits` for `cefalu_quartic`.

- `point_id`: base sampled point id
- `action_id`: symmetry action label
- `permutation`: JSON-encoded coordinate permutation
- `signs`: JSON-encoded sign pattern
- `canonical_key`: deterministic orbit representative label
- `is_canonical`: whether this row is the chosen canonical representative
- `orbit_size`: number of unique representatives recorded for the base point
- `z{0..3}_re`, `z{0..3}_im`: phase-normalized representative coordinates
- `family_lambda`: Cefalu family parameter

## `validation_report.json`

Includes:

- point count
- residual statistics
- invariant drift statistics
- chart distribution
- warnings
- pass/fail summary

## `summary.md`

Human-readable overview of the bundle contents and validation results.

## `symmetry_report.json`

Produced by `geocydata generate orbits` and `geocydata validate symmetry`.

Includes:

- group size
- orbit size statistics
- residual preservation statistics under symmetry actions
- canonicalization drift
- canonical invariant drift
- warnings and pass/fail summary

## Experiment run artifacts

Produced by `geocydata experiments run`.

- `config.json`: run configuration including bundle path, model mode, seed, and split fraction
- `metrics.json`: train/validation scores, error metrics, run time, feature dimension, geometry metadata, target name, and split sizes
- `predictions.parquet`: point ids, split labels, ground-truth target values, and predictions
- `summary.md`: short markdown summary of the run

Preferred target in Phase 5:

- `fs_scalar`: an ambient Fubini-Study determinant proxy computed from affine chart coordinates

Compatibility target:

- `invariant_weighted_sum`: the older Phase 4 convenience/debug target derived directly from invariant-table features

## Experiment comparison artifacts

Produced by `geocydata experiments compare`.

- `local/` and `global/`: per-model run artifacts
- `comparison.json`: side-by-side summary of local and global metrics, including metric deltas and the target used
- `comparison.md`: short markdown comparison report

## Naming notes

GeoCYData does not force a bundle directory name. The user supplies the output path with `--out`, and that directory name is recorded in the manifest.
