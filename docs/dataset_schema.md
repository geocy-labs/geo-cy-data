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

## Naming notes

GeoCYData does not force a bundle directory name. The user supplies the output path with `--out`, and that directory name is recorded in the manifest.
