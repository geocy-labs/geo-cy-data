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

- `config.json`: run configuration including bundle path, benchmark case id, model mode, split strategy, split seed, target metadata, and split fraction
- `metrics.json`: train/validation scores, error metrics, run time, feature dimension, geometry metadata, target status, feature mode, and split sizes
- `predictions.parquet`: point ids, split labels, ground-truth target values, and predictions
- `run_manifest.json`: explicit run protocol manifest for later audit and reuse
- `summary.md`: short markdown summary of the run

Preferred target in Phase 8:

- `hypersurface_fs_scalar`: a tangent-restricted Fubini-Study proxy computed from the affine chart metric and the local hypersurface gradient

Compatibility target:

- `fs_scalar`: the older ambient Fubini-Study proxy from affine chart coordinates
- `invariant_weighted_sum`: the older Phase 4 convenience/debug target derived directly from invariant-table features

## Experiment comparison artifacts

Produced by `geocydata experiments compare`.

- `local/` and `global/`: per-model run artifacts
- `comparison.json`: side-by-side summary of local and global metrics, including metric deltas and the target used
- `comparison.md`: short markdown comparison report

## Benchmark sweep artifacts

Produced by `geocydata experiments sweep`.

- `benchmark_protocol.json`: resolved preset, explicit overrides, target metadata, seeds, case ids, and split strategy
- `benchmark_manifest.json`: sweep target, seed list, benchmark cases, result count, and artifact paths
- `benchmark_results.csv`: one row per `(case, seed, model)` result with geometry, lambda, target, split sizes, validation metrics, runtime, bundle path, and run path
- `benchmark_results.json`: machine-readable JSON version of the tidy result table
- `benchmark_summary.md`: markdown summary with per-case winners and score/MSE deltas
- `benchmark_aggregated.csv`: one row per `(case, model)` aggregate with run counts plus mean and standard deviation for validation metrics across seeds
- `benchmark_aggregated.json`: machine-readable JSON version of the aggregated table
- `benchmark_aggregated_summary.md`: markdown summary of mean score gaps, uncertainty-style spreads, and whether global beat local on all seeds
- `benchmark_robustness.csv`: one row per case with explicit local/global score gaps, seed-wise win flags, and robustness indicators
- `benchmark_robustness.json`: machine-readable JSON version of the robustness table
- `benchmark_robustness_summary.md`: compact narrative summary highlighting the hardest case by variance and by smallest separation
- `paper_table.csv`: compact paper-style export with local/global mean scores and robustness flags
- `cases/<case_id>/seed_<seed>/case_manifest.json`: explicit case definition, target, split protocol, bundle path, and run paths
- `bundles/`: generated or reused benchmark bundles grouped by case id and seed
- `runs/`: per-case experiment run directories grouped by case id, seed, and model

## Benchmark release artifacts

Produced by `geocydata experiments release`.

- `release_manifest.json`: top-level publication-facing release manifest, including `release_version` and `benchmark_contract_version`
- `release_protocol.json`: frozen preset plus harder-slice configuration, also versioned
- `final_results.csv` and `final_results.json`: publication-facing core benchmark result table
- `final_robustness.csv` and `final_robustness.json`: publication-facing robustness table for the core benchmark
- `final_summary.md`: concise release summary
- `results_memo.md`: manuscript-style internal results memo describing protocol, findings, and limitations
- `core_benchmark/`: preserved preset-based sweep outputs
- `harder_slice/`: optional preserved outputs for the harder evaluation slice

## Release validation artifacts

Produced by `geocydata experiments validate-release`.

- `release_validation_report.json`: machine-readable validation report covering required files, required directories, manifest/protocol consistency, target metadata, preset metadata, and release-version fields
- `release_validation_summary.md`: short markdown validation summary suitable for external release checks

## Paper asset artifacts

Produced by `geocydata experiments build-paper-assets`.

- `paper/outline.md`: manuscript structure scaffold
- `paper/abstract_draft.md`: concise benchmark-aligned abstract draft
- `paper/introduction_notes.md`: introduction notes and framing constraints
- `paper/methods_notes.md`: protocol, target, seed, and split summary derived from the release
- `paper/results_notes.md`: benchmark findings and claim boundaries derived from the release outputs
- `paper/reproducibility_notes.md`: regeneration and validation notes referencing the release contract
- `paper/assets/core_results_table.csv` and `.md`: core case/model summary table
- `paper/assets/robustness_table.csv` and `.md`: robustness summary table
- `paper/assets/fig_core_scores.png`: mean validation score figure with error bars for the core benchmark
- `paper/assets/fig_hardest_case.png`: harder-slice or hardest-case comparison figure

These assets are downstream views of the release outputs; they should be rebuilt from a validated release rather than edited by hand first.

## Paper asset validation artifacts

Produced by `geocydata experiments validate-paper-assets`.

- `paper_validation_report.json`: machine-readable report covering structural file checks, table consistency against release outputs, figure presence, and lightweight finding-consistency checks
- `paper_validation_summary.md`: short markdown summary of the same checks

The finding checks are intentionally pragmatic and use explicit content checks against the generated notes. They are designed to catch drift in benchmark-facing claims, not to perform semantic review of prose.

## Naming notes

GeoCYData does not force a bundle directory name. The user supplies the output path with `--out`, and that directory name is recorded in the manifest.
