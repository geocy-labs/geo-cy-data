# GeoCYData

GeoCYData is a command-line dataset generator and validator for projective hypersurface Calabi-Yau benchmarks.

## Why this project exists

Research workflows around Calabi-Yau benchmark data often start as one-off scripts or notebook fragments. GeoCYData provides a lean, reproducible command-line foundation for generating benchmark bundles, validating basic geometric constraints, and exporting documented artifacts that are easy to inspect and test.

## Key features

- Reproducible bundle generation for the Fermat quartic benchmark in `P^3`
- Parameterized bundle generation for the Cefalu quartic family in `P^3`
- First-class Paper 1 / GlobalCY Cefalu cases at `lambda = 0.0, 0.75, 1.0, 1.5, 3.0`
- Fixed near-singular Cefalu slice presets around `lambda = 0.75` and `lambda = 1.0`
- Symmetry-orbit generation and canonical representatives for the Cefalu quartic family
- Lightweight experiment runner for local-vs-global representation comparison
- Geometry-derived experiment targets, with `hypersurface_fs_scalar` as the preferred hypersurface-aware benchmark target
- Standardized benchmark sweeps and publication-facing release packaging with per-seed, aggregated, and robustness markdown/JSON result summaries
- Versioned release validation and regeneration workflow for external reproducibility
- Paper-facing asset generation from frozen release outputs, including manuscript notes, tables, and simple figures
- Release-to-paper consistency validation for manuscript-facing assets
- Simple branch-based sampler for complex homogeneous points on the hypersurface
- Affine chart extraction and projective invariant feature generation
- Bundle-level validation with residual and invariance drift reporting
- Installable Python package with CLI, tests, CI, and minimal public-facing docs

## Installation

GeoCYData targets Python 3.11 or newer.

```bash
git clone https://github.com/geocy-labs/geo-cy-data.git
cd geo-cy-data
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Verify the install:

```bash
geocydata --help
geocydata geometry list
geocydata geometry show --geometry cefalu_quartic --lambda 1.0
```

## Fastest smoke test

```bash
geocydata generate bundle --geometry fermat_quartic --n 2000 --seed 7 --out outputs/demo
```

Parameterized Cefalu examples:

```bash
geocydata generate bundle --geometry cefalu_quartic --lambda 0.0 --n 2000 --seed 7 --out outputs/cefalu_lambda_0_0
geocydata generate bundle --geometry cefalu_quartic --lambda 0.75 --n 2000 --seed 7 --out outputs/cefalu_lambda_0_75
geocydata generate bundle --geometry cefalu_quartic --lambda 1.0 --n 2000 --seed 7 --out outputs/cefalu_lambda_1_0
geocydata generate bundle --geometry cefalu_quartic --lambda 1.5 --n 2000 --seed 7 --out outputs/cefalu_lambda_1_5
geocydata generate bundle --geometry cefalu_quartic --lambda 3.0 --n 2000 --seed 7 --out outputs/cefalu_lambda_3_0
geocydata generate orbits --geometry cefalu_quartic --lambda 1.0 --n 200 --seed 7 --out outputs/cefalu_orbits
geocydata experiments compare --bundle outputs/cefalu_lambda_1_0 --target hypersurface_fs_scalar --out runs/compare_hfs
geocydata experiments sweep --out runs/phase8_sweep --target hypersurface_fs_scalar --seeds 7 11 19
geocydata experiments sweep --preset paper_v1_default --out runs/paper_v1
geocydata experiments sweep --preset globalcy_paper1_core --out runs/globalcy_paper1_core
geocydata experiments sweep --preset globalcy_paper1_near_0_75 --out runs/globalcy_near_0_75
geocydata experiments sweep --preset globalcy_paper1_near_1_0 --out runs/globalcy_near_1_0
geocydata experiments release --preset paper_v1_default --out releases/paper_v1_release --include-hard-slice
geocydata experiments regenerate-release --preset paper_v1_default --out releases/paper_v1_release_regenerated --include-hard-slice
geocydata experiments validate-release --input releases/paper_v1_release
geocydata experiments build-paper-assets --input releases/paper_v1_release --out paper
geocydata experiments validate-paper-assets --release releases/paper_v1_release --paper paper
```

## Expected output bundle

```text
outputs/demo/
  manifest.json
  case_metadata.json
  points.parquet
  invariants.parquet
  sample_weights.parquet
  validation_report.json
  evaluation_summary.json
  summary.md
```

For `cefalu_quartic`, the model-ready direct bundle export also writes:

```text
outputs/cefalu_lambda_1_0/
  canonical_representatives.parquet
  canonical_invariants.parquet
  orbits.parquet
  symmetry_report.json
```

The output directory name is user-chosen. `outputs/demo` is only an example; any empty or existing directory path is valid.

Validate the generated bundle:

```bash
geocydata validate bundle --input outputs/demo
geocydata validate symmetry --input outputs/cefalu_orbits
```

Run the benchmark experiment scaffold:

```bash
geocydata experiments run --bundle outputs/cefalu_lambda_1_0 --model local --target hypersurface_fs_scalar --out runs/local_hfs
geocydata experiments run --bundle outputs/cefalu_lambda_1_0 --model global --target hypersurface_fs_scalar --out runs/global_hfs
geocydata experiments compare --bundle outputs/cefalu_lambda_1_0 --target hypersurface_fs_scalar --out runs/compare_hfs
geocydata experiments sweep --out runs/phase8_sweep --target hypersurface_fs_scalar --seeds 7 11 19
geocydata experiments sweep --preset paper_v1_default --out runs/paper_v1
geocydata experiments release --preset paper_v1_default --out releases/paper_v1_release --include-hard-slice
```

Legacy compatibility target examples remain available if you need to reproduce the earlier ambient proxy workflow:

```bash
geocydata experiments run --bundle outputs/cefalu_lambda_1_0 --model local --target fs_scalar --out runs/local_fs_legacy
geocydata experiments run --bundle outputs/cefalu_lambda_1_0 --model global --target fs_scalar --out runs/global_fs_legacy
```

## Supported geometries

- `fermat_quartic`: Fermat quartic hypersurface in `P^3`
- `cefalu_quartic`: parameterized quartic family with required `--lambda`

## Paper 1 / GlobalCY cases

The current first-class Paper 1 Cefalu cases are:

- `cefalu_lambda_0_0`
- `cefalu_lambda_0_75`
- `cefalu_lambda_1_0`
- `cefalu_lambda_1_5`
- `cefalu_lambda_3_0`

Fixed near-singular slices are also available through protocol presets:

- `globalcy_paper1_near_0_75`: `0.74, 0.75, 0.76`
- `globalcy_paper1_near_1_0`: `0.99, 1.0, 1.01`

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Dataset schema](docs/dataset_schema.md)
- [CLI reference](docs/cli_reference.md)
- [Development guide](docs/development.md)

## Paper assets

Frozen benchmark releases are the source of truth for manuscript-facing assets. Build the paper scaffold, tables, and figures from a validated release with:

```bash
geocydata experiments build-paper-assets --input releases/paper_v1_release --out paper
```

This writes:

- `paper/outline.md`
- `paper/abstract_draft.md`
- `paper/introduction_notes.md`
- `paper/methods_notes.md`
- `paper/results_notes.md`
- `paper/reproducibility_notes.md`
- `paper/assets/core_results_table.csv`
- `paper/assets/core_results_table.md`
- `paper/assets/robustness_table.csv`
- `paper/assets/robustness_table.md`
- `paper/assets/fig_core_scores.png`
- `paper/assets/fig_hardest_case.png`

Validate that the paper layer still matches the frozen release:

```bash
geocydata experiments validate-paper-assets --release releases/paper_v1_release --paper paper
```

This writes:

- `paper/paper_validation_report.json`
- `paper/paper_validation_summary.md`

## Current limitations

- The current samplers are documented branch-based smoke-test methods, not uniform samplers over the hypersurfaces
- Symmetry orbits currently target `cefalu_quartic` only
- The preferred experiment target is `hypersurface_fs_scalar`, a tangent-restricted Fubini-Study proxy that uses the local hypersurface gradient; it is stronger than the older ambient `fs_scalar`, but still not Ricci-flat metric learning
- The Phase 6 sweep standardizes first benchmark results across Fermat and two Cefalu settings, but it remains a lightweight benchmark pipeline rather than a final physics experiment setup
- Phase 7 adds multi-seed aggregation and uncertainty-style summaries, but this is still benchmark hardening rather than final metric-learning or physics modeling
- Phase 8 strengthens protocol metadata and benchmark bookkeeping, but it remains benchmark/protocol hardening rather than final metric learning
- Phase 9 adds named protocol presets and explicit robustness reporting, but this is still benchmark-results hardening rather than final metric learning
- Phase 10 packages the benchmark as a publication-facing release with a harder evaluation slice and a manuscript-style memo, but it is still not final metric learning
- Phase 11 adds versioned release validation and regeneration for external trust and reuse, but it does not change the underlying benchmark model class
- Bundle validation is intentionally basic and meant to catch obvious issues in generated artifacts

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing expectations, and the scope of the current milestone.
