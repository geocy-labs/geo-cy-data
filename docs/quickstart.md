# Quickstart

## Clone and install

```bash
git clone https://github.com/geocy-labs/geo-cy-data.git
cd geo-cy-data
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Confirm the CLI is available:

```bash
geocydata --help
geocydata geometry list
geocydata geometry show --geometry cefalu_quartic --lambda 1.0
```

## Generate a smoke-test bundle

```bash
geocydata generate bundle --geometry fermat_quartic --n 2000 --seed 7 --out outputs/demo
```

Generate a Cefalu family bundle:

```bash
geocydata generate bundle --geometry cefalu_quartic --lambda 0.75 --n 2000 --seed 7 --out outputs/cefalu_lambda_0_75
```

Paper 1 / GlobalCY fixed Cefalu cases:

```bash
geocydata generate bundle --geometry cefalu_quartic --lambda 0.0 --n 512 --seed 7 --out outputs/cefalu_lambda_0_0
geocydata generate bundle --geometry cefalu_quartic --lambda 0.75 --n 512 --seed 7 --out outputs/cefalu_lambda_0_75
geocydata generate bundle --geometry cefalu_quartic --lambda 1.0 --n 512 --seed 7 --out outputs/cefalu_lambda_1_0
geocydata generate bundle --geometry cefalu_quartic --lambda 1.5 --n 512 --seed 7 --out outputs/cefalu_lambda_1_5
geocydata generate bundle --geometry cefalu_quartic --lambda 3.0 --n 512 --seed 7 --out outputs/cefalu_lambda_3_0
```

Generate Cefalu symmetry orbits:

```bash
geocydata generate orbits --geometry cefalu_quartic --lambda 1.0 --n 200 --seed 7 --out outputs/cefalu_orbits
```

Run the first experiment scaffold:

```bash
geocydata experiments run --bundle outputs/cefalu_lambda_0_75 --model local --target hypersurface_fs_scalar --out runs/local_hfs
geocydata experiments run --bundle outputs/cefalu_lambda_0_75 --model global --target hypersurface_fs_scalar --out runs/global_hfs
geocydata experiments compare --bundle outputs/cefalu_lambda_0_75 --target hypersurface_fs_scalar --out runs/compare_hfs
geocydata experiments sweep --out runs/phase8_sweep --target hypersurface_fs_scalar --seeds 7 11 19
geocydata experiments sweep --preset paper_v1_default --out runs/paper_v1
geocydata experiments sweep --preset globalcy_paper1_core --out runs/globalcy_paper1_core
geocydata experiments sweep --preset globalcy_paper1_near_0_75 --out runs/globalcy_near_0_75
geocydata experiments sweep --preset globalcy_paper1_near_1_0 --out runs/globalcy_near_1_0
geocydata experiments sweep --preset cefalu_hard_regime_sweep_v1 --out runs/cefalu_hard_regime_v1
geocydata experiments release --preset paper_v1_default --out releases/paper_v1_release --include-hard-slice
geocydata experiments regenerate-release --preset paper_v1_default --out releases/paper_v1_release_regenerated --include-hard-slice
geocydata experiments validate-release --input releases/paper_v1_release
geocydata experiments build-paper-assets --input releases/paper_v1_release --out paper
geocydata experiments validate-paper-assets --release releases/paper_v1_release --paper paper
```

## Inspect outputs

The bundle directory contains:

- `manifest.json`: metadata for the run and artifact paths
- `case_metadata.json`: explicit `geometry`, `lambda`, `seed`, and `case_id`
- `points.parquet`: sampled homogeneous coordinates, chosen chart ids, affine chart coordinates, and case metadata
- `invariants.parquet`: projective invariant features plus case metadata
- `sample_weights.parquet`: model-facing sample weights with uniform, chart-balance, symmetry, and combined weights
- `validation_report.json`: dataset-level checks
- `evaluation_summary.json`: geometry-aware export hooks for chart consistency, invariance drift, symmetry consistency, positivity, and Euler-style summaries
- `summary.md`: human-readable run summary

For direct `cefalu_quartic` bundles, GeoCYData also writes:

- `canonical_representatives.parquet`
- `canonical_invariants.parquet`
- `orbits.parquet`
- `symmetry_report.json`

The Phase 6 benchmark sweep directory contains:

- `benchmark_manifest.json`: benchmark matrix, seeds, and artifact metadata
- `benchmark_results.csv`: tidy benchmark table across geometry cases and model modes
- `benchmark_results.json`: machine-readable copy of the same results
- `benchmark_summary.md`: markdown summary of per-case winners and metric deltas

With multiple seeds, the sweep directory also contains:

- `benchmark_aggregated.csv`: per-case and per-model means, standard deviations, and run counts
- `benchmark_aggregated.json`: machine-readable aggregated statistics
- `benchmark_aggregated_summary.md`: markdown uncertainty-style summary across seeds

Each experiment run also writes:

- `run_manifest.json`: explicit run protocol metadata including split strategy, split seed, feature mode, and benchmark case id

Each sweep case/seed combination writes:

- `cases/<case_id>/seed_<seed>/case_manifest.json`: explicit case definition, bundle path, run paths, target, and split protocol

Phase 9 also adds:

- `benchmark_protocol.json`: resolved preset plus any explicit overrides
- `benchmark_preset_manifest.json`: preset-level contract for downstream consumers such as GlobalCY
- `benchmark_robustness.csv` and `benchmark_robustness.json`: case-level robustness comparisons between local and global
- `benchmark_robustness_summary.md`: paper-style narrative summary of score gaps and seed robustness
- `paper_table.csv`: compact table suitable for later paper/report integration

Phase 10 release packaging adds:

- `release_manifest.json`: top-level release manifest
- `release_protocol.json`: frozen preset plus harder-slice configuration
- `final_results.csv` and `final_results.json`: publication-facing core benchmark table
- `final_robustness.csv` and `final_robustness.json`: publication-facing robustness table
- `final_summary.md`: concise release summary
- `results_memo.md`: manuscript-style internal results memo
- `core_benchmark/` and optional `harder_slice/`: underlying benchmark outputs kept intact

Phase 11 adds:

- `release_version` and `benchmark_contract_version` in release metadata
- `release_validation_report.json`: machine-readable release validation status
- `release_validation_summary.md`: short markdown validation summary

Phase 12A adds a paper-facing layer built from release outputs:

- `paper/outline.md`, `paper/abstract_draft.md`, `paper/introduction_notes.md`
- `paper/methods_notes.md`, `paper/results_notes.md`, `paper/reproducibility_notes.md`
- `paper/assets/core_results_table.csv` and `.md`
- `paper/assets/robustness_table.csv` and `.md`
- `paper/assets/fig_core_scores.png`
- `paper/assets/fig_hardest_case.png`

The release remains the source of truth. Build or refresh the paper assets only after the release has been regenerated and validated.

Phase 12B adds release-to-paper consistency validation:

- `paper_validation_report.json`: machine-readable validation report for the paper layer
- `paper_validation_summary.md`: short markdown summary of structural, table, figure, and finding checks

This validation checks that paper tables remain aligned with the release outputs and that the manuscript notes still encode the expected benchmark findings. It does not perform semantic language understanding beyond explicit content checks.

For external regeneration, rerun the same release preset:

```bash
geocydata experiments regenerate-release --preset paper_v1_default --out releases/paper_v1_release_regenerated --include-hard-slice
```

To validate an existing bundle:

```bash
geocydata validate bundle --input outputs/demo
geocydata validate symmetry --input outputs/cefalu_orbits
```

If validation succeeds, the JSON report will end with `"passed": true`.
