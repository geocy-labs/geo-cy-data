# CLI Reference

## Top-level help

```bash
geocydata --help
```

The top-level CLI shows the public command groups:

- `geometry`: inspect supported benchmarks
- `generate`: create bundle directories
- `validate`: re-check existing bundles
- `experiments`: run, compare, and sweep lightweight representation benchmarks

## List geometries

```bash
geocydata geometry list
```

## Show one geometry

```bash
geocydata geometry show --geometry cefalu_quartic --lambda 1.0
```

For `cefalu_quartic`, `--lambda` is required. For `fermat_quartic`, omit it.

## Generate a bundle

```bash
geocydata generate bundle --geometry fermat_quartic --n 2000 --seed 7 --out outputs/demo
geocydata generate bundle --geometry cefalu_quartic --lambda 1.0 --n 2000 --seed 7 --out outputs/cefalu_lambda_1_0
```

Arguments:

- `--geometry`: registered geometry name
- `--n`: number of points to sample
- `--seed`: random seed for reproducibility
- `--lambda`: required for `cefalu_quartic`, omitted for `fermat_quartic`
- `--out`: bundle directory to create or reuse

On success, the command prints the output path and the artifact filenames written into that directory.

For Paper 1 / GlobalCY use, the fixed first-class Cefalu cases are:

- `--lambda 0.0`
- `--lambda 0.75`
- `--lambda 1.0`
- `--lambda 1.5`
- `--lambda 3.0`

Direct model-ready bundles include:

- local chart coordinates in `points.parquet`
- projective invariants in `invariants.parquet`
- `sample_weights.parquet`
- `case_metadata.json`
- `evaluation_summary.json`

For direct `cefalu_quartic` bundles, they also include:

- `canonical_representatives.parquet`
- `canonical_invariants.parquet`
- `orbits.parquet` with lightweight orbit metadata
- `symmetry_report.json`

## Generate symmetry orbits

```bash
geocydata generate orbits --geometry cefalu_quartic --lambda 1.0 --n 200 --seed 7 --out outputs/cefalu_orbits
```

This command exports sampled base points, orbit representatives, canonical-representative invariants, a symmetry report, and a summary markdown file.

## Validate an existing bundle

```bash
geocydata validate bundle --input outputs/demo
```

Validation expects a bundle directory containing `points.parquet`. If required files or columns are missing, the CLI returns a non-zero exit code with a direct error message.

## Validate symmetry output

```bash
geocydata validate symmetry --input outputs/cefalu_orbits
```

## Run one experiment

```bash
geocydata experiments run --bundle outputs/cefalu_lambda_1_0 --model local --target hypersurface_fs_scalar --out runs/local_hfs
geocydata experiments run --bundle outputs/cefalu_lambda_1_0 --model global --target hypersurface_fs_scalar --out runs/global_hfs
```

The preferred target is `hypersurface_fs_scalar`, a tangent-restricted Fubini-Study proxy that uses the local hypersurface gradient. The older `fs_scalar` ambient proxy and `invariant_weighted_sum` debug target both remain available.

This command loads `points.parquet` and `invariants.parquet`, constructs the selected target, splits the bundle reproducibly, and writes run artifacts:

- `config.json`
- `metrics.json`
- `predictions.parquet`
- `run_manifest.json`
- `summary.md`

## Compare local and global models

```bash
geocydata experiments compare --bundle outputs/cefalu_lambda_1_0 --target hypersurface_fs_scalar --out runs/compare_hfs
```

This command runs both model modes on the same bundle and writes:

- `local/`
- `global/`
- `comparison.json`
- `comparison.md`

## Run the standardized benchmark sweep

```bash
geocydata experiments sweep --out runs/phase8_sweep --target hypersurface_fs_scalar --seed 7
geocydata experiments sweep --out runs/phase8_sweep --target hypersurface_fs_scalar --seeds 7 11 19
geocydata experiments sweep --preset paper_v1_default --out runs/paper_v1
```

Default Phase 6 cases:

- `fermat_quartic`
- `cefalu_lambda_0_75`
- `cefalu_lambda_1_0`

Additional Paper 1 presets:

- `globalcy_paper1_core`: `fermat_quartic`, `cefalu_lambda_0_0`, `cefalu_lambda_0_75`, `cefalu_lambda_1_0`, `cefalu_lambda_1_5`, `cefalu_lambda_3_0`
- `globalcy_paper1_near_0_75`: `cefalu_lambda_0_74`, `cefalu_lambda_0_75`, `cefalu_lambda_0_76`
- `globalcy_paper1_near_1_0`: `cefalu_lambda_0_99`, `cefalu_lambda_1_0`, `cefalu_lambda_1_01`

Paper II / GlobalCY II preset:

- `cefalu_hard_regime_sweep_v1`: `cefalu_lambda_0_50`, `cefalu_lambda_0_75`, `cefalu_lambda_0_90`, `cefalu_lambda_1_0`, `cefalu_lambda_1_10`

Useful flags:

- `--preset`: named benchmark protocol preset such as `paper_v1_fast`, `paper_v1_default`, or `paper_v1_multiseed`
- `--target`: experiment target, with `hypersurface_fs_scalar` as the preferred current benchmark choice
- `--seed`: deterministic sweep seed; repeat the option for compatibility with existing single-seed workflows
- `--seeds`: provide multiple seeds after one flag, for example `--seeds 7 11 19`
- `--n`: samples per generated bundle
- `--include`: repeat to restrict the case matrix to selected benchmark ids
- `--test-size`: validation fraction for every run

This command writes:

- `benchmark_protocol.json`
- `benchmark_manifest.json`
- `benchmark_preset_manifest.json`
- `benchmark_results.csv`
- `benchmark_results.json`
- `benchmark_summary.md`
- `benchmark_aggregated.csv`
- `benchmark_aggregated.json`
- `benchmark_aggregated_summary.md`
- `benchmark_robustness.csv`
- `benchmark_robustness.json`
- `benchmark_robustness_summary.md`
- `paper_table.csv`
- `cases/<case_id>/seed_<seed>/case_manifest.json`
- per-case `bundles/` and `runs/` subdirectories

For `cefalu_hard_regime_sweep_v1`, the preset manifest is the handoff contract GlobalCY should read first. It records benchmark version, case ids, lambda values, geometry family, and the available model-facing bundle views.

## Create a publication-facing release

```bash
geocydata experiments release --preset paper_v1_default --out releases/paper_v1_release
geocydata experiments release --preset paper_v1_default --out releases/paper_v1_release --include-hard-slice --hard-slice cefalu_hard_v1
geocydata experiments regenerate-release --preset paper_v1_default --out releases/paper_v1_release_regenerated --include-hard-slice
geocydata experiments validate-release --input releases/paper_v1_release
geocydata experiments build-paper-assets --input releases/paper_v1_release --out paper
geocydata experiments validate-paper-assets --release releases/paper_v1_release --paper paper
```

This command freezes a preset-based benchmark release and writes:

- `release_manifest.json`
- `release_protocol.json`
- `final_results.csv`
- `final_results.json`
- `final_robustness.csv`
- `final_robustness.json`
- `final_summary.md`
- `results_memo.md`
- `core_benchmark/`
- optional `harder_slice/`

Phase 11 validation outputs:

- `release_validation_report.json`
- `release_validation_summary.md`

## Build manuscript-facing paper assets

```bash
geocydata experiments build-paper-assets --input releases/paper_v1_release --out paper
```

This command reads the frozen release outputs and writes:

- manuscript scaffold notes under `paper/`
- publication-facing tables under `paper/assets/`
- simple matplotlib figures under `paper/assets/`

The paper-assets command does not rerun the benchmark. It treats the validated release as the source of truth for tables, figures, and manuscript notes.

## Validate manuscript-facing paper assets

```bash
geocydata experiments validate-paper-assets --release releases/paper_v1_release --paper paper
```

This command checks:

- required paper scaffold files exist
- required release files exist
- paper CSV tables match the release-derived source tables
- figure files exist and are non-empty
- `results_notes.md` still contains the expected benchmark findings for the frozen release

It writes:

- `paper_validation_report.json`
- `paper_validation_summary.md`
