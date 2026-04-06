# GeoCYData

GeoCYData is a command-line dataset generator and validator for projective hypersurface Calabi-Yau benchmarks.

## Why this project exists

Research workflows around Calabi-Yau benchmark data often start as one-off scripts or notebook fragments. GeoCYData provides a lean, reproducible command-line foundation for generating benchmark bundles, validating basic geometric constraints, and exporting documented artifacts that are easy to inspect and test.

## Key features

- Reproducible bundle generation for the Fermat quartic benchmark in `P^3`
- Parameterized bundle generation for the Cefalu quartic family in `P^3`
- Symmetry-orbit generation and canonical representatives for the Cefalu quartic family
- Lightweight experiment runner for local-vs-global representation comparison
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
geocydata generate bundle --geometry cefalu_quartic --lambda 0.75 --n 2000 --seed 7 --out outputs/cefalu_lambda_0_75
geocydata generate bundle --geometry cefalu_quartic --lambda 1.0 --n 2000 --seed 7 --out outputs/cefalu_lambda_1_0
geocydata generate orbits --geometry cefalu_quartic --lambda 1.0 --n 200 --seed 7 --out outputs/cefalu_orbits
geocydata experiments compare --bundle outputs/cefalu_lambda_1_0 --out runs/compare_cep1
```

## Expected output bundle

```text
outputs/demo/
  manifest.json
  points.parquet
  invariants.parquet
  validation_report.json
  summary.md
```

The output directory name is user-chosen. `outputs/demo` is only an example; any empty or existing directory path is valid.

Validate the generated bundle:

```bash
geocydata validate bundle --input outputs/demo
geocydata validate symmetry --input outputs/cefalu_orbits
```

Run the first experiment scaffold:

```bash
geocydata experiments run --bundle outputs/cefalu_lambda_1_0 --model local --out runs/local_cep1
geocydata experiments run --bundle outputs/cefalu_lambda_1_0 --model global --out runs/global_cep1
```

## Supported geometries

- `fermat_quartic`: Fermat quartic hypersurface in `P^3`
- `cefalu_quartic`: parameterized quartic family with required `--lambda`

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Dataset schema](docs/dataset_schema.md)
- [CLI reference](docs/cli_reference.md)
- [Development guide](docs/development.md)

## Current limitations

- The current samplers are documented branch-based smoke-test methods, not uniform samplers over the hypersurfaces
- Symmetry orbits currently target `cefalu_quartic` only
- The experiment runner compares simple sklearn baselines on a deterministic bundle-derived target; it is not yet Ricci-flat metric learning
- Bundle validation is intentionally basic and meant to catch obvious issues in generated artifacts

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing expectations, and the scope of the current milestone.
