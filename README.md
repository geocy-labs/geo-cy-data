# GeoCYData

GeoCYData is a command-line dataset generator and validator for projective hypersurface Calabi-Yau benchmarks.

## Why this project exists

Research workflows around Calabi-Yau benchmark data often start as one-off scripts or notebook fragments. GeoCYData provides a lean, reproducible command-line foundation for generating benchmark bundles, validating basic geometric constraints, and exporting documented artifacts that are easy to inspect and test.

## Key features

- Reproducible bundle generation for the Fermat quartic benchmark in `P^3`
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
```

## Fastest smoke test

```bash
geocydata generate bundle --geometry fermat_quartic --n 2000 --seed 7 --out outputs/demo
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
```

## Supported geometries

- `fermat_quartic`: Fermat quartic hypersurface in `P^3`
- More benchmark geometries are planned, but not included in `v0.1`

## Documentation

- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Dataset schema](docs/dataset_schema.md)
- [CLI reference](docs/cli_reference.md)
- [Development guide](docs/development.md)

## Current limitations

- `v0.1` supports only `fermat_quartic`
- The sampler is a documented branch-based smoke-test method, not a uniform sampler over the hypersurface
- Bundle validation is intentionally basic and meant to catch obvious issues in generated artifacts

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, testing expectations, and the scope of the current milestone.
