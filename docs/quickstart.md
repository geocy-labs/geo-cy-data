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
```

## Generate a smoke-test bundle

```bash
geocydata generate bundle --geometry fermat_quartic --n 2000 --seed 7 --out outputs/demo
```

## Inspect outputs

The bundle directory contains:

- `manifest.json`: metadata for the run and artifact paths
- `points.parquet`: sampled homogeneous and affine point data
- `invariants.parquet`: projective invariant features
- `validation_report.json`: dataset-level checks
- `summary.md`: human-readable run summary

To validate an existing bundle:

```bash
geocydata validate bundle --input outputs/demo
```

If validation succeeds, the JSON report will end with `"passed": true`.
