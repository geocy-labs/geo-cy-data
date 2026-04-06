# CLI Reference

## Top-level help

```bash
geocydata --help
```

The top-level CLI shows the three public command groups:

- `geometry`: inspect supported benchmarks
- `generate`: create bundle directories
- `validate`: re-check existing bundles

## List geometries

```bash
geocydata geometry list
```

## Generate a bundle

```bash
geocydata generate bundle --geometry fermat_quartic --n 2000 --seed 7 --out outputs/demo
```

Arguments:

- `--geometry`: registered geometry name
- `--n`: number of points to sample
- `--seed`: random seed for reproducibility
- `--out`: bundle directory to create or reuse

On success, the command prints the output path and the artifact filenames written into that directory.

## Validate an existing bundle

```bash
geocydata validate bundle --input outputs/demo
```

Validation expects a bundle directory containing `points.parquet`. If required files or columns are missing, the CLI returns a non-zero exit code with a direct error message.
