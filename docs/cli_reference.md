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
