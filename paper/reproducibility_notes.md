# Reproducibility Notes

- Release path: `C:\Users\fearl\OneDrive\Documents\ar\geo-cy-data\releases\paper_v1_release_regenerated`
- Release version: `paper_v1_release_v1`
- Benchmark contract version: `paper_v1_contract_v1`
- Preset: `paper_v1_default`
- Preferred target: `hypersurface_fs_scalar`

## Regeneration

```bash
geocydata experiments regenerate-release --preset paper_v1_default --out releases/paper_v1_release_regenerated_regenerated --include-hard-slice
```

## Validation

```bash
geocydata experiments validate-release --input C:\Users\fearl\OneDrive\Documents\ar\geo-cy-data\releases\paper_v1_release_regenerated
```

## Paper assets

```bash
geocydata experiments build-paper-assets --input C:\Users\fearl\OneDrive\Documents\ar\geo-cy-data\releases\paper_v1_release_regenerated --out paper
```

- The release outputs remain the source of truth for the paper tables, figures, and manuscript notes.
- Validation should succeed before paper-facing assets are circulated externally.
