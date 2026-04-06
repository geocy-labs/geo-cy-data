# Methods Notes

- Frozen protocol preset: `paper_v1_default`
- Target: `hypersurface_fs_scalar`
- Seeds: `7, 11, 19`
- Cases: `fermat_quartic, cefalu_lambda_0_75, cefalu_lambda_1_0`
- Samples per case: `200`
- Split strategy: `deterministic_random_train_validation_split`
- Models: local affine baseline and global invariant baseline, both using the existing lightweight sklearn runner
- Source of truth: release outputs under `final_results.csv`, `final_robustness.csv`, and the preserved benchmark subdirectories
