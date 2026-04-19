from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from geocydata.experiments.degeneration_fragility import export_degeneration_fragility
from geocydata.experiments.sweep import ensure_benchmark_bundle
from geocydata.registry.cases import get_case


def _materialize_case_seed_bundle(root: Path, *, case_id: str, seed: int, n: int = 12) -> None:
    case = get_case(case_id)
    ensure_benchmark_bundle(
        case=case,
        bundle_dir=root / "bundles" / case_id / f"seed_{seed}",
        n=n,
        seed=seed,
    )


def test_export_degeneration_fragility_outputs(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "cefalu_hard_regime_sweep_v1"
    cases = (
        "cefalu_lambda_0_50",
        "cefalu_lambda_0_75",
        "cefalu_lambda_0_90",
        "cefalu_lambda_1_0",
        "cefalu_lambda_1_10",
    )
    for case_id in cases:
        _materialize_case_seed_bundle(sweep_dir, case_id=case_id, seed=7)

    out_dir = tmp_path / "artifacts"
    result = export_degeneration_fragility(sweep_dir, output_dir=out_dir)

    csv_path = Path(result["csv_path"])
    json_path = Path(result["json_path"])
    parquet_path = Path(result["parquet_path"])

    assert csv_path.exists()
    assert json_path.exists()
    assert parquet_path.exists()

    casewise = pd.read_csv(csv_path)
    pointwise = pd.read_parquet(parquet_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert set(casewise["case_id"]) == set(cases)
    assert "fragility_q05" in casewise.columns
    assert "fragility_q10" in casewise.columns
    assert "fragility_mean" in casewise.columns
    assert "fragility_frac_below_eps" in casewise.columns

    assert "point_id" in pointwise.columns
    assert "case_id" in pointwise.columns
    assert "seed" in pointwise.columns
    assert "fragility_score" in pointwise.columns
    assert "fragile_flag" in pointwise.columns
    assert "fragile_cluster_id" in pointwise.columns

    assert payload["proxy_name"] == "cefalu_gradient_norm"
    assert payload["thresholding"]["eps_value"] > 0.0
