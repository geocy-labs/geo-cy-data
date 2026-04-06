"""Lightweight sklearn model definitions for experiment runs."""

from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_LABELS = {
    "local": "LocalPhiMLP",
    "global": "GlobalInvariantPhi",
}


def build_regressor(model_name: str) -> Pipeline:
    """Build a lightweight regressor for the requested experiment mode."""

    if model_name not in MODEL_LABELS:
        raise ValueError(f"Unsupported experiment model '{model_name}'.")
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
