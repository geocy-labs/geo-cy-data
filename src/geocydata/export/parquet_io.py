"""Parquet export helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a dataframe to Parquet."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    return target

