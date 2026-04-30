#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATHS = [
    Path("/home/yuyan/tabdpt_mz/data/trimmed/health_afr_gpt_analyzed_embedding.csv"),
    Path("/home/yuyan/tabdpt_mz/data/trimmed/health_us_gpt_analyzed_embedding.csv"),
]


def _fill_ot_with_neighbor_mean(series: pd.Series, *, window: int = 7) -> pd.Series:
    filled = series.copy()
    original = series.copy()

    missing_indices = np.flatnonzero(original.isna().to_numpy())
    for idx in missing_indices:
        start = max(0, idx - window)
        stop = min(len(original), idx + window + 1)

        prev_values = original.iloc[start:idx]
        next_values = original.iloc[idx + 1 : stop]
        neighbors = pd.concat([prev_values, next_values], ignore_index=True).dropna()
        if not neighbors.empty:
            filled.iloc[idx] = float(neighbors.mean())

    return filled


def _rebuild_ot_lags(df: pd.DataFrame) -> None:
    if "OT" not in df.columns:
        return

    lag_columns = [col for col in df.columns if col.startswith("OT_lag")]
    for col in lag_columns:
        try:
            lag = int(col.removeprefix("OT_lag"))
        except ValueError:
            continue
        df[col] = df["OT"].shift(lag)


def clean_health_file(path: Path) -> None:
    df = pd.read_csv(path)

    # Coerce any infinities to NaN across the full table first.
    df = df.replace([np.inf, -np.inf], np.nan)

    if "OT" in df.columns:
        df["OT"] = pd.to_numeric(df["OT"], errors="coerce")
        df["OT"] = _fill_ot_with_neighbor_mean(df["OT"], window=7)
        _rebuild_ot_lags(df)

    df.to_csv(path, index=False)


def main() -> None:
    for path in DATA_PATHS:
        clean_health_file(path)
        print(f"cleaned {path}")


if __name__ == "__main__":
    main()
