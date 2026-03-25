from __future__ import annotations

import ast

import numpy as np
import pandas as pd


def _parse_embedding_column(series: pd.Series) -> np.ndarray:
    """Parse a CSV embedding column (stringified list) to a float32 array (N, D)."""
    parsed = [np.asarray(ast.literal_eval(v), dtype=np.float32) for v in series.astype(str).tolist()]
    return np.stack(parsed, axis=0)


def load_tabular_text_dataset(
    *,
    path: str,
    date_column: str | None,
    numeric_features: list[str],
    target_column: str,
    embedding_lags: list[int],
    embedding_columns: list[str] | None,
    embedding_column_template: str | None,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load numeric features + target + text embeddings.

    Returns:
    - X: (N, F) numeric features
    - y: (N,) target
    - text: (N, L, D) text embeddings (lags are treated as separate text features)
    """
    df = pd.read_csv(path)
    df[target_column] = np.log1p(df[target_column].values)
    df[[col for col in numeric_features if "text_signal" not in col]] = np.log1p(df[[col for col in numeric_features if "text_signal" not in col]].values)
    df[[col for col in numeric_features if "text_signal" in col]] = df[[col for col in numeric_features if "text_signal"  in col]].values/6000

    # Sort by date to make chronological splits meaningful.
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)

    if max_rows is not None:
        df = df.head(max_rows).reset_index(drop=True)

    for col in [*numeric_features, target_column]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    X = df[numeric_features].astype(np.float32).to_numpy()
    y = df[target_column].astype(np.float32).to_numpy()

    if embedding_columns is not None:
        embedding_cols = embedding_columns
    else:
        if embedding_column_template is None:
            raise ValueError("Either embedding_columns or embedding_column_template must be provided.")
        embedding_cols = [embedding_column_template.format(lag=lag) for lag in embedding_lags]
    for col in embedding_cols:
        if col not in df.columns:
            raise ValueError(f"Missing embedding column '{col}' in {path}")

    # Each lag becomes a separate text feature: (N, L, D)
    by_lag = [_parse_embedding_column(df[col]) for col in embedding_cols]  # list[(N, D)]
    text = np.stack(by_lag, axis=1)  # (N, L, D)
    return X, y, text


# Backward-compatible alias for older scripts/notebooks.
load_climate_dataset = load_tabular_text_dataset
