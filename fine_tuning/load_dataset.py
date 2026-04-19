from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from tabdpt.feature_build import generate_calendar_features, generate_causal_seasonality_features


def _parse_embedding_column(series: pd.Series) -> np.ndarray:
    """Parse a CSV embedding column (stringified list) to a float32 array (N, D)."""
    parsed = [np.asarray(ast.literal_eval(v), dtype=np.float32) for v in series.astype(str).tolist()]
    return np.stack(parsed, axis=0)


def validate_direct_mode_numeric_features(numeric_features: list[str]) -> None:
    """
    Direct mode only supports causally reconstructible numeric rows.

    All numeric features must therefore be lag-aligned columns.
    """
    unsupported = [feature for feature in numeric_features if "_lag" not in feature]
    if unsupported:
        raise ValueError(
            "Direct mode only supports lag-derived numeric features. "
            f"Unsupported feature(s): {unsupported}"
        )


def _infer_calendar_frequency(timestamps: pd.Series) -> str:
    inferred = pd.infer_freq(pd.DatetimeIndex(timestamps))
    if inferred is None:
        raise ValueError("Could not infer timestamp frequency for causal calendar features.")
    inferred = inferred.upper()
    if inferred.startswith("H"):
        return "hour"
    if inferred.startswith("D"):
        return "day"
    if inferred.startswith("W"):
        return "week"
    if inferred.startswith("M"):
        return "month"
    if inferred.startswith("Q"):
        return "quarter"
    if inferred.startswith("Y") or inferred.startswith("A"):
        return "year"
    raise ValueError(f"Unsupported inferred frequency {inferred!r} for causal calendar features.")


def _empty_text_like(text: np.ndarray) -> np.ndarray:
    return np.zeros((0, *text.shape[1:]), dtype=np.float32)


def _build_source_target_rows(
    *,
    X_source: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
    target_calendar: np.ndarray,
    source_indices: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if source_indices.size == 0:
        feature_dim = X_source.shape[1] + target_calendar.shape[1]
        return (
            np.zeros((0, feature_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            _empty_text_like(text),
        )

    target_indices = source_indices + horizon - 1
    X = np.concatenate(
        [X_source[source_indices], target_calendar[target_indices]],
        axis=1,
    ).astype(np.float32, copy=False)
    y_out = y[target_indices].astype(np.float32, copy=False)
    text_out = text[source_indices].astype(np.float32, copy=False)
    return X, y_out, text_out


def load_tabular_text_dataset_with_timestamps(
    *,
    path: str,
    date_column: str | None,
    numeric_features: list[str],
    target_column: str,
    embedding_lags: list[int],
    embedding_columns: list[str] | None,
    embedding_column_template: str | None,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series | None]:
    """
    Load numeric features + target + text embeddings + timestamps.

    Temporal features are rebuilt causally per horizon at runtime, so config
    numeric features should only contain the source-side numeric block.
    """
    df = pd.read_csv(path)

    timestamps: pd.Series | None = None
    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)
        timestamps = df[date_column].copy()

    if max_rows is not None:
        df = df.head(max_rows).reset_index(drop=True)
        if timestamps is not None:
            timestamps = timestamps.head(max_rows).reset_index(drop=True)

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

    by_lag = [_parse_embedding_column(df[col]) for col in embedding_cols]
    text = np.stack(by_lag, axis=1) if by_lag else np.zeros((len(df), 0, 0), dtype=np.float32)
    return X, y, text, timestamps


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
    X, y, text, _ = load_tabular_text_dataset_with_timestamps(
        path=path,
        date_column=date_column,
        numeric_features=numeric_features,
        target_column=target_column,
        embedding_lags=embedding_lags,
        embedding_columns=embedding_columns,
        embedding_column_template=embedding_column_template,
        max_rows=max_rows,
    )
    return X, y, text


def build_causal_fixed_origin_horizon_splits(
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
    timestamps: pd.Series,
    *,
    context_ratio: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    horizon: int,
    calendar_frequency: str | None = None,
    seasonality_k: int = 3,
    seasonality_L: int | None = None,
) -> tuple[np.ndarray, ...]:
    """
    Build causal horizon-specific splits.

    For horizon h, each row uses:
    - source observed features from row t
    - source causal seasonality for row t built from observations through t - 1
    - target calendar from timestamp t + h - 1
    - label y[t + h - 1]

    Context rows are the initially observable prefix before the first unseen
    source row. Train/val/test rows are source-row blocks, so the first eval row
    in each split is the first unseen source row for that split.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if len(X) != len(y) or len(X) != len(text) or len(X) != len(timestamps):
        raise ValueError(
            "X, y, text, and timestamps must have equal length. "
            f"Got len(X)={len(X)}, len(y)={len(y)}, len(text)={len(text)}, len(timestamps)={len(timestamps)}."
        )

    n = len(y)
    ratio_sum = context_ratio + train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}")

    n_context = int(n * context_ratio)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_context - n_train - n_val
    if min(n_context, n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Invalid split sizes: n={n}, context={n_context}, train={n_train}, val={n_val}, test={n_test}"
        )

    timestamps = pd.Series(pd.to_datetime(timestamps), copy=False).reset_index(drop=True)
    if calendar_frequency is None:
        calendar_frequency = _infer_calendar_frequency(timestamps)
    target_calendar = generate_calendar_features(timestamps, calendar_frequency).astype(np.float32, copy=False)
    seasonality_by_observed_row = generate_causal_seasonality_features(
        y,
        k=seasonality_k,
        L=seasonality_L,
    ).astype(np.float32, copy=False)
    if seasonality_by_observed_row.shape[1] > 0:
        leading = np.full((1, seasonality_by_observed_row.shape[1]), np.nan, dtype=np.float32)
        source_seasonality = np.concatenate(
            [leading, seasonality_by_observed_row[:-1]],
            axis=0,
        ).astype(np.float32, copy=False)
    else:
        source_seasonality = seasonality_by_observed_row
    X_source = X.astype(np.float32, copy=False)
    if source_seasonality.shape[1] > 0:
        X_source = np.concatenate([X_source, source_seasonality], axis=1).astype(np.float32, copy=False)

    train_end = n_context + n_train
    val_end = train_end + n_val

    context_sources = np.arange(max(0, n_context - (horizon - 1)), dtype=np.int64)
    train_sources = np.arange(n_context, train_end, dtype=np.int64)
    val_sources = np.arange(train_end, val_end, dtype=np.int64)
    test_sources = np.arange(val_end, n - (horizon - 1), dtype=np.int64)

    X_context, y_context, text_context = _build_source_target_rows(
        X_source=X_source,
        y=y,
        text=text,
        target_calendar=target_calendar,
        source_indices=context_sources,
        horizon=horizon,
    )
    X_train, y_train, text_train = _build_source_target_rows(
        X_source=X_source,
        y=y,
        text=text,
        target_calendar=target_calendar,
        source_indices=train_sources,
        horizon=horizon,
    )
    X_val, y_val, text_val = _build_source_target_rows(
        X_source=X_source,
        y=y,
        text=text,
        target_calendar=target_calendar,
        source_indices=val_sources,
        horizon=horizon,
    )
    X_test, y_test, text_test = _build_source_target_rows(
        X_source=X_source,
        y=y,
        text=text,
        target_calendar=target_calendar,
        source_indices=test_sources,
        horizon=horizon,
    )
    return (
        X_context,
        y_context,
        text_context,
        X_train,
        y_train,
        text_train,
        X_val,
        y_val,
        text_val,
        X_test,
        y_test,
        text_test,
    )


def build_direct_multi_horizon_dataset(
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
    *,
    prediction_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Legacy direct multi-horizon alignment retained for compatibility with tests/tools.
    """
    if prediction_window <= 0:
        raise ValueError("prediction_window must be positive.")

    if len(X) != len(y) or len(X) != len(text):
        raise ValueError(
            "X, y, and text must contain the same number of rows. "
            f"Got len(X)={len(X)}, len(y)={len(y)}, len(text)={len(text)}."
        )

    if prediction_window > len(y):
        raise ValueError(
            "prediction_window is too large for the dataset length. "
            f"Got len(y)={len(y)} and prediction_window={prediction_window}."
        )

    X_aligned = X
    text_aligned = text
    Y_multi = np.full((len(y), prediction_window), np.nan, dtype=np.float32)
    for horizon_offset in range(prediction_window):
        usable_rows = len(y) - horizon_offset
        Y_multi[:usable_rows, horizon_offset] = y[horizon_offset:].astype(np.float32, copy=False)
    return X_aligned, Y_multi, text_aligned


def select_direct_horizon_targets(
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy direct-horizon selector retained for compatibility with older tests/tools."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    if y.ndim == 1:
        if horizon != 1:
            raise ValueError(f"Requested horizon {horizon}, but the loaded targets are single-horizon.")
        return X, y.astype(np.float32, copy=False), text

    prediction_window = int(y.shape[1])
    if horizon > prediction_window:
        raise ValueError(
            f"Requested horizon {horizon}, but only {prediction_window} horizon(s) are available."
        )

    horizon_targets = y[:, horizon - 1].astype(np.float32, copy=False)
    valid_rows = np.isfinite(horizon_targets)
    if np.all(valid_rows):
        return X, horizon_targets, text
    return X[valid_rows], horizon_targets[valid_rows], text[valid_rows]


# Backward-compatible alias for older scripts/notebooks.
load_climate_dataset = load_tabular_text_dataset
