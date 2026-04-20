from __future__ import annotations

import ast
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fine_tuning.eval_fine_tune import _compute_metrics
from tabdpt.feature_build import generate_calendar_features, generate_causal_seasonality_features
from tabdpt.utils import pad_x


MetricTriplet = tuple[float, float, float]

_LAGGED_FEATURE_RE = re.compile(r"^(?P<base>.+)_lag(?P<lag>\d+)$")
_FREQUENCY_TO_PANDAS = {
    "hour": "H",
    "day": "D",
    "week": "W",
    "month": "MS",
    "quarter": "QS",
    "year": "YS",
}


@dataclass(frozen=True)
class EmbeddingInputSpec:
    lag: int
    input_column: str
    lag0_column: str


@dataclass(frozen=True)
class BatchQueryBundle:
    batch_horizon: int
    source_indices: np.ndarray
    source_timestamps: pd.Series
    target_timestamps: pd.Series
    X_query_raw: np.ndarray
    text_query: np.ndarray | None


@dataclass(frozen=True)
class BatchSlotMetrics:
    slot: int
    count: int
    normalized: MetricTriplet
    real: MetricTriplet


@dataclass(frozen=True)
class BatchForecastRecord:
    slot: int
    source_index: int
    source_timestamp: str
    target_timestamp: str
    prediction_real: float
    model_name: str


def resolve_batch_horizon(prediction_window: int) -> int:
    if prediction_window <= 0:
        raise ValueError("prediction_window must be positive for batch mode.")
    return int(prediction_window)


def _parse_lagged_feature_name(name: str) -> tuple[str, int]:
    match = _LAGGED_FEATURE_RE.fullmatch(name)
    if match is None:
        raise ValueError(
            "Batch mode requires lag-derived feature names in the form '<base>_lagN'. "
            f"Got {name!r}."
        )
    return match.group("base"), int(match.group("lag"))


def _empty_text_like(text: np.ndarray | None) -> np.ndarray | None:
    if text is None:
        return None
    return np.zeros((0, *text.shape[1:]), dtype=np.float32)


def infer_calendar_frequency(timestamps: pd.Series) -> str:
    inferred = pd.infer_freq(pd.DatetimeIndex(timestamps))
    if inferred is None:
        raise ValueError("Could not infer timestamp frequency for batch mode.")
    inferred = inferred.upper()
    if inferred.startswith("H"):
        return "hour"
    if inferred.startswith("D"):
        return "day"
    if inferred.startswith("W"):
        return "week"
    if inferred.startswith("MS") or inferred.startswith("M"):
        return "month"
    if inferred.startswith("QS") or inferred.startswith("Q"):
        return "quarter"
    if inferred.startswith("YS") or inferred.startswith("Y") or inferred.startswith("AS") or inferred.startswith("A"):
        return "year"
    raise ValueError(f"Unsupported inferred frequency {inferred!r} for batch mode.")


def build_future_timestamps(
    timestamps: pd.Series,
    *,
    horizon: int,
    calendar_frequency: str | None,
) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    ts = pd.Series(pd.to_datetime(timestamps), copy=False).reset_index(drop=True)
    if len(ts) == 0:
        raise ValueError("Cannot build future timestamps from an empty timestamp series.")
    frequency = calendar_frequency or infer_calendar_frequency(ts)
    pandas_frequency = _FREQUENCY_TO_PANDAS.get(frequency)
    if pandas_frequency is None:
        raise ValueError(f"Unsupported calendar_frequency {frequency!r} for batch mode.")
    future = pd.date_range(ts.iloc[-1], periods=horizon + 1, freq=pandas_frequency)[1:]
    return pd.Series(future)


def build_source_feature_block(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    *,
    seasonality_k: int,
    seasonality_L: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    X_raw = X_raw.astype(np.float32, copy=False)
    seasonality_by_observed_row = generate_causal_seasonality_features(
        y_raw,
        k=seasonality_k,
        L=seasonality_L,
    ).astype(np.float32, copy=False)
    if seasonality_by_observed_row.shape[1] == 0:
        return X_raw, seasonality_by_observed_row

    leading = np.full((1, seasonality_by_observed_row.shape[1]), np.nan, dtype=np.float32)
    source_seasonality = np.concatenate([leading, seasonality_by_observed_row[:-1]], axis=0).astype(
        np.float32,
        copy=False,
    )
    source_features = np.concatenate([X_raw, source_seasonality], axis=1).astype(np.float32, copy=False)
    return source_features, seasonality_by_observed_row


def _parse_embedding_cell(value: object) -> np.ndarray:
    return np.asarray(ast.literal_eval(str(value)), dtype=np.float32)


def resolve_embedding_input_specs(
    *,
    embedding_lags: list[int],
    embedding_columns: list[str] | None,
    embedding_column_template: str | None,
    df: pd.DataFrame,
) -> list[EmbeddingInputSpec]:
    if not embedding_lags and not embedding_columns:
        return []

    if embedding_columns is not None:
        specs: list[EmbeddingInputSpec] = []
        for col in embedding_columns:
            base, lag = _parse_lagged_feature_name(col)
            lag0_column = f"{base}_lag0"
            if lag0_column not in df.columns:
                raise ValueError(
                    f"Batch mode cannot synthesize future text for {col!r}: missing required lag-0 column "
                    f"{lag0_column!r}."
                )
            specs.append(EmbeddingInputSpec(lag=lag, input_column=col, lag0_column=lag0_column))
        return specs

    if embedding_column_template is None:
        raise ValueError("Either embedding_columns or embedding_column_template must be provided.")

    specs = []
    for lag in embedding_lags:
        input_column = embedding_column_template.format(lag=lag)
        lag0_column = embedding_column_template.format(lag=0)
        if lag0_column not in df.columns:
            raise ValueError(
                f"Batch mode cannot synthesize future text for {input_column!r}: missing required lag-0 column "
                f"{lag0_column!r}."
            )
        specs.append(EmbeddingInputSpec(lag=int(lag), input_column=input_column, lag0_column=lag0_column))
    return specs


def _build_synthetic_numeric_row(
    df: pd.DataFrame,
    *,
    numeric_features: list[str],
    source_index: int,
) -> np.ndarray:
    row_values: list[float] = []
    for feature in numeric_features:
        base, lag = _parse_lagged_feature_name(feature)
        if base not in df.columns:
            raise ValueError(
                f"Batch mode cannot synthesize future numeric feature {feature!r}: missing raw base column "
                f"{base!r}."
            )
        raw_index = source_index - lag
        if raw_index < 0 or raw_index >= len(df):
            raise ValueError(
                f"Batch mode cannot synthesize future numeric feature {feature!r} for source index {source_index}: "
                f"required raw row {raw_index} is unavailable."
            )
        row_values.append(float(df.iloc[raw_index][base]))
    return np.asarray(row_values, dtype=np.float32)


def _build_synthetic_text_row(
    df: pd.DataFrame,
    *,
    specs: list[EmbeddingInputSpec],
    source_index: int,
) -> np.ndarray:
    if not specs:
        return np.zeros((0, 0), dtype=np.float32)

    lag0_cache: dict[str, np.ndarray] = {}
    row_values: list[np.ndarray] = []
    for spec in specs:
        if spec.lag0_column not in lag0_cache:
            lag0_cache[spec.lag0_column] = np.stack(
                [_parse_embedding_cell(v) for v in df[spec.lag0_column].tolist()],
                axis=0,
            ).astype(np.float32, copy=False)
        raw_index = source_index - spec.lag
        if raw_index < 0 or raw_index >= len(df):
            raise ValueError(
                f"Batch mode cannot synthesize future text feature {spec.input_column!r} for source index "
                f"{source_index}: required raw row {raw_index} is unavailable."
            )
        row_values.append(lag0_cache[spec.lag0_column][raw_index])
    return np.stack(row_values, axis=0).astype(np.float32, copy=False)


def build_fixed_horizon_future_queries(
    *,
    df: pd.DataFrame,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    text: np.ndarray | None,
    timestamps: pd.Series,
    numeric_features: list[str],
    calendar_frequency: str | None,
    seasonality_k: int,
    seasonality_L: int | None,
    horizon: int,
    embedding_lags: list[int] | None = None,
    embedding_columns: list[str] | None = None,
    embedding_column_template: str | None = None,
) -> BatchQueryBundle:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    n_rows = len(y_raw)
    if n_rows != len(X_raw) or len(timestamps) != n_rows or (text is not None and len(text) != n_rows):
        raise ValueError("X_raw, y_raw, text, and timestamps must describe the same number of source rows.")
    if horizon > n_rows:
        raise ValueError(
            f"Batch horizon {horizon} is too large for {n_rows} observed rows. "
            "At least horizon rows are required."
        )

    source_features, seasonality_by_observed_row = build_source_feature_block(
        X_raw,
        y_raw,
        seasonality_k=seasonality_k,
        seasonality_L=seasonality_L,
    )
    source_indices = np.arange(n_rows - horizon + 1, n_rows + 1, dtype=np.int64)
    target_timestamps = build_future_timestamps(
        timestamps,
        horizon=horizon,
        calendar_frequency=calendar_frequency,
    )
    frequency = calendar_frequency or infer_calendar_frequency(pd.Series(timestamps))
    target_calendar = generate_calendar_features(target_timestamps, frequency).astype(np.float32, copy=False)

    source_timestamp_values: list[pd.Timestamp] = []
    source_core_rows: list[np.ndarray] = []
    text_rows: list[np.ndarray] = []

    embedding_specs = resolve_embedding_input_specs(
        embedding_lags=list(embedding_lags or []),
        embedding_columns=embedding_columns,
        embedding_column_template=embedding_column_template,
        df=df,
    )
    empty_text = _empty_text_like(text)

    for source_index in source_indices:
        if source_index < n_rows:
            source_timestamp_values.append(pd.Timestamp(pd.to_datetime(timestamps.iloc[source_index])))
            source_core_rows.append(source_features[source_index].astype(np.float32, copy=False))
            if text is not None:
                text_rows.append(text[source_index].astype(np.float32, copy=False))
            continue

        if source_index != n_rows:
            raise ValueError(
                "Batch mode only supports synthesizing at most one future source row beyond the observed range."
            )

        synthetic_numeric = _build_synthetic_numeric_row(
            df,
            numeric_features=numeric_features,
            source_index=source_index,
        )
        if seasonality_by_observed_row.shape[1] > 0:
            synthetic_core = np.concatenate(
                [synthetic_numeric, seasonality_by_observed_row[-1].astype(np.float32, copy=False)],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            synthetic_core = synthetic_numeric
        source_core_rows.append(synthetic_core)
        source_timestamp_values.append(pd.Timestamp(target_timestamps.iloc[0]))
        if text is not None:
            if empty_text is not None and text.shape[1] == 0:
                text_rows.append(np.zeros(text.shape[1:], dtype=np.float32))
            else:
                text_rows.append(
                    _build_synthetic_text_row(
                        df,
                        specs=embedding_specs,
                        source_index=source_index,
                    ).astype(np.float32, copy=False)
                )

    X_query_raw = np.concatenate(
        [
            np.stack(source_core_rows, axis=0).astype(np.float32, copy=False),
            target_calendar,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    if text is None:
        text_query = None
    else:
        text_query = np.stack(text_rows, axis=0).astype(np.float32, copy=False)
    return BatchQueryBundle(
        batch_horizon=horizon,
        source_indices=source_indices,
        source_timestamps=pd.Series(source_timestamp_values),
        target_timestamps=target_timestamps,
        X_query_raw=X_query_raw,
        text_query=text_query,
    )


def _trim_support_tail(
    X_support: np.ndarray,
    y_support: np.ndarray,
    *,
    max_context: int | None,
    text_support: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if max_context is None or len(y_support) <= max_context:
        return X_support, y_support, text_support
    start = len(y_support) - max_context
    trimmed_text = None if text_support is None else text_support[start:]
    return X_support[start:], y_support[start:], trimmed_text


def predict_tabdpt_batch(
    reg,
    *,
    X_support_proc: np.ndarray,
    y_support: np.ndarray,
    X_query_proc: np.ndarray,
    max_context: int | None,
    use_text: bool,
    text_support: np.ndarray | None = None,
    text_query: np.ndarray | None = None,
) -> np.ndarray:
    X_support_proc, y_support, text_support = _trim_support_tail(
        X_support_proc,
        y_support,
        max_context=max_context,
        text_support=text_support,
    )
    if use_text and (text_support is None or text_query is None):
        raise ValueError("Text-enabled batch prediction requires support and query text arrays.")

    reg.model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_support_proc, dtype=torch.float32, device=reg.device).unsqueeze(0)
        X_train_tensor = pad_x(X_train_tensor, reg.max_features)
        X_query_tensor = torch.tensor(X_query_proc, dtype=torch.float32, device=reg.device).unsqueeze(0)
        X_query_tensor = pad_x(X_query_tensor, reg.max_features)
        y_support_tensor = torch.tensor(y_support, dtype=torch.float32, device=reg.device).unsqueeze(0)

        text_train_b = text_test_b = None
        if use_text:
            text_train_b, text_test_b = reg.text_embeddings_batched(text_support, text_query)

        pred = reg.model(
            x_src=torch.cat([X_train_tensor, X_query_tensor], dim=1),
            y_src=y_support_tensor.unsqueeze(-1),
            task="reg",
            text_train=text_train_b,
            text_test=text_test_b,
        )
        pred, _, _ = pred
    return pred.squeeze(0).squeeze(-1).detach().cpu().numpy().astype(np.float32, copy=False)


def predict_estimator_batch(
    estimator,
    *,
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    max_context: int | None,
) -> np.ndarray:
    X_support, y_support, _ = _trim_support_tail(
        X_support,
        y_support,
        max_context=max_context,
        text_support=None,
    )
    estimator.fit(X_support, y_support)
    return np.asarray(estimator.predict(X_query), dtype=np.float32).reshape(-1)


def evaluate_batch_slot_metrics(
    *,
    batch_horizon: int,
    X_history: np.ndarray,
    y_history: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    y_eval_real: np.ndarray,
    max_context: int | None,
    predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None], np.ndarray],
    inverse_transform_fn: Callable[[np.ndarray], np.ndarray],
    text_history: np.ndarray | None = None,
    text_eval: np.ndarray | None = None,
) -> list[BatchSlotMetrics]:
    if batch_horizon <= 0:
        raise ValueError("batch_horizon must be positive.")

    slot_preds_normalized: list[list[float]] = [[] for _ in range(batch_horizon)]
    slot_targets_normalized: list[list[float]] = [[] for _ in range(batch_horizon)]
    slot_targets_real: list[list[float]] = [[] for _ in range(batch_horizon)]

    max_start = len(y_eval) - batch_horizon + 1
    for start in range(max(0, max_start)):
        X_support = np.concatenate((X_history, X_eval[:start]), axis=0)
        y_support = np.concatenate((y_history, y_eval[:start]), axis=0)
        text_support = None
        if text_history is not None and text_eval is not None:
            text_support = np.concatenate((text_history, text_eval[:start]), axis=0)

        X_query = X_eval[start:start + batch_horizon]
        text_query = None if text_eval is None else text_eval[start:start + batch_horizon]
        preds_normalized = predict_fn(X_support, y_support, text_support, X_query, text_query)
        if len(preds_normalized) != batch_horizon:
            raise ValueError(
                f"Batch predictor returned {len(preds_normalized)} predictions for horizon {batch_horizon}."
            )
        for slot in range(batch_horizon):
            slot_preds_normalized[slot].append(float(preds_normalized[slot]))
            slot_targets_normalized[slot].append(float(y_eval[start + slot]))
            slot_targets_real[slot].append(float(y_eval_real[start + slot]))

    summaries: list[BatchSlotMetrics] = []
    for slot in range(batch_horizon):
        if not slot_preds_normalized[slot]:
            nan_triplet = (float("nan"), float("nan"), float("nan"))
            summaries.append(
                BatchSlotMetrics(
                    slot=slot + 1,
                    count=0,
                    normalized=nan_triplet,
                    real=nan_triplet,
                )
            )
            continue
        preds_normalized = np.asarray(slot_preds_normalized[slot], dtype=np.float32)
        targets_normalized = np.asarray(slot_targets_normalized[slot], dtype=np.float32)
        targets_real = np.asarray(slot_targets_real[slot], dtype=np.float32)
        preds_real = inverse_transform_fn(preds_normalized)
        summaries.append(
            BatchSlotMetrics(
                slot=slot + 1,
                count=len(preds_normalized),
                normalized=_compute_metrics(targets_normalized, preds_normalized),
                real=_compute_metrics(targets_real, preds_real),
            )
        )
    return summaries


def batch_forecast_records(
    *,
    predictions_real: np.ndarray,
    query_bundle: BatchQueryBundle,
    model_name: str,
) -> list[BatchForecastRecord]:
    if len(predictions_real) != query_bundle.batch_horizon:
        raise ValueError(
            f"Expected {query_bundle.batch_horizon} predictions, got {len(predictions_real)}."
        )
    records: list[BatchForecastRecord] = []
    for slot, prediction in enumerate(predictions_real, start=1):
        records.append(
            BatchForecastRecord(
                slot=slot,
                source_index=int(query_bundle.source_indices[slot - 1]) + 1,
                source_timestamp=str(pd.Timestamp(query_bundle.source_timestamps.iloc[slot - 1])),
                target_timestamp=str(pd.Timestamp(query_bundle.target_timestamps.iloc[slot - 1])),
                prediction_real=float(prediction),
                model_name=model_name,
            )
        )
    return records


def records_to_csv_rows(
    *,
    forecast_records: list[BatchForecastRecord],
    slot_metrics: list[BatchSlotMetrics],
    split_name: str,
    overall_normalized: MetricTriplet | None = None,
    overall_real: MetricTriplet | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if overall_normalized is not None and overall_real is not None:
        rows.append(
            {
                "record_type": "overall_metric",
                "split": split_name,
                "slot": "",
                "count": "",
                "mae_normalized": overall_normalized[0],
                "rmse_normalized": overall_normalized[1],
                "mape_normalized": overall_normalized[2],
                "mae_real": overall_real[0],
                "rmse_real": overall_real[1],
                "mape_real": overall_real[2],
            }
        )
    for metric in slot_metrics:
        rows.append(
            {
                "record_type": "slot_metric",
                "split": split_name,
                "slot": metric.slot,
                "count": metric.count,
                "mae_normalized": metric.normalized[0],
                "rmse_normalized": metric.normalized[1],
                "mape_normalized": metric.normalized[2],
                "mae_real": metric.real[0],
                "rmse_real": metric.real[1],
                "mape_real": metric.real[2],
            }
        )
    for record in forecast_records:
        row = asdict(record)
        row["record_type"] = "forecast"
        rows.append(row)
    return rows
