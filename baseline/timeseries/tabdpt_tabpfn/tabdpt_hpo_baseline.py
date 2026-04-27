"""HPO-aware TabDPT baseline runner without text attention."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fine_tuning.eval_fine_tune import _format_dual_metrics, evaluate_rolling  # noqa: E402
from fine_tuning.fine_tune_configs import DataConfig, load_fine_tune_config  # noqa: E402
from fine_tuning.load_dataset import (  # noqa: E402
    _build_source_target_rows,
    load_tabular_text_dataset_with_timestamps,
    select_direct_horizon_targets,
    validate_direct_mode_numeric_features,
)
from fine_tuning.random_search_configs import load_random_search_config  # noqa: E402
from fine_tuning.split_ts import time_split  # noqa: E402
from tabdpt import TabDPTRegressor  # noqa: E402
from tabdpt.feature_build import generate_calendar_features, generate_causal_seasonality_features  # noqa: E402


MetricTriplet = tuple[float, float, float]
DualMetricTriplet = tuple[MetricTriplet, MetricTriplet]


@dataclass(frozen=True)
class SummaryMetrics:
    mae: float
    rmse: float
    mape: float
    real_mae: float
    real_rmse: float
    real_mape: float


@dataclass(frozen=True)
class HorizonResult:
    horizon: int
    val: SummaryMetrics
    test: SummaryMetrics


@dataclass(frozen=True)
class MaxContextResult:
    max_context: int | None
    val: SummaryMetrics
    test: SummaryMetrics
    per_horizon: list[HorizonResult]


class _TeeWriter:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@dataclass(frozen=True)
class FineTuneDataSplits:
    X_context: np.ndarray
    y_context: np.ndarray
    text_context: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    text_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    text_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    text_test: np.ndarray
    y_level_context: np.ndarray | None = None
    y_level_train: np.ndarray | None = None
    y_level_val: np.ndarray | None = None
    y_level_test: np.ndarray | None = None
    ts_context: np.ndarray | None = None
    ts_train: np.ndarray | None = None
    ts_val: np.ndarray | None = None
    ts_test: np.ndarray | None = None
    calendar_frequency: str | None = None
    seasonality_k: int = 3
    seasonality_L: int | None = None


def _summary_metrics_from_dual(triplet: DualMetricTriplet) -> SummaryMetrics:
    normalized, real = triplet
    return SummaryMetrics(
        mae=float(normalized[0]),
        rmse=float(normalized[1]),
        mape=float(normalized[2]),
        real_mae=float(real[0]),
        real_rmse=float(real[1]),
        real_mape=float(real[2]),
    )


def _mean_summary_metrics(values: list[SummaryMetrics]) -> SummaryMetrics:
    return SummaryMetrics(
        mae=float(np.mean([item.mae for item in values])),
        rmse=float(np.mean([item.rmse for item in values])),
        mape=float(np.mean([item.mape for item in values])),
        real_mae=float(np.mean([item.real_mae for item in values])),
        real_rmse=float(np.mean([item.real_rmse for item in values])),
        real_mape=float(np.mean([item.real_mape for item in values])),
    )


def _serialize_summary_metrics(metrics: SummaryMetrics) -> dict[str, float]:
    return {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "mape": metrics.mape,
        "real_mae": metrics.real_mae,
        "real_rmse": metrics.real_rmse,
        "real_mape": metrics.real_mape,
    }


def _resolve_hpo_config_path(config_path: str) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (SCRIPT_DIR / path).resolve()


def _reset_compiler_state() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "reset"):
        dynamo.reset()


def _make_tabdpt(run_cfg: DataConfig) -> TabDPTRegressor:
    device = run_cfg.model.device
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabDPTRegressor(
        normalizer=None,
        device=device,
        model_weight_path=run_cfg.model.model_weight_path,
        use_flash=run_cfg.model.use_flash,
        compile=run_cfg.model.compile_model,
    )


def _transform_target_split(scaler: StandardScaler, y: np.ndarray) -> np.ndarray:
    return scaler.transform(y.reshape(-1, 1).astype(np.float64, copy=False)).reshape(-1).astype(np.float32, copy=False)


def _inverse_transform_targets(y: np.ndarray, target_scaler: StandardScaler | None) -> np.ndarray:
    if target_scaler is None:
        return y.astype(np.float32, copy=False)
    return target_scaler.inverse_transform(y.reshape(-1, 1).astype(np.float64, copy=False)).reshape(-1).astype(
        np.float32,
        copy=False,
    )


def normalize_fine_tune_splits(splits: FineTuneDataSplits) -> tuple[FineTuneDataSplits, StandardScaler]:
    X_fit = np.concatenate((splits.X_context, splits.X_train), axis=0)
    y_fit = np.concatenate((splits.y_context, splits.y_train), axis=0)

    X_context = splits.X_context.astype(np.float64, copy=True)
    X_train = splits.X_train.astype(np.float64, copy=True)
    X_val = splits.X_val.astype(np.float64, copy=True)
    X_test = splits.X_test.astype(np.float64, copy=True)

    for col_idx in range(X_fit.shape[1]):
        feature_scaler = StandardScaler()
        fit_column = X_fit[:, [col_idx]].astype(np.float64, copy=False)
        feature_scaler.fit(fit_column)
        X_context[:, [col_idx]] = feature_scaler.transform(splits.X_context[:, [col_idx]].astype(np.float64, copy=False))
        X_train[:, [col_idx]] = feature_scaler.transform(splits.X_train[:, [col_idx]].astype(np.float64, copy=False))
        X_val[:, [col_idx]] = feature_scaler.transform(splits.X_val[:, [col_idx]].astype(np.float64, copy=False))
        X_test[:, [col_idx]] = feature_scaler.transform(splits.X_test[:, [col_idx]].astype(np.float64, copy=False))

    target_scaler = StandardScaler()
    target_scaler.fit(y_fit.reshape(-1, 1).astype(np.float64, copy=False))

    return (
        FineTuneDataSplits(
            X_context=X_context.astype(np.float32, copy=False),
            y_context=_transform_target_split(target_scaler, splits.y_context),
            text_context=splits.text_context,
            X_train=X_train.astype(np.float32, copy=False),
            y_train=_transform_target_split(target_scaler, splits.y_train),
            text_train=splits.text_train,
            X_val=X_val.astype(np.float32, copy=False),
            y_val=_transform_target_split(target_scaler, splits.y_val),
            text_val=splits.text_val,
            X_test=X_test.astype(np.float32, copy=False),
            y_test=_transform_target_split(target_scaler, splits.y_test),
            text_test=splits.text_test,
            y_level_context=splits.y_level_context,
            y_level_train=splits.y_level_train,
            y_level_val=splits.y_level_val,
            y_level_test=splits.y_level_test,
            ts_context=splits.ts_context,
            ts_train=splits.ts_train,
            ts_val=splits.ts_val,
            ts_test=splits.ts_test,
            calendar_frequency=splits.calendar_frequency,
            seasonality_k=splits.seasonality_k,
            seasonality_L=splits.seasonality_L,
        ),
        target_scaler,
    )


def maybe_normalize_fine_tune_splits(
    splits: FineTuneDataSplits,
    *,
    normalize: bool,
) -> tuple[FineTuneDataSplits, StandardScaler | None]:
    if not normalize:
        return splits, None
    return normalize_fine_tune_splits(splits)


def preprocess_features(
    reg: TabDPTRegressor,
    X: np.ndarray,
    *,
    reduction_mode: str | None,
    reduction_payload: np.ndarray | None,
) -> np.ndarray:
    X_proc = X
    if reg.missing_indicators:
        inds = np.isnan(X_proc)[:, reg.has_missing_indicator].astype(float)
        X_proc = np.hstack((X_proc, inds))
    X_proc = reg.imputer.transform(X_proc)
    if reg.scaler:
        X_proc = reg.scaler.transform(X_proc)
        if reg.normalizer == "quantile-uniform":
            X_proc = 2 * X_proc - 1

    if reduction_mode == "pca":
        if reduction_payload is None:
            raise ValueError("PCA reduction requested without a payload.")
        X_proc = X_proc @ reduction_payload
    elif reduction_mode == "subsample":
        if reduction_payload is None:
            raise ValueError("Subsample reduction requested without a payload.")
        X_proc = X_proc[:, reduction_payload][:, :reg.max_features]

    return X_proc.astype(np.float32, copy=False)


def _apply_target_differencing_mode(
    *,
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
    timestamps: np.ndarray | None,
    numeric_features: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    if len(y) < 2:
        raise ValueError("target_differencing mode requires at least 2 rows.")

    y_diff = np.diff(y.astype(np.float32, copy=False))
    y_level = y[1:].astype(np.float32, copy=False)
    X_diff = X[1:].astype(np.float32, copy=True)
    text_diff = text[1:]
    timestamps_diff = None if timestamps is None else timestamps[1:]

    target_lag_regexes = (
        re.compile(rf"^{re.escape(target_column)}_lag(\d+)$"),
        re.compile(r"^target_diff_lag(\d+)$"),
    )
    target_lag_columns: list[tuple[int, int]] = []
    for col_idx, feature_name in enumerate(numeric_features):
        match = None
        for regex in target_lag_regexes:
            match = regex.fullmatch(feature_name)
            if match is not None:
                break
        if match is None:
            continue
        lag = int(match.group(1))
        target_lag_columns.append((col_idx, lag))

    if not target_lag_columns:
        return X_diff, y_diff.astype(np.float32, copy=False), text_diff, timestamps_diff, y_level

    first_valid_idx = max(lag for _, lag in target_lag_columns)
    if first_valid_idx >= len(y_diff):
        raise ValueError("target_differencing mode removed all rows after applying target lag requirements.")

    for col_idx, lag in target_lag_columns:
        X_diff[first_valid_idx:, col_idx] = y_diff[first_valid_idx - lag:len(y_diff) - lag]

    return (
        X_diff[first_valid_idx:],
        y_diff[first_valid_idx:].astype(np.float32, copy=False),
        text_diff[first_valid_idx:],
        None if timestamps_diff is None else timestamps_diff[first_valid_idx:],
        y_level[first_valid_idx:].astype(np.float32, copy=False),
    )


def load_fine_tune_arrays(run_cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    validate_direct_mode_numeric_features(run_cfg.numeric_features)
    X, y, text, timestamps = load_tabular_text_dataset_with_timestamps(
        path=run_cfg.data_path,
        date_column=run_cfg.date_column,
        numeric_features=run_cfg.numeric_features,
        target_column=run_cfg.target_column,
        embedding_lags=run_cfg.embedding_lags,
        embedding_columns=run_cfg.embedding_columns,
        embedding_column_template=run_cfg.embedding_column_template,
        max_rows=run_cfg.max_rows,
    )
    if X.shape[1] == 0:
        X = np.zeros((X.shape[0], 1), dtype=np.float32)
    timestamps_array = None if timestamps is None else timestamps.to_numpy()
    y_level_array: np.ndarray | None = None
    if run_cfg.target_mode == "target_differencing":
        X, y, text, timestamps_array, y_level_array = _apply_target_differencing_mode(
            X=X,
            y=y,
            text=text,
            timestamps=timestamps_array,
            numeric_features=run_cfg.numeric_features,
            target_column=run_cfg.target_column,
        )
    return X, y, text, timestamps_array, y_level_array


def split_fine_tune_data(
    run_cfg: DataConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
    timestamps: np.ndarray | None,
    y_level: np.ndarray | None = None,
) -> FineTuneDataSplits:
    split_values = time_split(
        X,
        y,
        text,
        context_ratio=run_cfg.context_ratio,
        train_ratio=run_cfg.train_ratio,
        val_ratio=run_cfg.val_ratio,
        test_ratio=run_cfg.test_ratio,
    )
    ts_context = ts_train = ts_val = ts_test = None
    if timestamps is not None:
        n_context = len(split_values[1])
        n_train = len(split_values[4])
        n_val = len(split_values[7])
        ts_context = timestamps[:n_context]
        ts_train = timestamps[n_context:n_context + n_train]
        ts_val = timestamps[n_context + n_train:n_context + n_train + n_val]
        ts_test = timestamps[n_context + n_train + n_val:]
    y_level_context = y_level_train = y_level_val = y_level_test = None
    if y_level is not None:
        n_context = len(split_values[1])
        n_train = len(split_values[4])
        n_val = len(split_values[7])
        y_level_context = y_level[:n_context]
        y_level_train = y_level[n_context:n_context + n_train]
        y_level_val = y_level[n_context + n_train:n_context + n_train + n_val]
        y_level_test = y_level[n_context + n_train + n_val:]
    return FineTuneDataSplits(
        X_context=split_values[0],
        y_context=split_values[1],
        text_context=split_values[2],
        X_train=split_values[3],
        y_train=split_values[4],
        text_train=split_values[5],
        X_val=split_values[6],
        y_val=split_values[7],
        text_val=split_values[8],
        X_test=split_values[9],
        y_test=split_values[10],
        text_test=split_values[11],
        y_level_context=y_level_context,
        y_level_train=y_level_train,
        y_level_val=y_level_val,
        y_level_test=y_level_test,
        ts_context=ts_context,
        ts_train=ts_train,
        ts_val=ts_val,
        ts_test=ts_test,
        calendar_frequency=run_cfg.calendar_frequency,
        seasonality_k=run_cfg.seasonality_k,
        seasonality_L=run_cfg.seasonality_L,
    )


def load_and_split_fine_tune_data(run_cfg: DataConfig) -> FineTuneDataSplits:
    X, y, text, timestamps, y_level = load_fine_tune_arrays(run_cfg)
    return split_fine_tune_data(run_cfg, X=X, y=y, text=text, timestamps=timestamps, y_level=y_level)


def prediction_horizons(run_cfg: DataConfig) -> range:
    return range(1, run_cfg.prediction_window + 1)


def _build_history_and_eval_split(
    splits: FineTuneDataSplits,
    *,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if split_name == "val":
        return (
            np.concatenate((splits.X_context, splits.X_train), axis=0),
            np.concatenate((splits.y_context, splits.y_train), axis=0),
            np.concatenate((splits.text_context, splits.text_train), axis=0),
            splits.X_val,
            splits.y_val,
            splits.text_val,
        )
    if split_name == "test":
        return (
            np.concatenate((splits.X_context, splits.X_train, splits.X_val), axis=0),
            np.concatenate((splits.y_context, splits.y_train, splits.y_val), axis=0),
            np.concatenate((splits.text_context, splits.text_train, splits.text_val), axis=0),
            splits.X_test,
            splits.y_test,
            splits.text_test,
        )
    raise ValueError(f"Unsupported split_name: {split_name!r}. Expected 'val' or 'test'.")


def _level_eval_context(
    raw_splits: FineTuneDataSplits,
    *,
    split_name: str,
) -> tuple[np.ndarray | None, float | None]:
    if (
        raw_splits.y_level_context is None
        or raw_splits.y_level_train is None
        or raw_splits.y_level_val is None
        or raw_splits.y_level_test is None
    ):
        return None, None

    if split_name == "val":
        y_level_hist = np.concatenate((raw_splits.y_level_context, raw_splits.y_level_train), axis=0)
        y_level_eval = raw_splits.y_level_val
    elif split_name == "test":
        y_level_hist = np.concatenate(
            (raw_splits.y_level_context, raw_splits.y_level_train, raw_splits.y_level_val),
            axis=0,
        )
        y_level_eval = raw_splits.y_level_test
    else:
        raise ValueError(f"Unsupported split_name: {split_name!r}. Expected 'val' or 'test'.")
    if len(y_level_hist) == 0 or len(y_level_eval) == 0:
        return None, None
    return y_level_hist, float(y_level_hist[-1])


def _compute_split_ratios(data_splits: FineTuneDataSplits) -> tuple[float, float, float, float]:
    total = (
        len(data_splits.y_context)
        + len(data_splits.y_train)
        + len(data_splits.y_val)
        + len(data_splits.y_test)
    )
    return (
        len(data_splits.y_context) / total,
        len(data_splits.y_train) / total,
        len(data_splits.y_val) / total,
        len(data_splits.y_test) / total,
    )


def _build_optional_causal_splits(
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
    calendar_frequency: str | None,
    seasonality_k: int,
    seasonality_L: int | None,
    include_calendar_features: bool,
) -> tuple[np.ndarray, ...]:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if len(X) != len(y) or len(X) != len(text) or len(X) != len(timestamps):
        raise ValueError("X, y, text, and timestamps must have equal length.")

    n = len(y)
    n_context = int(n * context_ratio)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_context - n_train - n_val
    if min(n_context, n_train, n_val, n_test) <= 0:
        raise ValueError(
            f"Invalid split sizes: n={n}, context={n_context}, train={n_train}, val={n_val}, test={n_test}"
        )

    timestamps = pd.Series(pd.to_datetime(timestamps), copy=False).reset_index(drop=True)
    target_calendar = (
        generate_calendar_features(timestamps, calendar_frequency).astype(np.float32, copy=False)
        if include_calendar_features
        else np.zeros((n, 0), dtype=np.float32)
    )
    seasonality_by_observed_row = generate_causal_seasonality_features(
        y,
        k=seasonality_k,
        L=seasonality_L,
    ).astype(np.float32, copy=False)
    if seasonality_by_observed_row.shape[1] > 0:
        leading = np.full((1, seasonality_by_observed_row.shape[1]), np.nan, dtype=np.float32)
        source_seasonality = np.concatenate([leading, seasonality_by_observed_row[:-1]], axis=0).astype(
            np.float32,
            copy=False,
        )
        X_source = np.concatenate([X.astype(np.float32, copy=False), source_seasonality], axis=1).astype(
            np.float32,
            copy=False,
        )
    else:
        X_source = X.astype(np.float32, copy=False)

    train_end = n_context + n_train
    val_end = train_end + n_val
    context_sources = np.arange(0, n_context - horizon + 1, dtype=np.int64)
    train_sources = np.arange(n_context, train_end - horizon + 1, dtype=np.int64)
    val_sources = np.arange(train_end, val_end - horizon + 1, dtype=np.int64)
    test_sources = np.arange(val_end, n - horizon + 1, dtype=np.int64)
    if min(len(context_sources), len(train_sources), len(val_sources), len(test_sources)) <= 0:
        raise ValueError(
            "Horizon trimming produced an empty split: "
            f"n={n}, context={n_context}, train={n_train}, val={n_val}, test={n_test}, horizon={horizon}"
        )

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


def _select_horizon_splits(
    data_splits: FineTuneDataSplits,
    *,
    horizon: int,
    include_calendar_features: bool,
    include_seasonal_features: bool,
) -> FineTuneDataSplits:
    if (
        data_splits.ts_context is not None
        and data_splits.ts_train is not None
        and data_splits.ts_val is not None
        and data_splits.ts_test is not None
    ):
        X_full = np.concatenate(
            (data_splits.X_context, data_splits.X_train, data_splits.X_val, data_splits.X_test),
            axis=0,
        )
        y_full = np.concatenate(
            (data_splits.y_context, data_splits.y_train, data_splits.y_val, data_splits.y_test),
            axis=0,
        )
        text_full = np.concatenate(
            (data_splits.text_context, data_splits.text_train, data_splits.text_val, data_splits.text_test),
            axis=0,
        )
        ts_full = pd.Series(
            np.concatenate((data_splits.ts_context, data_splits.ts_train, data_splits.ts_val, data_splits.ts_test))
        )
        context_ratio, train_ratio, val_ratio, test_ratio = _compute_split_ratios(data_splits)
        seasonality_k = data_splits.seasonality_k if include_seasonal_features else 0
        split_values = _build_optional_causal_splits(
            X_full,
            y_full,
            text_full,
            ts_full,
            context_ratio=context_ratio,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            horizon=horizon,
            calendar_frequency=data_splits.calendar_frequency,
            seasonality_k=seasonality_k,
            seasonality_L=data_splits.seasonality_L,
            include_calendar_features=include_calendar_features,
        )
        y_level_context = y_level_train = y_level_val = y_level_test = None
        if (
            data_splits.y_level_context is not None
            and data_splits.y_level_train is not None
            and data_splits.y_level_val is not None
            and data_splits.y_level_test is not None
        ):
            level_split_values = _build_optional_causal_splits(
                X_full,
                np.concatenate(
                    (
                        data_splits.y_level_context,
                        data_splits.y_level_train,
                        data_splits.y_level_val,
                        data_splits.y_level_test,
                    ),
                    axis=0,
                ),
                text_full,
                ts_full,
                context_ratio=context_ratio,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                horizon=horizon,
                calendar_frequency=data_splits.calendar_frequency,
                seasonality_k=seasonality_k,
                seasonality_L=data_splits.seasonality_L,
                include_calendar_features=include_calendar_features,
            )
            y_level_context = level_split_values[1]
            y_level_train = level_split_values[4]
            y_level_val = level_split_values[7]
            y_level_test = level_split_values[10]

        return FineTuneDataSplits(
            X_context=split_values[0],
            y_context=split_values[1],
            text_context=split_values[2],
            X_train=split_values[3],
            y_train=split_values[4],
            text_train=split_values[5],
            X_val=split_values[6],
            y_val=split_values[7],
            text_val=split_values[8],
            X_test=split_values[9],
            y_test=split_values[10],
            text_test=split_values[11],
            y_level_context=y_level_context,
            y_level_train=y_level_train,
            y_level_val=y_level_val,
            y_level_test=y_level_test,
            calendar_frequency=data_splits.calendar_frequency if include_calendar_features else None,
            seasonality_k=seasonality_k,
            seasonality_L=data_splits.seasonality_L if include_seasonal_features else None,
        )

    X_context, y_context, text_context = select_direct_horizon_targets(
        data_splits.X_context, data_splits.y_context, data_splits.text_context, horizon=horizon
    )
    X_train, y_train, text_train = select_direct_horizon_targets(
        data_splits.X_train, data_splits.y_train, data_splits.text_train, horizon=horizon
    )
    X_val, y_val, text_val = select_direct_horizon_targets(
        data_splits.X_val, data_splits.y_val, data_splits.text_val, horizon=horizon
    )
    X_test, y_test, text_test = select_direct_horizon_targets(
        data_splits.X_test, data_splits.y_test, data_splits.text_test, horizon=horizon
    )
    y_level_context = y_level_train = y_level_val = y_level_test = None
    if (
        data_splits.y_level_context is not None
        and data_splits.y_level_train is not None
        and data_splits.y_level_val is not None
        and data_splits.y_level_test is not None
    ):
        _, y_level_context, _ = select_direct_horizon_targets(
            data_splits.X_context, data_splits.y_level_context, data_splits.text_context, horizon=horizon
        )
        _, y_level_train, _ = select_direct_horizon_targets(
            data_splits.X_train, data_splits.y_level_train, data_splits.text_train, horizon=horizon
        )
        _, y_level_val, _ = select_direct_horizon_targets(
            data_splits.X_val, data_splits.y_level_val, data_splits.text_val, horizon=horizon
        )
        _, y_level_test, _ = select_direct_horizon_targets(
            data_splits.X_test, data_splits.y_level_test, data_splits.text_test, horizon=horizon
        )
    return FineTuneDataSplits(
        X_context=X_context,
        y_context=y_context,
        text_context=text_context,
        X_train=X_train,
        y_train=y_train,
        text_train=text_train,
        X_val=X_val,
        y_val=y_val,
        text_val=text_val,
        X_test=X_test,
        y_test=y_test,
        text_test=text_test,
        y_level_context=y_level_context,
        y_level_train=y_level_train,
        y_level_val=y_level_val,
        y_level_test=y_level_test,
        calendar_frequency=data_splits.calendar_frequency if include_calendar_features else None,
        seasonality_k=data_splits.seasonality_k if include_seasonal_features else 0,
        seasonality_L=data_splits.seasonality_L if include_seasonal_features else None,
    )


def _run_horizon(
    run_cfg: DataConfig,
    data_splits: FineTuneDataSplits,
    *,
    horizon: int,
    max_context: int | None,
    include_calendar_features: bool,
    include_seasonal_features: bool,
    normalize: bool,
) -> HorizonResult:
    print(f"\n-- Horizon {horizon}/{run_cfg.prediction_window} | max_context={max_context} --")
    raw_horizon_splits = _select_horizon_splits(
        data_splits,
        horizon=horizon,
        include_calendar_features=include_calendar_features,
        include_seasonal_features=include_seasonal_features,
    )
    normalized_horizon_splits, target_scaler = maybe_normalize_fine_tune_splits(
        raw_horizon_splits,
        normalize=normalize,
    )
    processed_splits = FineTuneDataSplits(
        X_context=normalized_horizon_splits.X_context,
        y_context=normalized_horizon_splits.y_context,
        text_context=normalized_horizon_splits.text_context,
        X_train=normalized_horizon_splits.X_train,
        y_train=normalized_horizon_splits.y_train,
        text_train=normalized_horizon_splits.text_train,
        X_val=normalized_horizon_splits.X_val,
        y_val=normalized_horizon_splits.y_val,
        text_val=normalized_horizon_splits.text_val,
        X_test=normalized_horizon_splits.X_test,
        y_test=normalized_horizon_splits.y_test,
        text_test=normalized_horizon_splits.text_test,
        y_level_context=raw_horizon_splits.y_level_context,
        y_level_train=raw_horizon_splits.y_level_train,
        y_level_val=raw_horizon_splits.y_level_val,
        y_level_test=raw_horizon_splits.y_level_test,
    )

    _reset_compiler_state()
    reg = _make_tabdpt(run_cfg)
    reg.fit(processed_splits.X_context, processed_splits.y_context)

    feature_splits = FineTuneDataSplits(
        X_context=preprocess_features(reg, processed_splits.X_context, reduction_mode=None, reduction_payload=None),
        y_context=processed_splits.y_context,
        text_context=processed_splits.text_context,
        X_train=preprocess_features(reg, processed_splits.X_train, reduction_mode=None, reduction_payload=None),
        y_train=processed_splits.y_train,
        text_train=processed_splits.text_train,
        X_val=preprocess_features(reg, processed_splits.X_val, reduction_mode=None, reduction_payload=None),
        y_val=processed_splits.y_val,
        text_val=processed_splits.text_val,
        X_test=preprocess_features(reg, processed_splits.X_test, reduction_mode=None, reduction_payload=None),
        y_test=processed_splits.y_test,
        text_test=processed_splits.text_test,
        y_level_context=raw_horizon_splits.y_level_context,
        y_level_train=raw_horizon_splits.y_level_train,
        y_level_val=raw_horizon_splits.y_level_val,
        y_level_test=raw_horizon_splits.y_level_test,
    )

    def _evaluate_split(split_name: str) -> SummaryMetrics:
        X_hist, y_hist, text_hist, X_eval, y_eval, text_eval = _build_history_and_eval_split(
            feature_splits,
            split_name=split_name,
        )
        _, _, _, _, y_eval_real, _ = _build_history_and_eval_split(raw_horizon_splits, split_name=split_name)
        _, previous_level = _level_eval_context(raw_horizon_splits, split_name=split_name)
        y_eval_level = raw_horizon_splits.y_level_val if split_name == "val" else raw_horizon_splits.y_level_test
        result = evaluate_rolling(
            reg,
            X_context_proc=X_hist,
            y_context=y_hist,
            text_context=text_hist,
            X_eval_proc=X_eval,
            y_eval=y_eval,
            y_eval_real=y_eval_real,
            text_eval=text_eval,
            use_text=False,
            label=f"{split_name.title()} baseline (no text attn) h={horizon} max_context={max_context}",
            max_context=max_context,
            target_scaler=target_scaler,
            horizon=horizon,
            y_eval_level=y_eval_level,
            previous_level=previous_level,
        )
        return _summary_metrics_from_dual(result)

    return HorizonResult(
        horizon=horizon,
        val=_evaluate_split("val"),
        test=_evaluate_split("test"),
    )


def _run_max_context(
    run_cfg: DataConfig,
    data_splits: FineTuneDataSplits,
    *,
    max_context: int | None,
    include_calendar_features: bool,
    include_seasonal_features: bool,
    normalize: bool,
) -> MaxContextResult:
    print(f"\n== max_context={max_context} ==")
    per_horizon = [
        _run_horizon(
            run_cfg,
            data_splits,
            horizon=horizon,
            max_context=max_context,
            include_calendar_features=include_calendar_features,
            include_seasonal_features=include_seasonal_features,
            normalize=normalize,
        )
        for horizon in prediction_horizons(run_cfg)
    ]
    return MaxContextResult(
        max_context=max_context,
        val=_mean_summary_metrics([item.val for item in per_horizon]),
        test=_mean_summary_metrics([item.test for item in per_horizon]),
        per_horizon=per_horizon,
    )


def _write_summary_json(
    output_dir: Path,
    *,
    hpo_config_path: Path,
    base_config_path: str,
    base_run_cfg: DataConfig,
    base_dataset: str | None,
    include_calendar_features: bool,
    include_seasonal_features: bool,
    normalize: bool,
    results: list[MaxContextResult],
) -> None:
    payload = {
        "hpo_config": str(hpo_config_path),
        "base_config": base_config_path,
        "base_dataset": base_dataset,
        "prediction_window": base_run_cfg.prediction_window,
        "feature_flags": {
            "calendar_features_enabled": include_calendar_features,
            "seasonal_features_enabled": include_seasonal_features,
            "normalization_enabled": normalize,
            "calendar_frequency": base_run_cfg.calendar_frequency if include_calendar_features else None,
            "seasonality_k": base_run_cfg.seasonality_k if include_seasonal_features else 0,
            "seasonality_L": base_run_cfg.seasonality_L if include_seasonal_features else None,
        },
        "max_context_values": [item.max_context for item in results],
        "results_by_max_context": [
            {
                "max_context": item.max_context,
                "val": _serialize_summary_metrics(item.val),
                "test": _serialize_summary_metrics(item.test),
                "per_horizon": [
                    {
                        "horizon": horizon_result.horizon,
                        "val": _serialize_summary_metrics(horizon_result.val),
                        "test": _serialize_summary_metrics(horizon_result.test),
                    }
                    for horizon_result in item.per_horizon
                ],
            }
            for item in results
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _print_summary(results: list[MaxContextResult]) -> None:
    print("\nSummary")
    for item in results:
        print(f"max_context={item.max_context}")
        _format_dual_metrics(
            "val_no_text",
            (item.val.mae, item.val.rmse, item.val.mape),
            (item.val.real_mae, item.val.real_rmse, item.val.real_mape),
        )
        _format_dual_metrics(
            "test_no_text",
            (item.test.mae, item.test.rmse, item.test.mape),
            (item.test.real_mae, item.test.real_rmse, item.test.real_mape),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HPO-aware TabDPT baseline without text attention.")
    parser.add_argument("--config", required=True, help="Path to the HPO YAML config.")
    parser.add_argument("--dataset", default=None, help="Optional dataset override for multi-dataset fine-tune configs.")
    parser.add_argument("--output-root", default=None, help="Optional output root. Defaults to results/hpo.")
    parser.add_argument("--disable-calendar-features", action="store_true")
    parser.add_argument("--disable-seasonal-features", action="store_true")
    parser.add_argument("--disable-normalization", action="store_true")
    args = parser.parse_args()

    hpo_config_path = _resolve_hpo_config_path(args.config)
    random_search_cfg = load_random_search_config(str(hpo_config_path))
    dataset_name = args.dataset if args.dataset is not None else random_search_cfg.base_dataset
    base_run_cfg = load_fine_tune_config(random_search_cfg.base_config, dataset_name)
    data_splits = load_and_split_fine_tune_data(base_run_cfg)
    include_calendar_features = not args.disable_calendar_features
    include_seasonal_features = not args.disable_seasonal_features
    normalize = not args.disable_normalization
    max_context_values = (
        list(random_search_cfg.search_space.max_context)
        if random_search_cfg.search_space.max_context is not None
        else [base_run_cfg.tuning.max_context]
    )

    config_stem = hpo_config_path.stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_root) if args.output_root is not None else Path("results") / "hpo"
    output_dir = output_base / f"{config_stem}_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    log_path = output_dir / "summary.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee_writer = _TeeWriter(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee_writer), contextlib.redirect_stderr(tee_writer):
            print(
                "HPO baseline setup | "
                f"hpo_config={hpo_config_path} | "
                f"base_config={random_search_cfg.base_config} | "
                f"dataset={dataset_name} | "
                f"prediction_window={base_run_cfg.prediction_window}"
            )
            print(
                "Feature flags | "
                f"calendar_features_enabled={include_calendar_features} | "
                f"seasonal_features_enabled={include_seasonal_features} | "
                f"normalization_enabled={normalize}"
            )
            print(
                "Split sizes | "
                f"context={len(data_splits.y_context)} train={len(data_splits.y_train)} "
                f"val={len(data_splits.y_val)} test={len(data_splits.y_test)}"
            )
            print(f"max_context_values={max_context_values}")

            results = [
                _run_max_context(
                    base_run_cfg,
                    data_splits,
                    max_context=max_context,
                    include_calendar_features=include_calendar_features,
                    include_seasonal_features=include_seasonal_features,
                    normalize=normalize,
                )
                for max_context in max_context_values
            ]
            _print_summary(results)

    _write_summary_json(
        output_dir,
        hpo_config_path=hpo_config_path,
        base_config_path=random_search_cfg.base_config,
        base_run_cfg=base_run_cfg,
        base_dataset=dataset_name,
        include_calendar_features=include_calendar_features,
        include_seasonal_features=include_seasonal_features,
        normalize=normalize,
        results=results,
    )
    print(f"Saved summary.log to: {log_path}")
    print(f"Saved summary.json to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
