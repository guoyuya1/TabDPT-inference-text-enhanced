"""
Config-driven fine-tuning of TabDPT's text-mixing parameters in configured transformer layers.

This script expects a processed CSV with numeric features, a numeric target, and
text embedding columns. It loads one YAML config, splits the data
chronologically into context / train / val / test, fits the base TabDPT
regressor on the context split, and then fine-tunes only the text-mixing
parameters in the text-enhanced transformer layers:

- the per-head `alpha` gate logits
- the text attention projection used to score train/test text pairs

Fine-tuning uses rolling one-step prediction on the train split, selects the
best epoch by validation MAE with early stopping, keeps the top 3 validation-MAE
checkpoints, restores the best text-mixing weights, and prints final held-out
test metrics.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import schedulefree  # type: ignore
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from tabdpt import TabDPTRegressor
from tabdpt.model import _share_text_attention_modules
from tabdpt.utils import pad_x

try:
    from .eval_fine_tune import (
        _build_rolling_train_step,
        _format_metrics,
        _format_dual_metrics,
        evaluate_rolling,
        evaluate_rolling_pca,
        evaluate_rolling_truncate_text,
    )
    from .fine_tune_configs import DataConfig, TuningConfig, load_fine_tune_config
    from .load_dataset import (
        build_causal_fixed_origin_horizon_splits,
        load_tabular_text_dataset,
        load_tabular_text_dataset_with_timestamps,
        select_direct_horizon_targets,
        validate_direct_mode_numeric_features,
    )
    from .split_ts import time_split
except ImportError:
    from eval_fine_tune import (
        _build_rolling_train_step,
        _format_metrics,
        _format_dual_metrics,
        evaluate_rolling,
        evaluate_rolling_pca,
        evaluate_rolling_truncate_text,
    )
    from fine_tune_configs import DataConfig, TuningConfig, load_fine_tune_config
    from load_dataset import (
        build_causal_fixed_origin_horizon_splits,
        load_tabular_text_dataset,
        load_tabular_text_dataset_with_timestamps,
        select_direct_horizon_targets,
        validate_direct_mode_numeric_features,
    )
    from split_ts import time_split


TOP_VALIDATION_MAE_MODELS = 3
MetricTriplet = tuple[float, float, float]
DualMetricTriplet = tuple[MetricTriplet, MetricTriplet]


@dataclass(frozen=True)
class ValidationMaeCheckpoint:
    epoch: int
    val_mae: float
    state_dict: dict[str, torch.Tensor]


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
    ts_context: np.ndarray | None = None
    ts_train: np.ndarray | None = None
    ts_val: np.ndarray | None = None
    ts_test: np.ndarray | None = None
    calendar_frequency: str | None = None
    seasonality_k: int = 3
    seasonality_L: int | None = None


@dataclass(frozen=True)
class PreparedFineTuneTrial:
    reg: TabDPTRegressor
    splits: FineTuneDataSplits
    raw_splits: FineTuneDataSplits
    X_context_proc: np.ndarray
    X_train_proc: np.ndarray
    X_val_proc: np.ndarray
    X_test_proc: np.ndarray
    target_scaler: StandardScaler | None
    prediction_horizon: int


@dataclass(frozen=True)
class RollingMetrics:
    loss: float
    mae: float
    rmse: float
    mape: float
    real_mae: float
    real_rmse: float
    real_mape: float


@dataclass(frozen=True)
class FineTuneOutcome:
    top_validation_mae_checkpoints: list[ValidationMaeCheckpoint]
    best_epoch: int
    best_score: float


@dataclass(frozen=True)
class HorizonRunResult:
    horizon: int
    best_epoch: int
    tuned_no_text_test_normalized: MetricTriplet
    tuned_no_text_test_real: MetricTriplet
    tuned_text_test_normalized: MetricTriplet
    tuned_text_test_real: MetricTriplet


def _clone_state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[name] = value.detach().cpu().clone()
        else:
            cloned[name] = copy.deepcopy(value)
    return cloned


def _maybe_record_top_validation_mae_checkpoint(
    checkpoints: list[ValidationMaeCheckpoint],
    *,
    epoch: int,
    val_mae: float,
    model: torch.nn.Module,
    limit: int = TOP_VALIDATION_MAE_MODELS,
) -> None:
    candidate_key = (val_mae, epoch)
    if len(checkpoints) >= limit:
        worst_checkpoint = max(checkpoints, key=lambda checkpoint: (checkpoint.val_mae, checkpoint.epoch))
        if candidate_key >= (worst_checkpoint.val_mae, worst_checkpoint.epoch):
            return

    checkpoints.append(
        ValidationMaeCheckpoint(
            epoch=epoch,
            val_mae=val_mae,
            state_dict=_clone_state_dict_to_cpu(model.state_dict()),
        )
    )
    checkpoints.sort(key=lambda checkpoint: (checkpoint.val_mae, checkpoint.epoch))
    del checkpoints[limit:]


def load_tabdpt_regressor(
    *,
    device: str | None,
    model_weight_path: str | None,
    text_attn_layers: list[int],
    use_flash: bool,
    compile_model: bool,
) -> TabDPTRegressor:
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabDPTRegressor(
        # Numeric features are already standardized in normalize_fine_tune_splits.
        # Keep TabDPT preprocessing aligned with the baseline by disabling its
        # internal scaler here.
        normalizer=None,
        device=device,
        text_attn_layers=text_attn_layers,
        model_weight_path=model_weight_path,
        use_flash=use_flash,
        compile=compile_model,
    )


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

    return X_proc.astype(np.float32)


def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _transform_target_split(
    scaler: StandardScaler,
    y: np.ndarray,
) -> np.ndarray:
    return scaler.transform(y.reshape(-1, 1).astype(np.float64, copy=False)).reshape(-1).astype(np.float32, copy=False)


def inverse_transform_targets(
    y: np.ndarray,
    target_scaler: StandardScaler | None,
) -> np.ndarray:
    if target_scaler is None:
        return y.astype(np.float32, copy=False)
    return (
        target_scaler.inverse_transform(y.reshape(-1, 1).astype(np.float64, copy=False))
        .reshape(-1)
        .astype(np.float32, copy=False)
    )


def normalize_fine_tune_splits(
    splits: FineTuneDataSplits,
) -> tuple[FineTuneDataSplits, StandardScaler]:
    X_fit = np.concatenate((splits.X_context, splits.X_train), axis=0)
    if len(X_fit) == 0:
        raise ValueError("Cannot fit feature normalizer on an empty context+train split.")

    y_fit = np.concatenate((splits.y_context, splits.y_train), axis=0)
    if len(y_fit) == 0:
        raise ValueError("Cannot fit target normalizer on an empty context+train split.")

    X_context = splits.X_context.astype(np.float64, copy=True)
    X_train = splits.X_train.astype(np.float64, copy=True)
    X_val = splits.X_val.astype(np.float64, copy=True)
    X_test = splits.X_test.astype(np.float64, copy=True)

    for col_idx in range(X_fit.shape[1]):
        feature_scaler = StandardScaler()
        fit_column = X_fit[:, [col_idx]].astype(np.float64, copy=False)
        feature_scaler.fit(fit_column)
        X_context[:, [col_idx]] = feature_scaler.transform(
            splits.X_context[:, [col_idx]].astype(np.float64, copy=False)
        )
        X_train[:, [col_idx]] = feature_scaler.transform(
            splits.X_train[:, [col_idx]].astype(np.float64, copy=False)
        )
        X_val[:, [col_idx]] = feature_scaler.transform(
            splits.X_val[:, [col_idx]].astype(np.float64, copy=False)
        )
        X_test[:, [col_idx]] = feature_scaler.transform(
            splits.X_test[:, [col_idx]].astype(np.float64, copy=False)
        )

    target_scaler = StandardScaler()
    target_scaler.fit(y_fit.reshape(-1, 1).astype(np.float64, copy=False))

    normalized_splits = FineTuneDataSplits(
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
        ts_context=splits.ts_context,
        ts_train=splits.ts_train,
        ts_val=splits.ts_val,
        ts_test=splits.ts_test,
        calendar_frequency=splits.calendar_frequency,
        seasonality_k=splits.seasonality_k,
        seasonality_L=splits.seasonality_L,
    )
    return normalized_splits, target_scaler


def load_fine_tune_arrays(
    run_cfg: DataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
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
    return X, y, text, timestamps_array


def prediction_horizons(run_cfg: DataConfig) -> range:
    return range(1, run_cfg.prediction_window + 1)


def split_fine_tune_data(
    run_cfg: DataConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
    timestamps: np.ndarray | None,
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
        ts_context=ts_context,
        ts_train=ts_train,
        ts_val=ts_val,
        ts_test=ts_test,
        calendar_frequency=run_cfg.calendar_frequency,
        seasonality_k=run_cfg.seasonality_k,
        seasonality_L=run_cfg.seasonality_L,
    )


def load_and_split_fine_tune_data(run_cfg: DataConfig) -> FineTuneDataSplits:
    X, y, text, timestamps = load_fine_tune_arrays(run_cfg)
    return split_fine_tune_data(run_cfg, X=X, y=y, text=text, timestamps=timestamps)


def select_fine_tune_splits_for_horizon(
    data_splits: FineTuneDataSplits,
    *,
    horizon: int,
) -> FineTuneDataSplits:
    if (
        data_splits.ts_context is not None
        and data_splits.ts_train is not None
        and data_splits.ts_val is not None
        and data_splits.ts_test is not None
    ):
        split_values = build_causal_fixed_origin_horizon_splits(
            np.concatenate((data_splits.X_context, data_splits.X_train, data_splits.X_val, data_splits.X_test), axis=0),
            np.concatenate((data_splits.y_context, data_splits.y_train, data_splits.y_val, data_splits.y_test), axis=0),
            np.concatenate((data_splits.text_context, data_splits.text_train, data_splits.text_val, data_splits.text_test), axis=0),
            pd.Series(
                np.concatenate(
                    (data_splits.ts_context, data_splits.ts_train, data_splits.ts_val, data_splits.ts_test),
                    axis=0,
                )
            ),
            calendar_frequency=data_splits.calendar_frequency,
            context_ratio=len(data_splits.y_context)
            / (
                len(data_splits.y_context)
                + len(data_splits.y_train)
                + len(data_splits.y_val)
                + len(data_splits.y_test)
            ),
            train_ratio=len(data_splits.y_train)
            / (
                len(data_splits.y_context)
                + len(data_splits.y_train)
                + len(data_splits.y_val)
                + len(data_splits.y_test)
            ),
            val_ratio=len(data_splits.y_val)
            / (
                len(data_splits.y_context)
                + len(data_splits.y_train)
                + len(data_splits.y_val)
                + len(data_splits.y_test)
            ),
            test_ratio=len(data_splits.y_test)
            / (
                len(data_splits.y_context)
                + len(data_splits.y_train)
                + len(data_splits.y_val)
                + len(data_splits.y_test)
            ),
            horizon=horizon,
            seasonality_k=data_splits.seasonality_k,
            seasonality_L=data_splits.seasonality_L,
        )
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
            calendar_frequency=data_splits.calendar_frequency,
            seasonality_k=data_splits.seasonality_k,
            seasonality_L=data_splits.seasonality_L,
        )

    X_context, y_context, text_context = select_direct_horizon_targets(
        data_splits.X_context,
        data_splits.y_context,
        data_splits.text_context,
        horizon=horizon,
    )
    X_train, y_train, text_train = select_direct_horizon_targets(
        data_splits.X_train,
        data_splits.y_train,
        data_splits.text_train,
        horizon=horizon,
    )
    X_val, y_val, text_val = select_direct_horizon_targets(
        data_splits.X_val,
        data_splits.y_val,
        data_splits.text_val,
        horizon=horizon,
    )
    X_test, y_test, text_test = select_direct_horizon_targets(
        data_splits.X_test,
        data_splits.y_test,
        data_splits.text_test,
        horizon=horizon,
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
        calendar_frequency=data_splits.calendar_frequency,
        seasonality_k=data_splits.seasonality_k,
        seasonality_L=data_splits.seasonality_L,
    )


def prepare_fine_tune_trial(
    run_cfg: DataConfig,
    *,
    data_splits: FineTuneDataSplits | None = None,
    horizon: int = 1,
    pure_tabdpt: bool = False,
) -> PreparedFineTuneTrial:
    if data_splits is None:
        data_splits = load_and_split_fine_tune_data(run_cfg)
    raw_selected_splits = select_fine_tune_splits_for_horizon(data_splits, horizon=horizon)
    selected_splits, target_scaler = normalize_fine_tune_splits(raw_selected_splits)

    reg = load_tabdpt_regressor(
        device=run_cfg.model.device,
        model_weight_path=run_cfg.model.model_weight_path,
        text_attn_layers=None if pure_tabdpt else run_cfg.model.text_attn_layers,
        use_flash=run_cfg.model.use_flash,
        compile_model=run_cfg.model.compile_model,
    )
    # Keep pure TabDPT baselines aligned with baseline/timeseries/tabdpt_tabpfn:
    # do not inject fine-tuning attention regularization knobs (qk_norm/dropout)
    # when text attention is disabled.
    if not pure_tabdpt:
        _configure_attention_regularization(reg, tuning_cfg=run_cfg.tuning)
    reg.fit(selected_splits.X_context, selected_splits.y_context, selected_splits.text_context)

    reduction_mode = None
    reduction_payload = None
    return PreparedFineTuneTrial(
        reg=reg,
        splits=selected_splits,
        raw_splits=raw_selected_splits,
        X_context_proc=preprocess_features(
            reg,
            selected_splits.X_context,
            reduction_mode=reduction_mode,
            reduction_payload=reduction_payload,
        ),
        X_train_proc=preprocess_features(
            reg,
            selected_splits.X_train,
            reduction_mode=reduction_mode,
            reduction_payload=reduction_payload,
        ),
        X_val_proc=preprocess_features(
            reg,
            selected_splits.X_val,
            reduction_mode=reduction_mode,
            reduction_payload=reduction_payload,
        ),
        X_test_proc=preprocess_features(
            reg,
            selected_splits.X_test,
            reduction_mode=reduction_mode,
            reduction_payload=reduction_payload,
        ),
        target_scaler=target_scaler,
        prediction_horizon=horizon,
    )


def build_history_and_eval_split(
    X_context: np.ndarray,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_context: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    text_context: np.ndarray,
    text_train: np.ndarray,
    text_val: np.ndarray,
    text_test: np.ndarray,
    split_name: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the rolling history prefix and eval segment for train, val, or test."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    def _trim_history_tail(
        X_hist: np.ndarray,
        y_hist: np.ndarray,
        text_hist: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        trim = max(0, horizon - 1)
        if trim == 0:
            return X_hist, y_hist, text_hist
        if trim >= len(y_hist):
            return X_hist[:0], y_hist[:0], text_hist[:0]
        return X_hist[:-trim], y_hist[:-trim], text_hist[:-trim]

    if split_name == "train":
        return X_context, y_context, text_context, X_train, y_train, text_train
    if split_name == "val":
        X_hist, y_hist, text_hist = _trim_history_tail(
            np.concatenate((X_context, X_train), axis=0),
            np.concatenate((y_context, y_train), axis=0),
            np.concatenate((text_context, text_train), axis=0),
        )
        return X_hist, y_hist, text_hist, X_val, y_val, text_val
    if split_name == "test":
        X_hist, y_hist, text_hist = _trim_history_tail(
            np.concatenate((X_context, X_train, X_val), axis=0),
            np.concatenate((y_context, y_train, y_val), axis=0),
            np.concatenate((text_context, text_train, text_val), axis=0),
        )
        return X_hist, y_hist, text_hist, X_test, y_test, text_test
    raise ValueError(f"Unsupported split_name: {split_name!r}. Expected 'train', 'val', or 'test'.")


def _prepared_eval_inputs(
    prepared_trial: PreparedFineTuneTrial,
    *,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splits = prepared_trial.splits
    raw_splits = prepared_trial.raw_splits
    X_hist, y_hist, text_hist, X_eval, y_eval, text_eval = build_history_and_eval_split(
        prepared_trial.X_context_proc,
        prepared_trial.X_train_proc,
        prepared_trial.X_val_proc,
        prepared_trial.X_test_proc,
        splits.y_context,
        splits.y_train,
        splits.y_val,
        splits.y_test,
        splits.text_context,
        splits.text_train,
        splits.text_val,
        splits.text_test,
        split_name,
        prepared_trial.prediction_horizon,
    )
    if split_name == "train":
        y_eval_real = raw_splits.y_train
    elif split_name == "val":
        y_eval_real = raw_splits.y_val
    elif split_name == "test":
        y_eval_real = raw_splits.y_test
    else:
        raise ValueError(f"Unsupported split_name: {split_name!r}. Expected 'train', 'val', or 'test'.")
    return X_hist, y_hist, text_hist, X_eval, y_eval, y_eval_real, text_eval


def _reset_compiler_state() -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "reset"):
        dynamo.reset()


def evaluate_fresh_baseline_metric(
    run_cfg: DataConfig,
    *,
    data_splits: FineTuneDataSplits,
    horizon: int,
    split_name: str,
    baseline_kind: str,
    label: str,
) -> DualMetricTriplet | tuple[DualMetricTriplet, str] | None:
    _reset_compiler_state()
    prepared_trial = prepare_fine_tune_trial(
        run_cfg,
        data_splits=data_splits,
        horizon=horizon,
        pure_tabdpt=baseline_kind != "with_text",
    )
    X_context_proc, y_context, text_context, X_eval_proc, y_eval, y_eval_real, text_eval = _prepared_eval_inputs(
        prepared_trial,
        split_name=split_name,
    )

    if baseline_kind == "no_text":
        return evaluate_rolling(
            prepared_trial.reg,
            X_context_proc=X_context_proc,
            y_context=y_context,
            text_context=text_context,
            X_eval_proc=X_eval_proc,
            y_eval=y_eval,
            y_eval_real=y_eval_real,
            text_eval=text_eval,
            use_text=False,
            label=label,
            max_context=run_cfg.tuning.max_context,
            target_scaler=prepared_trial.target_scaler,
            horizon=horizon,
        )
    if baseline_kind == "with_text":
        return evaluate_rolling(
            prepared_trial.reg,
            X_context_proc=X_context_proc,
            y_context=y_context,
            text_context=text_context,
            X_eval_proc=X_eval_proc,
            y_eval=y_eval,
            y_eval_real=y_eval_real,
            text_eval=text_eval,
            use_text=True,
            label=label,
            max_context=run_cfg.tuning.max_context,
            target_scaler=prepared_trial.target_scaler,
            horizon=horizon,
        )
    if baseline_kind == "pca":
        return evaluate_rolling_pca(
            prepared_trial.reg,
            X_context_proc=X_context_proc,
            y_context=y_context,
            text_context=text_context,
            X_eval_proc=X_eval_proc,
            y_eval=y_eval,
            y_eval_real=y_eval_real,
            text_eval=text_eval,
            label=label,
            max_context=run_cfg.tuning.max_context,
            target_scaler=prepared_trial.target_scaler,
            horizon=horizon,
        )
    if baseline_kind == "truncate_text":
        return evaluate_rolling_truncate_text(
            prepared_trial.reg,
            X_context_proc=X_context_proc,
            y_context=y_context,
            text_context=text_context,
            X_eval_proc=X_eval_proc,
            y_eval=y_eval,
            y_eval_real=y_eval_real,
            text_eval=text_eval,
            label=label,
            max_context=run_cfg.tuning.max_context,
            target_scaler=prepared_trial.target_scaler,
            horizon=horizon,
        )
    raise ValueError(
        f"Unsupported baseline_kind: {baseline_kind!r}. "
        "Expected 'no_text', 'with_text', 'pca', or 'truncate_text'."
    )


def _layer_label(layer_idx: int) -> str:
    return f"L{layer_idx + 1}"


def _get_text_enhanced_blocks(
    reg: TabDPTRegressor,
) -> list[tuple[int, torch.nn.Module]]:
    text_blocks: list[tuple[int, torch.nn.Module]] = []
    for layer_idx, block in enumerate(reg.model.transformer_encoder):
        has_gate = getattr(block, "alpha", None) is not None
        if not getattr(block, "text_enhanced", False) and not has_gate:
            continue
        if not has_gate:
            raise RuntimeError(
                f"{_layer_label(layer_idx)} is marked as text-enhanced but has no alpha gate."
            )
        text_blocks.append((layer_idx, block))
    if not text_blocks:
        raise RuntimeError("Model has no text-enhanced transformer blocks. Is text_enhanced enabled?")
    return text_blocks


def get_text_enhanced_gate_params(
    reg: TabDPTRegressor,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the trainable gate tensors for every text-enhanced transformer block."""
    return [(_layer_label(layer_idx), block.alpha) for layer_idx, block in _get_text_enhanced_blocks(reg)]


def get_last_layer_gate_param(reg: TabDPTRegressor) -> torch.nn.Parameter:
    """Backward-compatible wrapper that returns the final text-enhanced block's gate tensor."""
    return get_text_enhanced_gate_params(reg)[-1][1]


def _get_text_mixing_modules_for_block(
    block: torch.nn.Module,
    *,
    layer_label: str,
) -> list[tuple[str, torch.nn.Module]]:
    """
    Return a block's text-mixing modules.

    The current branch uses per-head text projections/norms stored in
    `ModuleList`s, but we keep fallbacks for older experimental layouts to make
    local iteration less brittle.
    """
    if getattr(block, "text_head_projs", None) is not None:
        text_modules: list[tuple[str, torch.nn.Module]] = [("head_proj", block.text_head_projs)]
        if getattr(block, "text_head_q_norms", None) is not None:
            text_modules.append(("text_q_norm", block.text_head_q_norms))
        if getattr(block, "text_head_k_norms", None) is not None:
            text_modules.append(("text_k_norm", block.text_head_k_norms))
        return text_modules
    if getattr(block, "text_shared_proj", None) is not None:
        text_modules = [("shared_proj", block.text_shared_proj)]
        if getattr(block, "text_q_norm", None) is not None:
            text_modules.append(("text_q_norm", block.text_q_norm))
        if getattr(block, "text_k_norm", None) is not None:
            text_modules.append(("text_k_norm", block.text_k_norm))
        return text_modules
    if getattr(block, "text_attn_linears", None) is not None:
        return [(f"text_attn_{idx}", module) for idx, module in enumerate(block.text_attn_linears)]
    raise RuntimeError(
        f"{layer_label} has no recognized text-mixing module. Expected `text_head_projs` "
        "for the current architecture, `text_shared_proj` for the prior one, or "
        "`text_attn_linears` for the legacy one."
    )


def _get_text_enhanced_block_specs(
    reg: TabDPTRegressor,
) -> list[tuple[str, torch.nn.Module, list[tuple[str, torch.nn.Module]]]]:
    return [
        (_layer_label(layer_idx), block, _get_text_mixing_modules_for_block(block, layer_label=_layer_label(layer_idx)))
        for layer_idx, block in _get_text_enhanced_blocks(reg)
    ]


def _set_text_attn_dropout_mode(
    reg: TabDPTRegressor,
    *,
    enabled: bool,
) -> None:
    seen_module_ids: set[int] = set()
    for _, block in _get_text_enhanced_blocks(reg):
        dropout = getattr(block, "text_attn_dropout", None)
        if dropout is None or id(dropout) in seen_module_ids:
            continue
        dropout.train(enabled)
        seen_module_ids.add(id(dropout))


def _resolve_base_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _attention_regularization_signature(tuning_cfg: TuningConfig) -> tuple[float, str]:
    return (
        float(tuning_cfg.attention_dropout_p),
        tuning_cfg.qk_norm_type,
    )


def _configure_attention_regularization(
    reg: TabDPTRegressor,
    *,
    tuning_cfg: TuningConfig,
) -> None:
    base_model = _resolve_base_model(reg.model)
    signature = _attention_regularization_signature(tuning_cfg)
    if getattr(base_model, "_attention_regularization_signature", None) == signature:
        return

    transformer_encoder = getattr(base_model, "transformer_encoder", None)
    if transformer_encoder is None:
        return

    for block in transformer_encoder:
        configure = getattr(block, "configure_attention_regularization", None)
        if configure is None:
            continue
        configure(
            attention_dropout_p=tuning_cfg.attention_dropout_p,
            qk_norm_type=tuning_cfg.qk_norm_type,
        )

    _share_text_attention_modules(transformer_encoder)
    setattr(base_model, "_attention_regularization_signature", signature)


def _text_attention_logit_l2_penalty(
    reg: TabDPTRegressor,
    *,
    text_train: torch.Tensor,
    text_test: torch.Tensor,
) -> torch.Tensor:
    penalty_terms: list[torch.Tensor] = []
    seen_module_groups: set[tuple[int, ...]] = set()

    for _, block, text_modules in _get_text_enhanced_block_specs(reg):
        logits_fn = getattr(block, "_text_attention_logits", None)
        if logits_fn is None:
            continue
        module_group_key = tuple(id(module) for _, module in text_modules)
        if module_group_key in seen_module_groups:
            continue
        penalty_terms.append(logits_fn(text_train, text_test).pow(2).mean())
        seen_module_groups.add(module_group_key)

    if not penalty_terms:
        return text_train.new_zeros(())
    return torch.stack(penalty_terms).mean()


def _freeze_all_but_text_mixing(
    reg: TabDPTRegressor,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Freeze the model except the text-enhanced blocks' gate and text-attention parameters."""
    for p in reg.model.parameters():
        p.requires_grad_(False)

    gate_params: list[torch.nn.Parameter] = []
    text_attn_params: list[torch.nn.Parameter] = []
    seen_param_ids: set[int] = set()
    for _, gate in get_text_enhanced_gate_params(reg):
        if id(gate) in seen_param_ids:
            continue
        gate.requires_grad_(True)
        gate_params.append(gate)
        seen_param_ids.add(id(gate))

    for _, _, text_modules in _get_text_enhanced_block_specs(reg):
        for _, module in text_modules:
            for param in module.parameters():
                if id(param) in seen_param_ids:
                    continue
                param.requires_grad_(True)
                text_attn_params.append(param)
                seen_param_ids.add(id(param))
    return gate_params, text_attn_params


def freeze_all_but_last_text_mixing(
    reg: TabDPTRegressor,
) -> tuple[torch.nn.Parameter, list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Backward-compatible wrapper that now unfreezes every text-enhanced block."""
    gate_params, text_attn_params = _freeze_all_but_text_mixing(reg)
    return gate_params[-1], gate_params, text_attn_params


def freeze_all_but_last_gate(reg: TabDPTRegressor) -> tuple[torch.nn.Parameter, list[torch.nn.Parameter]]:
    """Backward-compatible wrapper that returns the combined gate + text-mixing params."""
    gate, gate_params, text_attn_params = freeze_all_but_last_text_mixing(reg)
    return gate, gate_params + text_attn_params


def build_text_mixing_optimizer(
    *,
    gate_params: list[torch.nn.Parameter],
    text_attn_params: list[torch.nn.Parameter],
    tuning_cfg: TuningConfig,
) -> schedulefree.AdamWScheduleFree:
    """Build one optimizer with separate LR groups for gate and text-attention params."""
    if not gate_params:
        raise ValueError("Expected at least one gate parameter for text-mixing fine-tuning.")
    if not text_attn_params:
        raise ValueError("Expected at least one text-attention parameter for text-mixing fine-tuning.")

    gate_param_ids = {id(param) for param in gate_params}
    text_attn_param_ids = {id(param) for param in text_attn_params}
    if gate_param_ids & text_attn_param_ids:
        raise ValueError("Gate and text-attention parameter groups must be disjoint.")

    param_groups = [
        {
            "params": gate_params,
            "lr": tuning_cfg.gate_lr,
            "weight_decay": 0.0,
        },
        {
            "params": text_attn_params,
            "lr": tuning_cfg.text_attn_lr,
            "weight_decay": 0.0,
        },
    ]
    return schedulefree.AdamWScheduleFree(
        param_groups,
        lr=tuning_cfg.gate_lr,
        weight_decay=0.0,
    )


def gate_stats(gate_logits: torch.Tensor) -> str:
    """Human-readable summary for per-head gate logits and their sigmoid values."""
    logits = gate_logits.detach().float().cpu().reshape(-1)
    gate = torch.sigmoid(logits)
    sample_count = min(6, gate.numel())
    sample_vals = gate[:sample_count].tolist()
    sample_str = ", ".join(f"{v:.3f}" for v in sample_vals)
    if gate.numel() > sample_count:
        sample_str = f"{sample_str}, ..."
    return (
        f"logits(mean/min/max)={logits.mean().item():.4f}/{logits.min().item():.4f}/{logits.max().item():.4f} | "
        f"sigmoid(mean/min/max)={gate.mean().item():.4f}/{gate.min().item():.4f}/{gate.max().item():.4f} | "
        f"sigmoid(sample)=[{sample_str}]"
    )


def text_mixing_gate_stats(reg: TabDPTRegressor) -> str:
    """Human-readable summary for the gates in every text-enhanced transformer block."""
    return " || ".join(
        f"{layer_label} gate: {gate_stats(gate_logits)}"
        for layer_label, gate_logits in get_text_enhanced_gate_params(reg)
    )


def _text_module_param_summary(label: str, module: torch.nn.Module) -> str:
    submodules = list(module) if isinstance(module, torch.nn.ModuleList) else [module]
    resolved_modules: list[torch.nn.Module] = []
    for submodule in submodules:
        weighted_module = submodule
        if not hasattr(weighted_module, "weight"):
            weighted_module = next((m for m in submodule.modules() if isinstance(m, torch.nn.Linear)), None)
            if weighted_module is None:
                continue
        resolved_modules.append(weighted_module)
    if not resolved_modules:
        return f"{label}(no-linear)"

    if len(resolved_modules) == 1:
        weighted_module = resolved_modules[0]
        weight = weighted_module.weight.detach().float().cpu().reshape(-1)
        w_val = weight.item() if weight.numel() == 1 else weight.mean().item()
        w_norm = weight.norm().item()
        line = f"{label}(||W||={w_norm:.3f}, mean={w_val:.6f}"
        bias_param = getattr(weighted_module, "bias", None)
        if bias_param is not None:
            bias = bias_param.detach().float().cpu().reshape(-1)
            b_val = bias.item() if bias.numel() == 1 else bias.mean().item()
            line += f", b={b_val:.3f}"
        line += ")"
        return line

    weight_norms = torch.tensor(
        [weighted_module.weight.detach().float().cpu().reshape(-1).norm().item() for weighted_module in resolved_modules]
    )
    weight_means = torch.tensor(
        [weighted_module.weight.detach().float().cpu().reshape(-1).mean().item() for weighted_module in resolved_modules]
    )
    line = (
        f"{label}(heads={len(resolved_modules)}, "
        f"||W|| mean/min/max={weight_norms.mean().item():.3f}/"
        f"{weight_norms.min().item():.3f}/{weight_norms.max().item():.3f}, "
        f"mean(mean)={weight_means.mean().item():.6f}"
    )
    bias_means = [
        bias_param.detach().float().cpu().reshape(-1).mean().item()
        for weighted_module in resolved_modules
        for bias_param in [getattr(weighted_module, "bias", None)]
        if bias_param is not None
    ]
    if bias_means:
        line += f", b_mean={float(np.mean(bias_means)):.3f}"
    line += ")"
    return line


def get_trainining_info(reg: TabDPTRegressor, gate_logits: torch.Tensor | None = None) -> str:
    """Compact one-line summary for each text-enhanced block's gate and text-mixing params."""
    del gate_logits
    layer_summaries: list[str] = []
    for layer_label, block, text_modules in _get_text_enhanced_block_specs(reg):
        module_summaries = [_text_module_param_summary(module_label, module) for module_label, module in text_modules]
        layer_summaries.append(
            f"{layer_label} gate: {gate_stats(block.alpha)} | {' '.join(module_summaries)}"
        )
    return " || ".join(layer_summaries)


def format_text_score_info(
    reg: TabDPTRegressor,
    *,
    text_train: torch.Tensor,
    text_test: torch.Tensor,
    sample_size: int = 8,
) -> str:
    """
    Summarize the raw text score logits produced by the learned projection path.

    The current text-enhanced model computes scores with each text-enhanced
    block's `_text_attention_logits(...)`, which returns raw pre-softmax logits
    with shape (B, H, N_test, N_train).
    """
    try:
        text_blocks = _get_text_enhanced_blocks(reg)
    except RuntimeError:
        return "text_score(unavailable)"

    summaries: list[str] = []
    for layer_idx, block in text_blocks:
        if getattr(block, "_text_attention_logits", None) is None:
            summaries.append(f"{_layer_label(layer_idx)} text_score(unavailable)")
            continue

        with torch.no_grad():
            logits = block._text_attention_logits(text_train, text_test).detach().float().cpu()

        flat = logits.reshape(-1)
        head0_query = logits[0, 0, 0].reshape(-1)
        sample_count = min(sample_size, head0_query.numel())
        sample_vals = ", ".join(f"{v:.4f}" for v in head0_query[:sample_count].tolist())
        if head0_query.numel() > sample_count:
            sample_vals = f"{sample_vals}, ..."

        summaries.append(
            f"{_layer_label(layer_idx)} text_score(shape={tuple(logits.shape)}, "
            f"mean/min/max={flat.mean().item():.4f}/{flat.min().item():.4f}/{flat.max().item():.4f}, "
            f"head0_sample=[{sample_vals}])"
        )

    return " || ".join(summaries)


def _normalized_regression_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mean_y: torch.Tensor,
    std_y: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    mean_y = mean_y.reshape(-1)
    std_y = std_y.reshape(-1)
    y_pred_norm = (y_pred - mean_y) / std_y
    y_true_norm = (y_true - mean_y) / std_y
    if loss_type == "l1":
        return torch.nn.functional.l1_loss(y_pred_norm, y_true_norm)
    if loss_type == "l2":
        return torch.nn.functional.mse_loss(y_pred_norm, y_true_norm)
    raise ValueError(f"Unsupported tuning loss_type: {loss_type!r}. Expected 'l1' or 'l2'.")


def _early_stopping_score(
    *,
    metric_name: str,
    mae: float,
    rmse: float,
    mape: float,
) -> float:
    metric_name = metric_name.lower()
    if metric_name == "mae":
        return mae
    if metric_name == "rmse":
        return rmse
    if metric_name == "mape":
        return mape
    raise ValueError(
        f"Unsupported early_stopping_metric: {metric_name!r}. Expected one of 'mae', 'rmse', or 'mape'."
    )


def _evaluate_rolling_loss_and_mae(
    reg: TabDPTRegressor,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray | None,
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    y_eval_real: np.ndarray,
    text_eval: np.ndarray | None,
    use_text: bool,
    max_context: int | None,
    loss_type: str,
    target_scaler: StandardScaler | None,
    horizon: int = 1,
) -> tuple[float, float, float, float, float, float, float]:
    """Run rolling evaluation and return average normalized loss plus metrics."""
    if use_text and (text_context is None or text_eval is None):
        raise ValueError("Rolling eval with text requires context and eval text arrays.")

    reg.model.eval()
    preds = np.zeros(len(y_eval), dtype=np.float32)
    loss_sum = 0.0
    with torch.no_grad():
        for idx in range(len(y_eval)):
            X_train_step, y_train_step, text_train_step = _build_rolling_train_step(
                X_context_proc=X_context_proc,
                y_context=y_context,
                text_context=text_context if use_text else None,
                X_eval_proc=X_eval_proc,
                y_eval=y_eval,
                text_eval=text_eval if use_text else None,
                idx=idx,
                horizon=horizon,
                max_context=max_context,
            )
            X_test_step = X_eval_proc[idx:idx + 1]
            if use_text:
                train_text_batch = text_train_step[None, ...]
                test_text_batch = text_eval[idx:idx + 1][None, ...]
                text_train_b, text_test_b = reg.text_embeddings_batched(
                    train_text_batch,
                    test_text_batch,
                )
            else:
                text_train_b = text_test_b = None

            X_train_tensor = torch.tensor(X_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_train_tensor = pad_x(X_train_tensor, reg.max_features)
            X_test_tensor = torch.tensor(X_test_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_test_tensor = pad_x(X_test_tensor, reg.max_features)
            y_context_tensor = torch.tensor(y_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

            pred, std_y, mean_y = reg.model(
                x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                y_src=y_context_tensor.unsqueeze(-1),
                task="reg",
                text_train=text_train_b,
                text_test=text_test_b,
            )
            pred = pred.squeeze(-1).reshape(-1)
            preds[idx] = pred.detach().cpu().numpy()[0]

            y_target = torch.tensor(
                y_eval[idx:idx + 1],
                dtype=torch.float32,
                device=reg.device,
            )
            point_loss = _normalized_regression_loss(
                pred,
                y_target,
                mean_y,
                std_y,
                loss_type,
            )
            loss_sum += float(point_loss.detach().cpu())

    mse = mean_squared_error(y_eval, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_eval, preds)
    denom = np.clip(np.abs(y_eval), 1e-8, None)
    mape = float(np.mean(np.abs((y_eval - preds) / denom)) * 100.0)
    real_preds = inverse_transform_targets(preds, target_scaler)
    real_mse = mean_squared_error(y_eval_real, real_preds)
    real_rmse = float(np.sqrt(real_mse))
    real_mae = mean_absolute_error(y_eval_real, real_preds)
    real_denom = np.clip(np.abs(y_eval_real), 1e-8, None)
    real_mape = float(np.mean(np.abs((y_eval_real - real_preds) / real_denom)) * 100.0)
    avg_loss = loss_sum / len(y_eval)
    return avg_loss, mae, rmse, mape, real_mae, real_rmse, real_mape


def evaluate_prepared_split(
    prepared_trial: PreparedFineTuneTrial,
    *,
    split_name: str,
    use_text: bool,
    tuning_cfg: TuningConfig,
) -> RollingMetrics:
    X_context_proc, y_context, text_context, X_eval_proc, y_eval, y_eval_real, text_eval = _prepared_eval_inputs(
        prepared_trial,
        split_name=split_name,
    )
    loss, mae, rmse, mape, real_mae, real_rmse, real_mape = _evaluate_rolling_loss_and_mae(
        prepared_trial.reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        y_eval_real=y_eval_real,
        text_eval=text_eval,
        use_text=use_text,
        max_context=tuning_cfg.max_context,
        loss_type=tuning_cfg.loss_type,
        target_scaler=prepared_trial.target_scaler,
        horizon=prepared_trial.prediction_horizon,
    )
    return RollingMetrics(
        loss=loss,
        mae=mae,
        rmse=rmse,
        mape=mape,
        real_mae=real_mae,
        real_rmse=real_rmse,
        real_mape=real_mape,
    )


def _print_epoch_section(
    *,
    epoch: int,
    train_metrics: RollingMetrics,
    val_metrics: RollingMetrics,
    param_stats: str,
) -> None:
    print(f"\n== Epoch {epoch:02d} ==")
    print(f"Train  | Loss [normalized]: {train_metrics.loss:.4f}")
    _format_dual_metrics(
        "Train",
        (train_metrics.mae, train_metrics.rmse, train_metrics.mape),
        (train_metrics.real_mae, train_metrics.real_rmse, train_metrics.real_mape),
    )
    print(f"Val    | Loss [normalized]: {val_metrics.loss:.4f}")
    _format_dual_metrics(
        "Val",
        (val_metrics.mae, val_metrics.rmse, val_metrics.mape),
        (val_metrics.real_mae, val_metrics.real_rmse, val_metrics.real_mape),
    )
    print(f"Params | {param_stats}")


def fine_tune_prepared_trial(
    prepared_trial: PreparedFineTuneTrial,
    *,
    tuning_cfg: TuningConfig,
) -> FineTuneOutcome:
    splits = prepared_trial.splits
    return fine_tune_external_gate(
        prepared_trial.reg,
        X_context_proc=prepared_trial.X_context_proc,
        y_context=splits.y_context,
        text_context=splits.text_context,
        X_train_proc=prepared_trial.X_train_proc,
        y_train=splits.y_train,
        y_train_real=prepared_trial.raw_splits.y_train,
        text_train=splits.text_train,
        tuning_cfg=tuning_cfg,
        X_val_proc=prepared_trial.X_val_proc,
        y_val=splits.y_val,
        y_val_real=prepared_trial.raw_splits.y_val,
        text_val=splits.text_val,
        target_scaler=prepared_trial.target_scaler,
        prediction_horizon=prepared_trial.prediction_horizon,
    )


def fine_tune_external_gate(
    reg: TabDPTRegressor,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray,
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    y_train_real: np.ndarray,
    text_train: np.ndarray,
    tuning_cfg: TuningConfig,
    X_val_proc: np.ndarray,
    y_val: np.ndarray,
    y_val_real: np.ndarray,
    text_val: np.ndarray,
    target_scaler: StandardScaler | None,
    prediction_horizon: int = 1,
) -> FineTuneOutcome:
    """
    Fine-tune only the text-enhanced blocks' text-mixing parameters.

    Why call the model directly?
    - `reg.predict()` disables gradients, so we can't optimize with it.
    - We still re-use everything from `reg.fit()`:
        - fitted imputers/scalers
        - batching helpers that shape text inputs for the text-enhanced blocks

    Rolling window (per step inside each epoch):
    - Fixed context = global context + train rows before this batch
    - Within the batch, each row is predicted from base context plus earlier batch rows
    - Gradients are accumulated point-by-point and averaged over the batch
    """
    if tuning_cfg.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")

    _configure_attention_regularization(reg, tuning_cfg=tuning_cfg)
    _, gate_params, text_attn_params = freeze_all_but_last_text_mixing(reg)
    text_block_labels = ", ".join(layer_label for layer_label, _ in get_text_enhanced_gate_params(reg))
    optimizer = build_text_mixing_optimizer(
        gate_params=gate_params,
        text_attn_params=text_attn_params,
        tuning_cfg=tuning_cfg,
    )
    print(
        f"Tuning text-mixing params on blocks [{text_block_labels}] "
        f"(loss_type={tuning_cfg.loss_type}, gate_lr={tuning_cfg.gate_lr}, "
        f"text_attn_lr={tuning_cfg.text_attn_lr}, optimizer={optimizer.__class__.__name__}, "
        f"attn_dropout={tuning_cfg.attention_dropout_p}, qk_norm={tuning_cfg.qk_norm_type}, "
        f"text_logit_l2={tuning_cfg.text_attention_logit_l2}, "
        f"early_stopping_metric={tuning_cfg.early_stopping_metric}, "
        f"patience={tuning_cfg.early_stopping_patience})"
    )

    reg.model.eval()
    _set_text_attn_dropout_mode(reg, enabled=True)
    if hasattr(optimizer, "train"):
        optimizer.train()

    num_steps = int(np.ceil(len(y_train) / tuning_cfg.tune_batch_size))
    best_score = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    top_validation_mae_checkpoints: list[ValidationMaeCheckpoint] = []
    epochs_without_improvement = 0
    val_context_proc, val_y_context, val_text_context, _, _, _ = build_history_and_eval_split(
        X_context_proc,
        X_train_proc,
        X_val_proc,
        X_val_proc[:0],
        y_context,
        y_train,
        y_val,
        y_val[:0],
        text_context,
        text_train,
        text_val,
        text_val[:0],
        "val",
        prediction_horizon,
    )

    for epoch in range(1, tuning_cfg.epochs + 1):
        _set_text_attn_dropout_mode(reg, enabled=True)
        for step_idx in range(num_steps):
            optimizer.zero_grad()

            start = step_idx * tuning_cfg.tune_batch_size
            end = min(len(y_train), start + tuning_cfg.tune_batch_size)

            current_batch_size = end - start

            for point_idx in range(start, end):
                X_context_step, y_context_step, text_context_step = _build_rolling_train_step(
                    X_context_proc=X_context_proc,
                    y_context=y_context,
                    text_context=text_context,
                    X_eval_proc=X_train_proc,
                    y_eval=y_train,
                    text_eval=text_train,
                    idx=point_idx,
                    horizon=prediction_horizon,
                    max_context=tuning_cfg.max_context,
                )

                train_text_batch = text_context_step[None, ...]
                test_text_batch = text_train[point_idx:point_idx + 1][None, ...]
                text_train_b, text_test_b = reg.text_embeddings_batched(
                    train_text_batch,
                    test_text_batch,
                )

                X_train_tensor = torch.tensor(X_context_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
                X_train_tensor = pad_x(X_train_tensor, reg.max_features)
                X_test_tensor = torch.tensor(
                    X_train_proc[point_idx:point_idx + 1],
                    dtype=torch.float32,
                    device=reg.device,
                ).unsqueeze(0)
                X_test_tensor = pad_x(X_test_tensor, reg.max_features)
                y_context_tensor = torch.tensor(y_context_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

                pred_point, std_y, mean_y = reg.model(
                    x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                    y_src=y_context_tensor.unsqueeze(-1),
                    task="reg",
                    text_train=text_train_b,
                    text_test=text_test_b,
                )
                pred_point = pred_point.squeeze(-1).reshape(-1)

                y_point_target = torch.tensor(
                    y_train[point_idx:point_idx + 1],
                    dtype=torch.float32,
                    device=reg.device,
                )
                point_loss = _normalized_regression_loss(
                    pred_point,
                    y_point_target,
                    mean_y,
                    std_y,
                    tuning_cfg.loss_type,
                )
                if tuning_cfg.text_attention_logit_l2 > 0.0:
                    point_loss = point_loss + (
                        tuning_cfg.text_attention_logit_l2
                        * _text_attention_logit_l2_penalty(
                            reg,
                            text_train=text_train_b,
                            text_test=text_test_b,
                        )
                    )
                (point_loss / current_batch_size).backward()

            optimizer.step()

            if tuning_cfg.gate_logit_clamp is not None:
                with torch.no_grad():
                    for gate in gate_params:
                        gate.clamp_(-tuning_cfg.gate_logit_clamp, tuning_cfg.gate_logit_clamp)

        _set_text_attn_dropout_mode(reg, enabled=False)
        train_loss, train_mae, train_rmse, train_mape, train_real_mae, train_real_rmse, train_real_mape = _evaluate_rolling_loss_and_mae(
            reg,
            X_context_proc=X_context_proc,
            y_context=y_context,
            text_context=text_context,
            X_eval_proc=X_train_proc,
            y_eval=y_train,
            y_eval_real=y_train_real,
            text_eval=text_train,
            use_text=True,
            max_context=tuning_cfg.max_context,
            loss_type=tuning_cfg.loss_type,
            target_scaler=target_scaler,
            horizon=prediction_horizon,
        )
        val_loss, val_mae, val_rmse, val_mape, val_real_mae, val_real_rmse, val_real_mape = _evaluate_rolling_loss_and_mae(
            reg,
            X_context_proc=val_context_proc,
            y_context=val_y_context,
            text_context=val_text_context,
            X_eval_proc=X_val_proc,
            y_eval=y_val,
            y_eval_real=y_val_real,
            text_eval=text_val,
            use_text=True,
            max_context=tuning_cfg.max_context,
            loss_type=tuning_cfg.loss_type,
            target_scaler=target_scaler,
            horizon=prediction_horizon,
        )
        param_stats = get_trainining_info(reg)
        train_metrics = RollingMetrics(
            loss=train_loss,
            mae=train_mae,
            rmse=train_rmse,
            mape=train_mape,
            real_mae=train_real_mae,
            real_rmse=train_real_rmse,
            real_mape=train_real_mape,
        )
        val_metrics = RollingMetrics(
            loss=val_loss,
            mae=val_mae,
            rmse=val_rmse,
            mape=val_mape,
            real_mae=val_real_mae,
            real_rmse=val_real_rmse,
            real_mape=val_real_mape,
        )
        _print_epoch_section(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            param_stats=param_stats,
        )
        _maybe_record_top_validation_mae_checkpoint(
            top_validation_mae_checkpoints,
            epoch=epoch,
            val_mae=val_mae,
            model=reg.model,
        )
        current_score = _early_stopping_score(
            metric_name=tuning_cfg.early_stopping_metric,
            mae=val_mae,
            rmse=val_rmse,
            mape=val_mape,
        )
        if current_score < best_score:
            best_score = current_score
            best_epoch = epoch
            best_state_dict = _clone_state_dict_to_cpu(reg.model.state_dict())
            epochs_without_improvement = 0
            print(
                f"New best validation {tuning_cfg.early_stopping_metric.upper()}: "
                f"{current_score:.4f} at epoch {epoch:02d}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement for {epochs_without_improvement} epoch(s) "
                f"(best epoch {best_epoch:02d}, best {tuning_cfg.early_stopping_metric.upper()}={best_score:.4f})"
            )
            if epochs_without_improvement >= tuning_cfg.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch:02d}")
                break

    if best_state_dict is None:
        raise RuntimeError("Early stopping did not record a best model state.")

    reg.model.load_state_dict(best_state_dict)
    _set_text_attn_dropout_mode(reg, enabled=False)
    print(
        f"Restored best text-mixing params from epoch {best_epoch:02d} "
        f"(validation {tuning_cfg.early_stopping_metric.upper()}={best_score:.4f})"
    )
    print(
        "Top validation-MAE checkpoints: "
        + ", ".join(
            f"#{rank} epoch {checkpoint.epoch:02d} (MAE={checkpoint.val_mae:.4f})"
            for rank, checkpoint in enumerate(top_validation_mae_checkpoints, start=1)
        )
    )

    for p in reg.model.parameters():
        p.requires_grad_(False)

    return FineTuneOutcome(
        top_validation_mae_checkpoints=top_validation_mae_checkpoints,
        best_epoch=best_epoch,
        best_score=best_score,
    )


def _mean_metric_triplets(metric_triplets: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(
        float(np.mean([triplet[idx] for triplet in metric_triplets]))
        for idx in range(3)
    )


def _run_single_horizon(
    run_cfg: DataConfig,
    *,
    data_splits: FineTuneDataSplits,
    horizon: int,
) -> HorizonRunResult:
    print(f"\n{'=' * 16} Horizon {horizon}/{run_cfg.prediction_window} {'=' * 16}")

    print("\n== Baseline (before tuning) ==")
    baseline_no_text_train = evaluate_fresh_baseline_metric(
        run_cfg,
        data_splits=data_splits,
        horizon=horizon,
        split_name="train",
        baseline_kind="no_text",
        label="Train (no text attn)",
    )
    baseline_pca_train = evaluate_fresh_baseline_metric(
        run_cfg,
        data_splits=data_splits,
        horizon=horizon,
        split_name="train",
        baseline_kind="pca",
        label="Train (PCA)",
    )
    baseline_truncate_train = evaluate_fresh_baseline_metric(
        run_cfg,
        data_splits=data_splits,
        horizon=horizon,
        split_name="train",
        baseline_kind="truncate_text",
        label="Train (text truncate)",
    )
    baseline_no_text_val = evaluate_fresh_baseline_metric(
        run_cfg,
        data_splits=data_splits,
        horizon=horizon,
        split_name="val",
        baseline_kind="no_text",
        label="Val (no text attn)",
    )
    baseline_text_val = evaluate_fresh_baseline_metric(
        run_cfg,
        data_splits=data_splits,
        horizon=horizon,
        split_name="val",
        baseline_kind="with_text",
        label="Val (with text attn)",
    )
    baseline_pca_val = evaluate_fresh_baseline_metric(
        run_cfg,
        data_splits=data_splits,
        horizon=horizon,
        split_name="val",
        baseline_kind="pca",
        label="Val (PCA)",
    )
    baseline_truncate_val = evaluate_fresh_baseline_metric(
        run_cfg,
        data_splits=data_splits,
        horizon=horizon,
        split_name="val",
        baseline_kind="truncate_text",
        label="Val (text truncate)",
    )

    _reset_compiler_state()
    prepared_trial = prepare_fine_tune_trial(run_cfg, data_splits=data_splits, horizon=horizon)
    reg = prepared_trial.reg
    splits = prepared_trial.splits
    X_context_proc = prepared_trial.X_context_proc
    X_train_proc = prepared_trial.X_train_proc
    X_val_proc = prepared_trial.X_val_proc
    X_test_proc = prepared_trial.X_test_proc
    y_context = splits.y_context
    text_context = splits.text_context
    y_train = splits.y_train
    text_train = splits.text_train
    y_val = splits.y_val
    text_val = splits.text_val
    y_test = splits.y_test
    text_test = splits.text_test
    y_train_real = prepared_trial.raw_splits.y_train
    y_test_real = prepared_trial.raw_splits.y_test
    if run_cfg.tuning.debug_text_effect:
        delta_mae = baseline_text_val[0][0] - baseline_no_text_val[0][0]
        delta_rmse = baseline_text_val[0][1] - baseline_no_text_val[0][1]
        delta_mape = baseline_text_val[0][2] - baseline_no_text_val[0][2]
        print(
            f"Val text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
        print("Initial text-mixing params:", get_trainining_info(reg))
    else:
        print("Initial text-mixing gates:", text_mixing_gate_stats(reg))

    print("\n== Fine-tuning text-mixing parameters ==")
    fine_tune_outcome = fine_tune_prepared_trial(
        prepared_trial,
        tuning_cfg=run_cfg.tuning,
    )

    print("\n== After tuning ==")
    _format_dual_metrics("Train (no text attn)", *baseline_no_text_train)
    _format_dual_metrics("Train (PCA)", *baseline_pca_train)
    if baseline_truncate_train is not None:
        _format_dual_metrics(baseline_truncate_train[1], *baseline_truncate_train[0])
    train_context_proc, train_y_context, train_text_context, train_eval_proc, train_y_eval, train_y_eval_real, train_text_eval = (
        _prepared_eval_inputs(prepared_trial, split_name="train")
    )
    evaluate_rolling(
        reg,
        X_context_proc=train_context_proc,
        y_context=train_y_context,
        text_context=train_text_context,
        X_eval_proc=train_eval_proc,
        y_eval=train_y_eval,
        y_eval_real=train_y_eval_real,
        text_eval=train_text_eval,
        use_text=True,
        label="Train (with text attn)",
        max_context=run_cfg.tuning.max_context,
        target_scaler=prepared_trial.target_scaler,
        horizon=horizon,
    )
    test_context_proc, test_y_context, test_text_context, test_eval_proc, test_y_eval, test_y_eval_real, test_text_eval = (
        _prepared_eval_inputs(prepared_trial, split_name="test")
    )
    tuned_no_text_test = evaluate_rolling(
        reg,
        X_context_proc=test_context_proc,
        y_context=test_y_context,
        text_context=test_text_context,
        X_eval_proc=test_eval_proc,
        y_eval=test_y_eval,
        y_eval_real=test_y_eval_real,
        text_eval=test_text_eval,
        use_text=False,
        label="Test (no text attn)",
        max_context=run_cfg.tuning.max_context,
        target_scaler=prepared_trial.target_scaler,
        horizon=horizon,
    )
    evaluate_rolling_pca(
        reg,
        X_context_proc=test_context_proc,
        y_context=test_y_context,
        text_context=test_text_context,
        X_eval_proc=test_eval_proc,
        y_eval=test_y_eval,
        y_eval_real=test_y_eval_real,
        text_eval=test_text_eval,
        label="Test (PCA)",
        max_context=run_cfg.tuning.max_context,
        target_scaler=prepared_trial.target_scaler,
        horizon=horizon,
    )
    evaluate_rolling_truncate_text(
        reg,
        X_context_proc=test_context_proc,
        y_context=test_y_context,
        text_context=test_text_context,
        X_eval_proc=test_eval_proc,
        y_eval=test_y_eval,
        y_eval_real=test_y_eval_real,
        text_eval=test_text_eval,
        label="Test (text truncate)",
        max_context=run_cfg.tuning.max_context,
        target_scaler=prepared_trial.target_scaler,
        horizon=horizon,
    )
    tuned_text_test = evaluate_rolling(
        reg,
        X_context_proc=test_context_proc,
        y_context=test_y_context,
        text_context=test_text_context,
        X_eval_proc=test_eval_proc,
        y_eval=test_y_eval,
        y_eval_real=test_y_eval_real,
        text_eval=test_text_eval,
        use_text=True,
        label="Test (with text attn)",
        max_context=run_cfg.tuning.max_context,
        target_scaler=prepared_trial.target_scaler,
        horizon=horizon,
    )
    if run_cfg.tuning.debug_text_effect:
        delta_mae = tuned_text_test[0][0] - tuned_no_text_test[0][0]
        delta_rmse = tuned_text_test[0][1] - tuned_no_text_test[0][1]
        delta_mape = tuned_text_test[0][2] - tuned_no_text_test[0][2]
        print(
            f"Test text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
        print("Tuned text-mixing params:", get_trainining_info(reg))
    else:
        print("Tuned text-mixing gates:", text_mixing_gate_stats(reg))

    restored_best_state_dict = _clone_state_dict_to_cpu(reg.model.state_dict())
    print(
        f"\n== Top {len(fine_tune_outcome.top_validation_mae_checkpoints)} "
        "Validation-MAE Models On Test =="
    )
    for rank, checkpoint in enumerate(fine_tune_outcome.top_validation_mae_checkpoints, start=1):
        reg.model.load_state_dict(checkpoint.state_dict)
        print(f"Rank {rank} | Epoch {checkpoint.epoch:02d} | Val MAE: {checkpoint.val_mae:.4f}")
        evaluate_rolling(
            reg,
            X_context_proc=test_context_proc,
            y_context=test_y_context,
            text_context=test_text_context,
            X_eval_proc=test_eval_proc,
            y_eval=test_y_eval,
            y_eval_real=test_y_eval_real,
            text_eval=test_text_eval,
            use_text=True,
            label=f"Test top {rank} (with text attn)",
            max_context=run_cfg.tuning.max_context,
            target_scaler=prepared_trial.target_scaler,
            horizon=horizon,
        )
    reg.model.load_state_dict(restored_best_state_dict)

    return HorizonRunResult(
        horizon=horizon,
        best_epoch=fine_tune_outcome.best_epoch,
        tuned_no_text_test_normalized=tuned_no_text_test[0],
        tuned_no_text_test_real=tuned_no_text_test[1],
        tuned_text_test_normalized=tuned_text_test[0],
        tuned_text_test_real=tuned_text_test[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune text-mixing parameters on the final text-enhanced transformer layers."
    )
    parser.add_argument("--dataset", help="Dataset key in the config file.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    args = parser.parse_args()

    run_cfg = load_fine_tune_config(args.config, args.dataset)
    set_random_seeds(run_cfg.seed)

    if args.dataset:
        print(f"Using dataset '{args.dataset}' from {args.config}")
    else:
        print(f"Using dataset config {args.config}")

    data_splits = load_and_split_fine_tune_data(run_cfg)
    print(
        "Split sizes: "
        f"context={len(data_splits.y_context)} train={len(data_splits.y_train)} "
        f"val={len(data_splits.y_val)} test={len(data_splits.y_test)} | "
        f"prediction_window={run_cfg.prediction_window}"
    )

    horizon_results = [
        _run_single_horizon(run_cfg, data_splits=data_splits, horizon=horizon)
        for horizon in prediction_horizons(run_cfg)
    ]

    if len(horizon_results) > 1:
        print("\n================ Mean Summary Across Horizons ================")
        _format_dual_metrics(
            "Mean test (no text attn)",
            _mean_metric_triplets([result.tuned_no_text_test_normalized for result in horizon_results]),
            _mean_metric_triplets([result.tuned_no_text_test_real for result in horizon_results]),
        )
        _format_dual_metrics(
            "Mean test (with text attn)",
            _mean_metric_triplets([result.tuned_text_test_normalized for result in horizon_results]),
            _mean_metric_triplets([result.tuned_text_test_real for result in horizon_results]),
        )
        print(
            "Best epochs by horizon: "
            + ", ".join(f"h{result.horizon}={result.best_epoch}" for result in horizon_results)
        )


if __name__ == "__main__":
    main()
