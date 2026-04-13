from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import csv
import json
import multiprocessing
import random
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    from .eval_fine_tune import evaluate_rolling, evaluate_rolling_pca, evaluate_rolling_truncate_text
    from .fine_tune_configs import DataConfig, load_fine_tune_config
    from .fine_tune_dpt import (
        FineTuneDataSplits,
        PreparedFineTuneTrial,
        RollingMetrics,
        evaluate_prepared_split,
        fine_tune_prepared_trial,
        load_and_split_fine_tune_data,
        prepare_fine_tune_trial,
        set_random_seeds,
    )
    from .random_search_configs import RandomSearchConfig, SearchSpaceConfig, load_random_search_config
except ImportError:
    from eval_fine_tune import evaluate_rolling, evaluate_rolling_pca, evaluate_rolling_truncate_text
    from fine_tune_configs import DataConfig, load_fine_tune_config
    from fine_tune_dpt import (
        FineTuneDataSplits,
        PreparedFineTuneTrial,
        RollingMetrics,
        evaluate_prepared_split,
        fine_tune_prepared_trial,
        load_and_split_fine_tune_data,
        prepare_fine_tune_trial,
        set_random_seeds,
    )
    from random_search_configs import RandomSearchConfig, SearchSpaceConfig, load_random_search_config


SEARCH_FIELD_NAMES = (
    "text_attn_layers",
    "epochs",
    "gate_lr",
    "text_attn_lr",
    "gate_logit_clamp",
    "tune_batch_size",
    "max_context",
)


@dataclass(frozen=True)
class RandomSearchTrialSpec:
    text_attn_layers: list[int]
    epochs: int
    gate_lr: float
    text_attn_lr: float
    gate_logit_clamp: float | None
    tune_batch_size: int
    max_context: int | None


@dataclass(frozen=True)
class RandomSearchTrialResult:
    trial_index: int
    text_attn_layers: list[int]
    epochs: int
    gate_lr: float
    text_attn_lr: float
    gate_logit_clamp: float | None
    tune_batch_size: int
    max_context: int | None
    val_loss: float
    val_mae: float
    val_rmse: float
    val_mape: float
    ranking_score: float
    best_epoch: int | None = None
    test_loss: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_mape: float | None = None
    horizon_metrics: list["HorizonTrialMetrics"] | None = None


@dataclass(frozen=True)
class HorizonTrialMetrics:
    horizon: int
    best_epoch: int | None
    val_loss: float
    val_mae: float
    val_rmse: float
    val_mape: float
    test_loss: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_mape: float | None = None


@dataclass(frozen=True)
class SummaryMetrics:
    mae: float
    rmse: float
    mape: float


@dataclass(frozen=True)
class SharedBaselineResult:
    text_attn_layers: list[int]
    max_context: int | None
    val_no_text: SummaryMetrics
    val_with_text: SummaryMetrics
    val_pca: SummaryMetrics
    val_truncate: SummaryMetrics | None
    val_truncate_label: str | None
    test_no_text: SummaryMetrics
    test_with_text: SummaryMetrics
    test_pca: SummaryMetrics
    test_truncate: SummaryMetrics | None
    test_truncate_label: str | None
    per_horizon: list["HorizonBaselineResult"] | None = None


@dataclass(frozen=True)
class HorizonBaselineResult:
    horizon: int
    val_no_text: SummaryMetrics
    val_with_text: SummaryMetrics
    val_pca: SummaryMetrics
    val_truncate: SummaryMetrics | None
    val_truncate_label: str | None
    test_no_text: SummaryMetrics
    test_with_text: SummaryMetrics
    test_pca: SummaryMetrics
    test_truncate: SummaryMetrics | None
    test_truncate_label: str | None


@dataclass(frozen=True)
class TrialExecutionArtifacts:
    result: RandomSearchTrialResult
    checkpoint_path: str
    assigned_device: str | None = None


class _TeeWriter:
    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _clone_state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[name] = value.detach().cpu().clone()
        else:
            cloned[name] = copy.deepcopy(value)
    return cloned


def _metrics_from_triplet(metrics: tuple[float, float, float]) -> SummaryMetrics:
    return SummaryMetrics(mae=metrics[0], rmse=metrics[1], mape=metrics[2])


def _mean_summary_metrics(metrics_list: list[SummaryMetrics]) -> SummaryMetrics:
    return SummaryMetrics(
        mae=float(np.mean([metrics.mae for metrics in metrics_list])),
        rmse=float(np.mean([metrics.rmse for metrics in metrics_list])),
        mape=float(np.mean([metrics.mape for metrics in metrics_list])),
    )


def _mean_rolling_metrics(metrics_list: list[RollingMetrics]) -> RollingMetrics:
    return RollingMetrics(
        loss=float(np.mean([metrics.loss for metrics in metrics_list])),
        mae=float(np.mean([metrics.mae for metrics in metrics_list])),
        rmse=float(np.mean([metrics.rmse for metrics in metrics_list])),
        mape=float(np.mean([metrics.mape for metrics in metrics_list])),
    )


def _serialize_summary_metrics(metrics: SummaryMetrics | None, *, label: str | None = None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    payload = {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "mape": metrics.mape,
    }
    if label is not None:
        payload["label"] = label
    return payload


def _serialize_horizon_trial_metrics(metrics: HorizonTrialMetrics) -> dict[str, Any]:
    return {
        "horizon": metrics.horizon,
        "best_epoch": metrics.best_epoch,
        "val_loss": metrics.val_loss,
        "val_mae": metrics.val_mae,
        "val_rmse": metrics.val_rmse,
        "val_mape": metrics.val_mape,
        "test_loss": metrics.test_loss,
        "test_mae": metrics.test_mae,
        "test_rmse": metrics.test_rmse,
        "test_mape": metrics.test_mape,
    }


def _serialize_horizon_baseline_result(result: HorizonBaselineResult) -> dict[str, Any]:
    return {
        "horizon": result.horizon,
        "val": {
            "no_text": _serialize_summary_metrics(result.val_no_text),
            "with_text": _serialize_summary_metrics(result.val_with_text),
            "pca": _serialize_summary_metrics(result.val_pca),
            "truncate": _serialize_summary_metrics(
                result.val_truncate,
                label=result.val_truncate_label,
            ),
        },
        "test": {
            "no_text": _serialize_summary_metrics(result.test_no_text),
            "with_text": _serialize_summary_metrics(result.test_with_text),
            "pca": _serialize_summary_metrics(result.test_pca),
            "truncate": _serialize_summary_metrics(
                result.test_truncate,
                label=result.test_truncate_label,
            ),
        },
    }


def _serialize_shared_baseline_result(shared_baseline: SharedBaselineResult) -> dict[str, Any]:
    return {
        "text_attn_layers": list(shared_baseline.text_attn_layers),
        "max_context": shared_baseline.max_context,
        "val": {
            "no_text": _serialize_summary_metrics(shared_baseline.val_no_text),
            "with_text": _serialize_summary_metrics(shared_baseline.val_with_text),
            "pca": _serialize_summary_metrics(shared_baseline.val_pca),
            "truncate": _serialize_summary_metrics(
                shared_baseline.val_truncate,
                label=shared_baseline.val_truncate_label,
            ),
        },
        "test": {
            "no_text": _serialize_summary_metrics(shared_baseline.test_no_text),
            "with_text": _serialize_summary_metrics(shared_baseline.test_with_text),
            "pca": _serialize_summary_metrics(shared_baseline.test_pca),
            "truncate": _serialize_summary_metrics(
                shared_baseline.test_truncate,
                label=shared_baseline.test_truncate_label,
            ),
        },
        "per_horizon": (
            None
            if shared_baseline.per_horizon is None
            else [_serialize_horizon_baseline_result(result) for result in shared_baseline.per_horizon]
        ),
    }


def _baseline_eval_inputs(
    prepared_trial: PreparedFineTuneTrial,
    *,
    split_name: str,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    splits = prepared_trial.splits
    if split_name == "val":
        return (
            np.concatenate((prepared_trial.X_context_proc, prepared_trial.X_train_proc)),
            np.concatenate((splits.y_context, splits.y_train)),
            np.concatenate((splits.text_context, splits.text_train), axis=0),
            prepared_trial.X_val_proc,
            splits.y_val,
            splits.text_val,
        )
    if split_name == "test":
        return (
            np.concatenate((prepared_trial.X_context_proc, prepared_trial.X_train_proc, prepared_trial.X_val_proc)),
            np.concatenate((splits.y_context, splits.y_train, splits.y_val)),
            np.concatenate((splits.text_context, splits.text_train, splits.text_val), axis=0),
            prepared_trial.X_test_proc,
            splits.y_test,
            splits.text_test,
        )
    raise ValueError(f"Unsupported split_name: {split_name!r}. Expected 'val' or 'test'.")


def run_shared_baseline_evaluation(
    base_run_cfg: DataConfig,
    *,
    data_splits: FineTuneDataSplits,
    logs_dir: Path,
) -> SharedBaselineResult:
    baseline_log_path = logs_dir / "baseline.log"
    with baseline_log_path.open("w", encoding="utf-8") as log_file:
        tee_writer = _TeeWriter(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee_writer), contextlib.redirect_stderr(tee_writer):
            print(
                "\n== Shared HPO Baseline ==\n"
                f"Config | text_attn_layers={base_run_cfg.model.text_attn_layers} | "
                f"max_context={base_run_cfg.tuning.max_context}"
            )
            print(
                "This baseline runs once from the base fine-tune config and is attached to the HPO summary."
            )
            per_horizon_results: list[HorizonBaselineResult] = []
            for horizon in range(1, base_run_cfg.prediction_window + 1):
                print(f"\n-- Shared baseline horizon {horizon}/{base_run_cfg.prediction_window} --")
                set_random_seeds(base_run_cfg.seed)
                prepared_trial = prepare_fine_tune_trial(
                    base_run_cfg,
                    data_splits=data_splits,
                    horizon=horizon,
                )

                val_inputs = _baseline_eval_inputs(prepared_trial, split_name="val")
                val_no_text = _metrics_from_triplet(
                    evaluate_rolling(
                        prepared_trial.reg,
                        X_context_proc=val_inputs[0],
                        y_context=val_inputs[1],
                        text_context=val_inputs[2],
                        X_eval_proc=val_inputs[3],
                        y_eval=val_inputs[4],
                        text_eval=val_inputs[5],
                        use_text=False,
                        label="Baseline val (no text attn)",
                        max_context=base_run_cfg.tuning.max_context,
                        horizon=horizon,
                    )
                )
                val_with_text = _metrics_from_triplet(
                    evaluate_rolling(
                        prepared_trial.reg,
                        X_context_proc=val_inputs[0],
                        y_context=val_inputs[1],
                        text_context=val_inputs[2],
                        X_eval_proc=val_inputs[3],
                        y_eval=val_inputs[4],
                        text_eval=val_inputs[5],
                        use_text=True,
                        label="Baseline val (with text attn)",
                        max_context=base_run_cfg.tuning.max_context,
                        horizon=horizon,
                    )
                )
                val_pca = _metrics_from_triplet(
                    evaluate_rolling_pca(
                        prepared_trial.reg,
                        X_context_proc=val_inputs[0],
                        y_context=val_inputs[1],
                        text_context=val_inputs[2],
                        X_eval_proc=val_inputs[3],
                        y_eval=val_inputs[4],
                        text_eval=val_inputs[5],
                        label="Baseline val (PCA)",
                        max_context=base_run_cfg.tuning.max_context,
                        horizon=horizon,
                    )
                )
                val_truncate_result = evaluate_rolling_truncate_text(
                    prepared_trial.reg,
                    X_context_proc=val_inputs[0],
                    y_context=val_inputs[1],
                    text_context=val_inputs[2],
                    X_eval_proc=val_inputs[3],
                    y_eval=val_inputs[4],
                    text_eval=val_inputs[5],
                    label="Baseline val (text truncate)",
                    max_context=base_run_cfg.tuning.max_context,
                    horizon=horizon,
                )

                test_inputs = _baseline_eval_inputs(prepared_trial, split_name="test")
                test_no_text = _metrics_from_triplet(
                    evaluate_rolling(
                        prepared_trial.reg,
                        X_context_proc=test_inputs[0],
                        y_context=test_inputs[1],
                        text_context=test_inputs[2],
                        X_eval_proc=test_inputs[3],
                        y_eval=test_inputs[4],
                        text_eval=test_inputs[5],
                        use_text=False,
                        label="Baseline test (no text attn)",
                        max_context=base_run_cfg.tuning.max_context,
                        horizon=horizon,
                    )
                )
                test_with_text = _metrics_from_triplet(
                    evaluate_rolling(
                        prepared_trial.reg,
                        X_context_proc=test_inputs[0],
                        y_context=test_inputs[1],
                        text_context=test_inputs[2],
                        X_eval_proc=test_inputs[3],
                        y_eval=test_inputs[4],
                        text_eval=test_inputs[5],
                        use_text=True,
                        label="Baseline test (with text attn)",
                        max_context=base_run_cfg.tuning.max_context,
                        horizon=horizon,
                    )
                )
                test_pca = _metrics_from_triplet(
                    evaluate_rolling_pca(
                        prepared_trial.reg,
                        X_context_proc=test_inputs[0],
                        y_context=test_inputs[1],
                        text_context=test_inputs[2],
                        X_eval_proc=test_inputs[3],
                        y_eval=test_inputs[4],
                        text_eval=test_inputs[5],
                        label="Baseline test (PCA)",
                        max_context=base_run_cfg.tuning.max_context,
                        horizon=horizon,
                    )
                )
                test_truncate_result = evaluate_rolling_truncate_text(
                    prepared_trial.reg,
                    X_context_proc=test_inputs[0],
                    y_context=test_inputs[1],
                    text_context=test_inputs[2],
                    X_eval_proc=test_inputs[3],
                    y_eval=test_inputs[4],
                    text_eval=test_inputs[5],
                    label="Baseline test (text truncate)",
                    max_context=base_run_cfg.tuning.max_context,
                    horizon=horizon,
                )

                val_truncate = None
                val_truncate_label = None
                if val_truncate_result is not None:
                    val_truncate = _metrics_from_triplet(val_truncate_result[0])
                    val_truncate_label = val_truncate_result[1]

                test_truncate = None
                test_truncate_label = None
                if test_truncate_result is not None:
                    test_truncate = _metrics_from_triplet(test_truncate_result[0])
                    test_truncate_label = test_truncate_result[1]

                per_horizon_results.append(
                    HorizonBaselineResult(
                        horizon=horizon,
                        val_no_text=val_no_text,
                        val_with_text=val_with_text,
                        val_pca=val_pca,
                        val_truncate=val_truncate,
                        val_truncate_label=val_truncate_label,
                        test_no_text=test_no_text,
                        test_with_text=test_with_text,
                        test_pca=test_pca,
                        test_truncate=test_truncate,
                        test_truncate_label=test_truncate_label,
                    )
                )

    return SharedBaselineResult(
        text_attn_layers=list(base_run_cfg.model.text_attn_layers),
        max_context=base_run_cfg.tuning.max_context,
        val_no_text=_mean_summary_metrics([result.val_no_text for result in per_horizon_results]),
        val_with_text=_mean_summary_metrics([result.val_with_text for result in per_horizon_results]),
        val_pca=_mean_summary_metrics([result.val_pca for result in per_horizon_results]),
        val_truncate=(
            None
            if any(result.val_truncate is None for result in per_horizon_results)
            else _mean_summary_metrics([result.val_truncate for result in per_horizon_results if result.val_truncate is not None])
        ),
        val_truncate_label=next(
            (result.val_truncate_label for result in per_horizon_results if result.val_truncate_label is not None),
            None,
        ),
        test_no_text=_mean_summary_metrics([result.test_no_text for result in per_horizon_results]),
        test_with_text=_mean_summary_metrics([result.test_with_text for result in per_horizon_results]),
        test_pca=_mean_summary_metrics([result.test_pca for result in per_horizon_results]),
        test_truncate=(
            None
            if any(result.test_truncate is None for result in per_horizon_results)
            else _mean_summary_metrics([result.test_truncate for result in per_horizon_results if result.test_truncate is not None])
        ),
        test_truncate_label=next(
            (result.test_truncate_label for result in per_horizon_results if result.test_truncate_label is not None),
            None,
        ),
        per_horizon=per_horizon_results,
    )


def build_search_choices(
    base_run_cfg: DataConfig,
    search_space_cfg: SearchSpaceConfig,
) -> dict[str, list[object]]:
    return {
        "text_attn_layers": (
            [list(choice) for choice in search_space_cfg.text_attn_layers]
            if search_space_cfg.text_attn_layers is not None
            else [list(base_run_cfg.model.text_attn_layers)]
        ),
        "epochs": search_space_cfg.epochs if search_space_cfg.epochs is not None else [base_run_cfg.tuning.epochs],
        "gate_lr": search_space_cfg.gate_lr if search_space_cfg.gate_lr is not None else [base_run_cfg.tuning.gate_lr],
        "text_attn_lr": (
            search_space_cfg.text_attn_lr
            if search_space_cfg.text_attn_lr is not None
            else [base_run_cfg.tuning.text_attn_lr]
        ),
        "gate_logit_clamp": (
            search_space_cfg.gate_logit_clamp
            if search_space_cfg.gate_logit_clamp is not None
            else [base_run_cfg.tuning.gate_logit_clamp]
        ),
        "tune_batch_size": (
            search_space_cfg.tune_batch_size
            if search_space_cfg.tune_batch_size is not None
            else [base_run_cfg.tuning.tune_batch_size]
        ),
        "max_context": search_space_cfg.max_context if search_space_cfg.max_context is not None else [base_run_cfg.tuning.max_context],
    }


def count_search_combinations(search_choices: dict[str, list[object]]) -> int:
    total = 1
    for field_name in SEARCH_FIELD_NAMES:
        total *= len(search_choices[field_name])
    return total


def _decode_trial_spec(
    search_choices: dict[str, list[object]],
    *,
    flat_index: int,
) -> RandomSearchTrialSpec:
    resolved_values: dict[str, object] = {}
    remaining = flat_index
    for field_name in reversed(SEARCH_FIELD_NAMES):
        field_choices = search_choices[field_name]
        remaining, choice_idx = divmod(remaining, len(field_choices))
        choice = field_choices[choice_idx]
        if field_name == "text_attn_layers":
            choice = list(choice)
        resolved_values[field_name] = choice
    return RandomSearchTrialSpec(
        **{field_name: resolved_values[field_name] for field_name in SEARCH_FIELD_NAMES}
    )


def sample_trial_specs(
    base_run_cfg: DataConfig,
    random_search_cfg: RandomSearchConfig,
) -> tuple[list[RandomSearchTrialSpec], int]:
    search_choices = build_search_choices(base_run_cfg, random_search_cfg.search_space)
    total_combinations = count_search_combinations(search_choices)
    sampled_count = min(random_search_cfg.trials, total_combinations)
    sampled_indices = random.Random(random_search_cfg.seed).sample(range(total_combinations), k=sampled_count)
    return [
        _decode_trial_spec(search_choices, flat_index=flat_index)
        for flat_index in sampled_indices
    ], total_combinations


def resolve_trial_config(
    base_run_cfg: DataConfig,
    trial_spec: RandomSearchTrialSpec,
) -> DataConfig:
    updated_model_cfg = replace(
        base_run_cfg.model,
        text_attn_layers=list(trial_spec.text_attn_layers),
    )
    updated_tuning_cfg = replace(
        base_run_cfg.tuning,
        epochs=trial_spec.epochs,
        gate_lr=trial_spec.gate_lr,
        text_attn_lr=trial_spec.text_attn_lr,
        gate_logit_clamp=trial_spec.gate_logit_clamp,
        tune_batch_size=trial_spec.tune_batch_size,
        max_context=trial_spec.max_context,
    )
    return replace(base_run_cfg, model=updated_model_cfg, tuning=updated_tuning_cfg)


def objective_score(metric_name: str, metrics: RollingMetrics) -> float:
    metric_name = metric_name.lower()
    if metric_name == "mae":
        return metrics.mae
    if metric_name == "rmse":
        return metrics.rmse
    if metric_name == "mape":
        return metrics.mape
    raise ValueError(
        f"Unsupported early_stopping_metric: {metric_name!r}. Expected one of 'mae', 'rmse', or 'mape'."
    )


def trial_sort_key(result: RandomSearchTrialResult) -> tuple[float, float, int]:
    return (result.ranking_score, result.val_mae, result.trial_index)


def execute_random_search_trial(
    trial_cfg: DataConfig,
    *,
    trial_index: int,
    data_splits: FineTuneDataSplits,
) -> tuple[RandomSearchTrialResult, dict[int, dict[str, Any]]]:
    print(
        f"\n== Random Search Trial {trial_index:02d} ==\n"
        f"Config | text_attn_layers={trial_cfg.model.text_attn_layers} | "
        f"epochs={trial_cfg.tuning.epochs} | gate_lr={trial_cfg.tuning.gate_lr} | "
        f"text_attn_lr={trial_cfg.tuning.text_attn_lr} | "
        f"gate_logit_clamp={trial_cfg.tuning.gate_logit_clamp} | "
        f"tune_batch_size={trial_cfg.tuning.tune_batch_size} | "
        f"max_context={trial_cfg.tuning.max_context} | "
        f"prediction_window={trial_cfg.prediction_window}"
    )
    print("Per-trial baseline evaluations are skipped; shared HPO baseline is evaluated once separately.")
    horizon_metrics: list[HorizonTrialMetrics] = []
    checkpoint_bundle: dict[int, dict[str, Any]] = {}
    val_metrics_by_horizon: list[RollingMetrics] = []

    for horizon in range(1, trial_cfg.prediction_window + 1):
        print(f"\n-- Trial {trial_index:02d} horizon {horizon}/{trial_cfg.prediction_window} --")
        set_random_seeds(trial_cfg.seed)
        prepared_trial = prepare_fine_tune_trial(
            trial_cfg,
            data_splits=data_splits,
            horizon=horizon,
        )
        fine_tune_outcome = fine_tune_prepared_trial(prepared_trial, tuning_cfg=trial_cfg.tuning)
        val_metrics = evaluate_prepared_split(
            prepared_trial,
            split_name="val",
            use_text=True,
            tuning_cfg=trial_cfg.tuning,
        )
        val_metrics_by_horizon.append(val_metrics)
        horizon_metrics.append(
            HorizonTrialMetrics(
                horizon=horizon,
                best_epoch=fine_tune_outcome.best_epoch,
                val_loss=val_metrics.loss,
                val_mae=val_metrics.mae,
                val_rmse=val_metrics.rmse,
                val_mape=val_metrics.mape,
            )
        )
        checkpoint_bundle[horizon] = _clone_state_dict_to_cpu(prepared_trial.reg.model.state_dict())
        print(
            f"Horizon {horizon:02d} complete | best_epoch={fine_tune_outcome.best_epoch} | "
            f"val_mae={val_metrics.mae:.6f} | val_rmse={val_metrics.rmse:.6f} | "
            f"val_mape={val_metrics.mape:.6f}%"
        )

    aggregate_val_metrics = _mean_rolling_metrics(val_metrics_by_horizon)
    ranking_score = objective_score(trial_cfg.tuning.early_stopping_metric, aggregate_val_metrics)
    result = RandomSearchTrialResult(
        trial_index=trial_index,
        text_attn_layers=list(trial_cfg.model.text_attn_layers),
        epochs=trial_cfg.tuning.epochs,
        gate_lr=trial_cfg.tuning.gate_lr,
        text_attn_lr=trial_cfg.tuning.text_attn_lr,
        gate_logit_clamp=trial_cfg.tuning.gate_logit_clamp,
        tune_batch_size=trial_cfg.tuning.tune_batch_size,
        max_context=trial_cfg.tuning.max_context,
        val_loss=aggregate_val_metrics.loss,
        val_mae=aggregate_val_metrics.mae,
        val_rmse=aggregate_val_metrics.rmse,
        val_mape=aggregate_val_metrics.mape,
        ranking_score=ranking_score,
        best_epoch=(
            horizon_metrics[0].best_epoch
            if trial_cfg.prediction_window == 1
            else None
        ),
        horizon_metrics=horizon_metrics,
    )
    print(
        f"Trial {trial_index:02d} complete | objective={ranking_score:.6f} | "
        f"val_mae={aggregate_val_metrics.mae:.6f} | val_rmse={aggregate_val_metrics.rmse:.6f} | "
        f"val_mape={aggregate_val_metrics.mape:.6f}%"
    )
    return result, checkpoint_bundle


def _extract_gpu_id_from_device(device: str | None) -> int | None:
    if not isinstance(device, str) or not device.startswith("cuda:"):
        return None
    _, _, gpu_id = device.partition(":")
    if not gpu_id.isdigit():
        raise ValueError(f"Unsupported CUDA device specifier: {device!r}")
    return int(gpu_id)


def _build_parallel_device_slots(
    *,
    max_parallel_trials: int,
    gpu_ids: list[int] | None,
    base_device: str | None,
) -> list[str | None]:
    if max_parallel_trials <= 0:
        raise ValueError("max_parallel_trials must be positive.")

    normalized_base_device = base_device.lower() if isinstance(base_device, str) else None
    if normalized_base_device == "cpu":
        return ["cpu"] * max_parallel_trials

    resolved_gpu_ids = list(gpu_ids) if gpu_ids is not None else None
    if resolved_gpu_ids is None:
        explicit_gpu_id = _extract_gpu_id_from_device(base_device)
        if explicit_gpu_id is not None:
            resolved_gpu_ids = [explicit_gpu_id]
        elif normalized_base_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Parallel GPU execution was requested with model.device='cuda', "
                    "but no CUDA devices are visible to PyTorch."
                )
            resolved_gpu_ids = list(range(torch.cuda.device_count()))
        elif torch.cuda.is_available():
            resolved_gpu_ids = list(range(torch.cuda.device_count()))

    if resolved_gpu_ids:
        visible_gpu_count = torch.cuda.device_count()
        if visible_gpu_count == 0:
            raise RuntimeError(
                "Parallel GPU execution was requested, but no CUDA devices are visible to PyTorch."
            )
        invalid_gpu_ids = [gpu_id for gpu_id in resolved_gpu_ids if gpu_id >= visible_gpu_count]
        if invalid_gpu_ids:
            raise ValueError(
                "Configured gpu_ids exceed the visible CUDA device count "
                f"({visible_gpu_count}): {invalid_gpu_ids}"
            )
        return [
            f"cuda:{resolved_gpu_ids[slot_index % len(resolved_gpu_ids)]}"
            for slot_index in range(max_parallel_trials)
        ]

    fallback_device = base_device if base_device not in {None, "auto"} else None
    return [fallback_device] * max_parallel_trials


def _execute_trial_with_logging(
    trial_cfg: DataConfig,
    *,
    trial_index: int,
    data_splits: FineTuneDataSplits,
    trial_log_path: str | Path,
    checkpoint_path: str | Path,
    assigned_device: str | None,
    tee_stdout: bool,
) -> TrialExecutionArtifacts:
    resolved_trial_cfg = trial_cfg
    if assigned_device is not None and assigned_device != trial_cfg.model.device:
        resolved_trial_cfg = replace(
            trial_cfg,
            model=replace(trial_cfg.model, device=assigned_device),
        )

    trial_log_path = Path(trial_log_path)
    checkpoint_path = Path(checkpoint_path)
    with trial_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        streams: tuple[Any, ...] = (sys.stdout, log_file) if tee_stdout else (log_file,)
        tee_writer = _TeeWriter(*streams)
        with contextlib.redirect_stdout(tee_writer), contextlib.redirect_stderr(tee_writer):
            if assigned_device is not None:
                print(f"Assigned device | {assigned_device}")
            result, checkpoint_bundle = execute_random_search_trial(
                resolved_trial_cfg,
                trial_index=trial_index,
                data_splits=data_splits,
            )

    torch.save(checkpoint_bundle, checkpoint_path)
    return TrialExecutionArtifacts(
        result=result,
        checkpoint_path=str(checkpoint_path),
        assigned_device=assigned_device,
    )


def _run_trial_batch(
    trial_cfgs: dict[int, DataConfig],
    *,
    data_splits: FineTuneDataSplits,
    logs_dir: Path,
    checkpoints_dir: Path,
    max_parallel_trials: int,
    gpu_ids: list[int] | None,
) -> tuple[list[RandomSearchTrialResult], dict[int, str]]:
    trial_items = list(trial_cfgs.items())
    if not trial_items:
        return [], {}

    if max_parallel_trials <= 1:
        results: list[RandomSearchTrialResult] = []
        checkpoint_paths: dict[int, str] = {}
        for trial_index, trial_cfg in trial_items:
            trial_log_path = logs_dir / f"trial_{trial_index:03d}.log"
            checkpoint_path = checkpoints_dir / f"trial_{trial_index:03d}.pt"
            artifacts = _execute_trial_with_logging(
                trial_cfg,
                trial_index=trial_index,
                data_splits=data_splits,
                trial_log_path=trial_log_path,
                checkpoint_path=checkpoint_path,
                assigned_device=None,
                tee_stdout=True,
            )
            results.append(artifacts.result)
            checkpoint_paths[trial_index] = artifacts.checkpoint_path
        return results, checkpoint_paths

    device_slots = _build_parallel_device_slots(
        max_parallel_trials=max_parallel_trials,
        gpu_ids=gpu_ids,
        base_device=trial_items[0][1].model.device,
    )
    print(
        "Parallel trial execution enabled | "
        f"max_parallel_trials={max_parallel_trials} | "
        f"device_slots={device_slots}"
    )

    results = []
    checkpoint_paths = {}
    mp_context = multiprocessing.get_context("spawn")
    trial_iter = iter(trial_items)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(device_slots),
        mp_context=mp_context,
    ) as executor:
        future_states: dict[concurrent.futures.Future[TrialExecutionArtifacts], tuple[int, str | None]] = {}

        def submit_next(assigned_device: str | None) -> bool:
            try:
                next_trial_index, next_trial_cfg = next(trial_iter)
            except StopIteration:
                return False

            future = executor.submit(
                _execute_trial_with_logging,
                next_trial_cfg,
                trial_index=next_trial_index,
                data_splits=data_splits,
                trial_log_path=str(logs_dir / f"trial_{next_trial_index:03d}.log"),
                checkpoint_path=str(checkpoints_dir / f"trial_{next_trial_index:03d}.pt"),
                assigned_device=assigned_device,
                tee_stdout=False,
            )
            future_states[future] = (next_trial_index, assigned_device)
            print(
                f"Started trial {next_trial_index:03d} | "
                f"device={assigned_device or next_trial_cfg.model.device or 'auto'}"
            )
            return True

        for assigned_device in device_slots:
            if not submit_next(assigned_device):
                break

        while future_states:
            done, _ = concurrent.futures.wait(
                tuple(future_states),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                submitted_trial_index, assigned_device = future_states.pop(future)
                artifacts = future.result()
                result = artifacts.result
                results.append(result)
                checkpoint_paths[result.trial_index] = artifacts.checkpoint_path
                print(
                    f"Completed trial {result.trial_index:03d} | "
                    f"device={artifacts.assigned_device or assigned_device or 'auto'} | "
                    f"objective={result.ranking_score:.6f} | val_mae={result.val_mae:.6f}"
                )
                if result.trial_index != submitted_trial_index:
                    raise RuntimeError(
                        "Parallel trial bookkeeping mismatch: "
                        f"submitted trial {submitted_trial_index} returned {result.trial_index}."
                    )
                submit_next(assigned_device)

    return results, checkpoint_paths


def _serialize_trial_result(
    result: RandomSearchTrialResult,
    *,
    rank: int,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "trial_index": result.trial_index,
        "best_epoch": result.best_epoch,
        "text_attn_layers": result.text_attn_layers,
        "epochs": result.epochs,
        "gate_lr": result.gate_lr,
        "text_attn_lr": result.text_attn_lr,
        "gate_logit_clamp": result.gate_logit_clamp,
        "tune_batch_size": result.tune_batch_size,
        "max_context": result.max_context,
        "val_loss": result.val_loss,
        "val_mae": result.val_mae,
        "val_rmse": result.val_rmse,
        "val_mape": result.val_mape,
        "ranking_score": result.ranking_score,
        "test_loss": result.test_loss,
        "test_mae": result.test_mae,
        "test_rmse": result.test_rmse,
        "test_mape": result.test_mape,
    }


def _write_trial_results_csv(
    output_dir: Path,
    ranked_results: list[RandomSearchTrialResult],
) -> None:
    csv_path = output_dir / "trial_results.csv"
    fieldnames = [
        "rank",
        "trial_index",
        "best_epoch",
        "text_attn_layers",
        "epochs",
        "gate_lr",
        "text_attn_lr",
        "gate_logit_clamp",
        "tune_batch_size",
        "max_context",
        "val_loss",
        "val_mae",
        "val_rmse",
        "val_mape",
        "ranking_score",
        "test_loss",
        "test_mae",
        "test_rmse",
        "test_mape",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(ranked_results, start=1):
            row = _serialize_trial_result(result, rank=rank)
            row["text_attn_layers"] = json.dumps(row["text_attn_layers"])
            writer.writerow(row)


def _write_best_fine_tune_config(
    output_dir: Path,
    *,
    best_run_cfg: DataConfig,
) -> None:
    output_path = output_dir / "best_fine_tune_config.yaml"
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(best_run_cfg), f, sort_keys=False)


def _append_test_metrics_to_trial_log(
    logs_dir: Path,
    *,
    trial_index: int,
    rank: int,
    metrics: RollingMetrics,
    horizon_metrics: list[HorizonTrialMetrics] | None = None,
) -> None:
    trial_log_path = logs_dir / f"trial_{trial_index:03d}.log"
    with trial_log_path.open("a", encoding="utf-8") as f:
        f.write(
            "\n== Test Evaluation For Ranked Trial ==\n"
            f"rank={rank} | trial_index={trial_index} | "
            f"test_loss={metrics.loss:.6f} | test_mae={metrics.mae:.6f} | "
            f"test_rmse={metrics.rmse:.6f} | test_mape={metrics.mape:.6f}%\n"
        )
        if horizon_metrics is not None:
            for horizon_metric in horizon_metrics:
                f.write(
                    f"horizon={horizon_metric.horizon} | best_epoch={horizon_metric.best_epoch} | "
                    f"test_loss={horizon_metric.test_loss:.6f} | "
                    f"test_mae={horizon_metric.test_mae:.6f} | "
                    f"test_rmse={horizon_metric.test_rmse:.6f} | "
                    f"test_mape={horizon_metric.test_mape:.6f}%\n"
                )


def _write_summary_json(
    output_dir: Path,
    *,
    random_search_cfg: RandomSearchConfig,
    objective_metric: str,
    total_combinations: int,
    ranked_results: list[RandomSearchTrialResult],
    shared_baseline: SharedBaselineResult,
) -> None:
    top_limit = min(random_search_cfg.top_k, len(ranked_results))
    best_trial = ranked_results[0]
    summary = {
        "base_config": random_search_cfg.base_config,
        "base_dataset": random_search_cfg.base_dataset,
        "objective_metric": objective_metric,
        "max_parallel_trials": random_search_cfg.max_parallel_trials,
        "gpu_ids": random_search_cfg.gpu_ids,
        "baseline_evaluations_skipped_per_trial": True,
        "shared_baseline_evaluated": True,
        "shared_baseline": _serialize_shared_baseline_result(shared_baseline),
        "requested_trials": random_search_cfg.trials,
        "evaluated_trials": len(ranked_results),
        "total_combinations": total_combinations,
        "top_k": top_limit,
        "top_k_test_evaluated": top_limit,
        "best_trial": {
            **_serialize_trial_result(best_trial, rank=1),
            "horizon_metrics": (
                None
                if best_trial.horizon_metrics is None
                else [_serialize_horizon_trial_metrics(metrics) for metrics in best_trial.horizon_metrics]
            ),
        },
        "top_trials": [
            {
                **_serialize_trial_result(result, rank=rank),
                "horizon_metrics": (
                    None
                    if result.horizon_metrics is None
                    else [_serialize_horizon_trial_metrics(metrics) for metrics in result.horizon_metrics]
                ),
            }
            for rank, result in enumerate(ranked_results[:top_limit], start=1)
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _write_summary_log(
    output_dir: Path,
    *,
    random_search_cfg: RandomSearchConfig,
    objective_metric: str,
    total_combinations: int,
    ranked_results: list[RandomSearchTrialResult],
    shared_baseline: SharedBaselineResult,
) -> None:
    top_limit = min(random_search_cfg.top_k, len(ranked_results))
    best_trial = ranked_results[0]
    lines = [
        "Random Search Summary",
        f"base_config: {random_search_cfg.base_config}",
        f"base_dataset: {random_search_cfg.base_dataset}",
        f"objective_metric: {objective_metric}",
        f"max_parallel_trials: {random_search_cfg.max_parallel_trials}",
        f"gpu_ids: {random_search_cfg.gpu_ids}",
        "baseline_evaluations_skipped_per_trial: true",
        "shared_baseline_evaluated: true",
        f"requested_trials: {random_search_cfg.trials}",
        f"evaluated_trials: {len(ranked_results)}",
        f"total_combinations: {total_combinations}",
        f"top_k_test_evaluated: {top_limit}",
        "",
        "Shared Baseline",
        (
            f"params | text_attn_layers={shared_baseline.text_attn_layers} | "
            f"max_context={shared_baseline.max_context}"
        ),
        f"prediction_window={len(shared_baseline.per_horizon) if shared_baseline.per_horizon is not None else 1}",
        (
            f"val_no_text | mae={shared_baseline.val_no_text.mae:.6f} | "
            f"rmse={shared_baseline.val_no_text.rmse:.6f} | "
            f"mape={shared_baseline.val_no_text.mape:.6f}%"
        ),
        (
            f"val_with_text | mae={shared_baseline.val_with_text.mae:.6f} | "
            f"rmse={shared_baseline.val_with_text.rmse:.6f} | "
            f"mape={shared_baseline.val_with_text.mape:.6f}%"
        ),
        (
            f"val_pca | mae={shared_baseline.val_pca.mae:.6f} | "
            f"rmse={shared_baseline.val_pca.rmse:.6f} | "
            f"mape={shared_baseline.val_pca.mape:.6f}%"
        ),
        (
            "val_truncate | unavailable"
            if shared_baseline.val_truncate is None
            else (
                f"val_truncate | label={shared_baseline.val_truncate_label} | "
                f"mae={shared_baseline.val_truncate.mae:.6f} | "
                f"rmse={shared_baseline.val_truncate.rmse:.6f} | "
                f"mape={shared_baseline.val_truncate.mape:.6f}%"
            )
        ),
        (
            f"test_no_text | mae={shared_baseline.test_no_text.mae:.6f} | "
            f"rmse={shared_baseline.test_no_text.rmse:.6f} | "
            f"mape={shared_baseline.test_no_text.mape:.6f}%"
        ),
        (
            f"test_with_text | mae={shared_baseline.test_with_text.mae:.6f} | "
            f"rmse={shared_baseline.test_with_text.rmse:.6f} | "
            f"mape={shared_baseline.test_with_text.mape:.6f}%"
        ),
        (
            f"test_pca | mae={shared_baseline.test_pca.mae:.6f} | "
            f"rmse={shared_baseline.test_pca.rmse:.6f} | "
            f"mape={shared_baseline.test_pca.mape:.6f}%"
        ),
        (
            "test_truncate | unavailable"
            if shared_baseline.test_truncate is None
            else (
                f"test_truncate | label={shared_baseline.test_truncate_label} | "
                f"mae={shared_baseline.test_truncate.mae:.6f} | "
                f"rmse={shared_baseline.test_truncate.rmse:.6f} | "
                f"mape={shared_baseline.test_truncate.mape:.6f}%"
            )
        ),
        "",
        "Best Trial",
        (
            f"rank=1 | trial_index={best_trial.trial_index} | "
            f"best_epoch={best_trial.best_epoch} | "
            f"objective={best_trial.ranking_score:.6f} | "
            f"val_mae={best_trial.val_mae:.6f} | "
            f"val_rmse={best_trial.val_rmse:.6f} | "
            f"val_mape={best_trial.val_mape:.6f}% | "
            f"test_mae={best_trial.test_mae:.6f} | "
            f"test_rmse={best_trial.test_rmse:.6f} | "
            f"test_mape={best_trial.test_mape:.6f}%"
        ),
        (
            f"params | text_attn_layers={best_trial.text_attn_layers} | "
            f"epochs={best_trial.epochs} | gate_lr={best_trial.gate_lr} | "
            f"text_attn_lr={best_trial.text_attn_lr} | "
            f"gate_logit_clamp={best_trial.gate_logit_clamp} | "
            f"tune_batch_size={best_trial.tune_batch_size} | "
            f"max_context={best_trial.max_context}"
        ),
        "",
    ]
    if shared_baseline.per_horizon is not None:
        for horizon_result in shared_baseline.per_horizon:
            lines.append(
                (
                    f"baseline_horizon={horizon_result.horizon} | "
                    f"val_with_text_mae={horizon_result.val_with_text.mae:.6f} | "
                    f"val_with_text_rmse={horizon_result.val_with_text.rmse:.6f} | "
                    f"test_with_text_mae={horizon_result.test_with_text.mae:.6f} | "
                    f"test_with_text_rmse={horizon_result.test_with_text.rmse:.6f}"
                )
            )
    if best_trial.horizon_metrics is not None:
        for metrics in best_trial.horizon_metrics:
            lines.append(
                (
                    f"best_trial_horizon={metrics.horizon} | best_epoch={metrics.best_epoch} | "
                    f"val_mae={metrics.val_mae:.6f} | val_rmse={metrics.val_rmse:.6f} | "
                    f"test_mae={metrics.test_mae:.6f} | test_rmse={metrics.test_rmse:.6f}"
                )
            )
    lines.append(f"Top {top_limit} Trials")
    for rank, result in enumerate(ranked_results[:top_limit], start=1):
        lines.append(
            (
                f"{rank}. trial_index={result.trial_index} | best_epoch={result.best_epoch} | "
                f"objective={result.ranking_score:.6f} | "
                f"val_mae={result.val_mae:.6f} | val_rmse={result.val_rmse:.6f} | "
                f"val_mape={result.val_mape:.6f}% | "
                f"test_mae={result.test_mae:.6f} | test_rmse={result.test_rmse:.6f} | "
                f"test_mape={result.test_mape:.6f}% | text_attn_layers={result.text_attn_layers} | "
                f"epochs={result.epochs} | gate_lr={result.gate_lr} | "
                f"text_attn_lr={result.text_attn_lr} | gate_logit_clamp={result.gate_logit_clamp} | "
                f"tune_batch_size={result.tune_batch_size} | max_context={result.max_context}"
            )
        )
        if result.horizon_metrics is not None:
            for metrics in result.horizon_metrics:
                lines.append(
                    (
                        f"trial_index={result.trial_index} horizon={metrics.horizon} | "
                        f"best_epoch={metrics.best_epoch} | val_mae={metrics.val_mae:.6f} | "
                        f"val_rmse={metrics.val_rmse:.6f} | "
                        f"test_mae={metrics.test_mae:.6f} | test_rmse={metrics.test_rmse:.6f}"
                    )
                )

    with (output_dir / "summary.log").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_random_search(
    random_search_cfg: RandomSearchConfig,
    *,
    config_stem: str,
    output_root: str | Path | None = None,
    base_run_cfg: DataConfig | None = None,
    data_splits: FineTuneDataSplits | None = None,
) -> dict[str, Any]:
    if base_run_cfg is None:
        base_run_cfg = load_fine_tune_config(random_search_cfg.base_config, random_search_cfg.base_dataset)
    if data_splits is None:
        data_splits = load_and_split_fine_tune_data(base_run_cfg)

    print(
        "Random search setup | "
        f"base_config={random_search_cfg.base_config} | "
        f"dataset={random_search_cfg.base_dataset} | "
        f"objective={base_run_cfg.tuning.early_stopping_metric} | "
        f"prediction_window={base_run_cfg.prediction_window}"
    )
    print(
        "Split sizes: "
        f"context={len(data_splits.y_context)} train={len(data_splits.y_train)} "
        f"val={len(data_splits.y_val)} test={len(data_splits.y_test)}"
    )

    trial_specs, total_combinations = sample_trial_specs(base_run_cfg, random_search_cfg)
    if random_search_cfg.trials > total_combinations:
        print(
            f"Requested {random_search_cfg.trials} trials but search space has only "
            f"{total_combinations} combinations; clamping to {total_combinations}."
        )
    print(
        f"Random search space | total_combinations={total_combinations} | "
        f"running_trials={len(trial_specs)} | seed={random_search_cfg.seed}"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(output_root) if output_root is not None else Path("results") / "hpo"
    output_dir = output_base / f"{config_stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=False)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    shared_baseline = run_shared_baseline_evaluation(
        base_run_cfg,
        data_splits=data_splits,
        logs_dir=logs_dir,
    )

    trial_cfgs: dict[int, DataConfig] = {}
    for trial_index, trial_spec in enumerate(trial_specs, start=1):
        trial_cfgs[trial_index] = resolve_trial_config(base_run_cfg, trial_spec)

    results, checkpoint_paths = _run_trial_batch(
        trial_cfgs,
        data_splits=data_splits,
        logs_dir=logs_dir,
        checkpoints_dir=checkpoints_dir,
        max_parallel_trials=random_search_cfg.max_parallel_trials,
        gpu_ids=random_search_cfg.gpu_ids,
    )

    if not results:
        raise RuntimeError("Random search did not execute any trials.")

    ranked_results = sorted(results, key=trial_sort_key)
    best_result = ranked_results[0]
    best_run_cfg = trial_cfgs[best_result.trial_index]
    top_limit = min(random_search_cfg.top_k, len(ranked_results))
    print(f"\nEvaluating top {top_limit} ranked trial(s) on the test split...")
    ranked_results_with_test = list(ranked_results)
    ranked_index_by_trial = {result.trial_index: idx for idx, result in enumerate(ranked_results_with_test)}
    for rank, result in enumerate(ranked_results[:top_limit], start=1):
        trial_cfg = trial_cfgs[result.trial_index]
        checkpoint_bundle = torch.load(checkpoint_paths[result.trial_index], map_location="cpu")
        if not isinstance(checkpoint_bundle, dict):
            raise RuntimeError(
                f"Expected checkpoint bundle dict for trial {result.trial_index}, got {type(checkpoint_bundle)!r}."
            )

        horizon_test_metrics: list[RollingMetrics] = []
        updated_horizon_metrics: list[HorizonTrialMetrics] = []
        horizon_metrics_source = result.horizon_metrics or [
            HorizonTrialMetrics(
                horizon=1,
                best_epoch=result.best_epoch,
                val_loss=result.val_loss,
                val_mae=result.val_mae,
                val_rmse=result.val_rmse,
                val_mape=result.val_mape,
            )
        ]
        for horizon_metric in horizon_metrics_source:
            prepared_trial = prepare_fine_tune_trial(
                trial_cfg,
                data_splits=data_splits,
                horizon=horizon_metric.horizon,
            )
            prepared_trial.reg.model.load_state_dict(checkpoint_bundle[horizon_metric.horizon])
            test_metrics = evaluate_prepared_split(
                prepared_trial,
                split_name="test",
                use_text=True,
                tuning_cfg=trial_cfg.tuning,
            )
            horizon_test_metrics.append(test_metrics)
            updated_horizon_metrics.append(
                replace(
                    horizon_metric,
                    test_loss=test_metrics.loss,
                    test_mae=test_metrics.mae,
                    test_rmse=test_metrics.rmse,
                    test_mape=test_metrics.mape,
                )
            )

        test_metrics = _mean_rolling_metrics(horizon_test_metrics)
        ranked_results_with_test[ranked_index_by_trial[result.trial_index]] = replace(
            result,
            test_loss=test_metrics.loss,
            test_mae=test_metrics.mae,
            test_rmse=test_metrics.rmse,
            test_mape=test_metrics.mape,
            horizon_metrics=updated_horizon_metrics,
        )
        _append_test_metrics_to_trial_log(
            logs_dir,
            trial_index=result.trial_index,
            rank=rank,
            metrics=test_metrics,
            horizon_metrics=updated_horizon_metrics,
        )
        print(
            f"Test rank {rank} | trial_index={result.trial_index} | "
            f"test_mae={test_metrics.mae:.6f} | test_rmse={test_metrics.rmse:.6f} | "
            f"test_mape={test_metrics.mape:.6f}%"
        )
    ranked_results = ranked_results_with_test
    best_result = ranked_results[0]
    print(
        "\n== Random Search Best Trial ==\n"
        f"Rank 1 | trial_index={best_result.trial_index} | "
        f"best_epoch={best_result.best_epoch} | "
        f"objective={best_result.ranking_score:.6f} | "
        f"val_mae={best_result.val_mae:.6f} | "
        f"test_mae={best_result.test_mae:.6f}"
    )

    print(f"Top {top_limit} trials:")
    for rank, result in enumerate(ranked_results[:top_limit], start=1):
        print(
            f"{rank}. trial_index={result.trial_index} | best_epoch={result.best_epoch} | "
            f"objective={result.ranking_score:.6f} | "
            f"val_mae={result.val_mae:.6f} | test_mae={result.test_mae:.6f} | "
            f"text_attn_layers={result.text_attn_layers}"
        )
        if result.horizon_metrics is not None:
            for metrics in result.horizon_metrics:
                print(
                    f"   horizon={metrics.horizon} | best_epoch={metrics.best_epoch} | "
                    f"val_mae={metrics.val_mae:.6f} | test_mae={metrics.test_mae:.6f}"
                )

    _write_trial_results_csv(output_dir, ranked_results)
    _write_best_fine_tune_config(output_dir, best_run_cfg=best_run_cfg)
    _write_summary_json(
        output_dir,
        random_search_cfg=random_search_cfg,
        objective_metric=best_run_cfg.tuning.early_stopping_metric,
        total_combinations=total_combinations,
        ranked_results=ranked_results,
        shared_baseline=shared_baseline,
    )
    _write_summary_log(
        output_dir,
        random_search_cfg=random_search_cfg,
        objective_metric=best_run_cfg.tuning.early_stopping_metric,
        total_combinations=total_combinations,
        ranked_results=ranked_results,
        shared_baseline=shared_baseline,
    )
    return {
        "output_dir": str(output_dir),
        "logs_dir": str(logs_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "ranked_results": ranked_results,
        "shared_baseline": shared_baseline,
        "best_test_metrics": RollingMetrics(
            loss=best_result.test_loss,
            mae=best_result.test_mae,
            rmse=best_result.test_rmse,
            mape=best_result.test_mape,
        ),
        "best_run_cfg": best_run_cfg,
        "total_combinations": total_combinations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run random-search HPO over fine-tuning configs.")
    parser.add_argument("--config", required=True, help="Path to the random-search YAML config.")
    args = parser.parse_args()

    random_search_cfg = load_random_search_config(args.config)
    run_random_search(
        random_search_cfg,
        config_stem=Path(args.config).stem,
    )


if __name__ == "__main__":
    main()
