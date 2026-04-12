from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import random
import re
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


def _extract_best_epoch_from_trial_log(trial_log_path: Path) -> int | None:
    log_text = trial_log_path.read_text(encoding="utf-8")
    match = re.search(r"Restored best text-mixing params from epoch\s+(\d+)", log_text)
    if match is None:
        return None
    return int(match.group(1))


def _metrics_from_triplet(metrics: tuple[float, float, float]) -> SummaryMetrics:
    return SummaryMetrics(mae=metrics[0], rmse=metrics[1], mape=metrics[2])


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
            set_random_seeds(base_run_cfg.seed)
            prepared_trial = prepare_fine_tune_trial(base_run_cfg, data_splits=data_splits)

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

    return SharedBaselineResult(
        text_attn_layers=list(base_run_cfg.model.text_attn_layers),
        max_context=base_run_cfg.tuning.max_context,
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
) -> tuple[RandomSearchTrialResult, PreparedFineTuneTrial]:
    print(
        f"\n== Random Search Trial {trial_index:02d} ==\n"
        f"Config | text_attn_layers={trial_cfg.model.text_attn_layers} | "
        f"epochs={trial_cfg.tuning.epochs} | gate_lr={trial_cfg.tuning.gate_lr} | "
        f"text_attn_lr={trial_cfg.tuning.text_attn_lr} | "
        f"gate_logit_clamp={trial_cfg.tuning.gate_logit_clamp} | "
        f"tune_batch_size={trial_cfg.tuning.tune_batch_size} | "
        f"max_context={trial_cfg.tuning.max_context}"
    )
    print("Per-trial baseline evaluations are skipped; shared HPO baseline is evaluated once separately.")
    set_random_seeds(trial_cfg.seed)
    prepared_trial = prepare_fine_tune_trial(trial_cfg, data_splits=data_splits)
    fine_tune_prepared_trial(prepared_trial, tuning_cfg=trial_cfg.tuning)
    val_metrics = evaluate_prepared_split(
        prepared_trial,
        split_name="val",
        use_text=True,
        tuning_cfg=trial_cfg.tuning,
    )
    ranking_score = objective_score(trial_cfg.tuning.early_stopping_metric, val_metrics)
    result = RandomSearchTrialResult(
        trial_index=trial_index,
        text_attn_layers=list(trial_cfg.model.text_attn_layers),
        epochs=trial_cfg.tuning.epochs,
        gate_lr=trial_cfg.tuning.gate_lr,
        text_attn_lr=trial_cfg.tuning.text_attn_lr,
        gate_logit_clamp=trial_cfg.tuning.gate_logit_clamp,
        tune_batch_size=trial_cfg.tuning.tune_batch_size,
        max_context=trial_cfg.tuning.max_context,
        val_loss=val_metrics.loss,
        val_mae=val_metrics.mae,
        val_rmse=val_metrics.rmse,
        val_mape=val_metrics.mape,
        ranking_score=ranking_score,
    )
    print(
        f"Trial {trial_index:02d} complete | objective={ranking_score:.6f} | "
        f"val_mae={val_metrics.mae:.6f} | val_rmse={val_metrics.rmse:.6f} | "
        f"val_mape={val_metrics.mape:.6f}%"
    )
    return result, prepared_trial


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
) -> None:
    trial_log_path = logs_dir / f"trial_{trial_index:03d}.log"
    with trial_log_path.open("a", encoding="utf-8") as f:
        f.write(
            "\n== Test Evaluation For Ranked Trial ==\n"
            f"rank={rank} | trial_index={trial_index} | "
            f"test_loss={metrics.loss:.6f} | test_mae={metrics.mae:.6f} | "
            f"test_rmse={metrics.rmse:.6f} | test_mape={metrics.mape:.6f}%\n"
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
        "baseline_evaluations_skipped_per_trial": True,
        "shared_baseline_evaluated": True,
        "shared_baseline": _serialize_shared_baseline_result(shared_baseline),
        "requested_trials": random_search_cfg.trials,
        "evaluated_trials": len(ranked_results),
        "total_combinations": total_combinations,
        "top_k": top_limit,
        "top_k_test_evaluated": top_limit,
        "best_trial": _serialize_trial_result(best_trial, rank=1),
        "top_trials": [
            _serialize_trial_result(result, rank=rank)
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
        f"Top {top_limit} Trials",
    ]
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
        f"objective={base_run_cfg.tuning.early_stopping_metric}"
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
    shared_baseline = run_shared_baseline_evaluation(
        base_run_cfg,
        data_splits=data_splits,
        logs_dir=logs_dir,
    )

    best_result: RandomSearchTrialResult | None = None
    best_run_cfg: DataConfig | None = None
    results: list[RandomSearchTrialResult] = []
    trial_cfgs: dict[int, DataConfig] = {}
    tuned_state_dicts: dict[int, dict[str, Any]] = {}

    for trial_index, trial_spec in enumerate(trial_specs, start=1):
        trial_cfg = resolve_trial_config(base_run_cfg, trial_spec)
        trial_cfgs[trial_index] = trial_cfg
        trial_log_path = logs_dir / f"trial_{trial_index:03d}.log"
        with trial_log_path.open("w", encoding="utf-8") as log_file:
            tee_writer = _TeeWriter(sys.stdout, log_file)
            with contextlib.redirect_stdout(tee_writer), contextlib.redirect_stderr(tee_writer):
                result, prepared_trial = execute_random_search_trial(
                    trial_cfg,
                    trial_index=trial_index,
                    data_splits=data_splits,
                )
        result = replace(result, best_epoch=_extract_best_epoch_from_trial_log(trial_log_path))
        results.append(result)
        tuned_state_dicts[trial_index] = _clone_state_dict_to_cpu(prepared_trial.reg.model.state_dict())
        if best_result is None or trial_sort_key(result) < trial_sort_key(best_result):
            best_result = result
            best_run_cfg = trial_cfg

    if best_result is None or best_run_cfg is None:
        raise RuntimeError("Random search did not execute any trials.")

    ranked_results = sorted(results, key=trial_sort_key)
    top_limit = min(random_search_cfg.top_k, len(ranked_results))
    print(f"\nEvaluating top {top_limit} ranked trial(s) on the test split...")
    ranked_results_with_test = list(ranked_results)
    ranked_index_by_trial = {result.trial_index: idx for idx, result in enumerate(ranked_results_with_test)}
    for rank, result in enumerate(ranked_results[:top_limit], start=1):
        trial_cfg = trial_cfgs[result.trial_index]
        prepared_trial = prepare_fine_tune_trial(trial_cfg, data_splits=data_splits)
        prepared_trial.reg.model.load_state_dict(tuned_state_dicts[result.trial_index])
        test_metrics = evaluate_prepared_split(
            prepared_trial,
            split_name="test",
            use_text=True,
            tuning_cfg=trial_cfg.tuning,
        )
        ranked_results_with_test[ranked_index_by_trial[result.trial_index]] = replace(
            result,
            test_loss=test_metrics.loss,
            test_mae=test_metrics.mae,
            test_rmse=test_metrics.rmse,
            test_mape=test_metrics.mape,
        )
        _append_test_metrics_to_trial_log(
            logs_dir,
            trial_index=result.trial_index,
            rank=rank,
            metrics=test_metrics,
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
