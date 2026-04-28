from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import csv
import json
import multiprocessing
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
        evaluate_fresh_baseline_metric,
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
        evaluate_fresh_baseline_metric,
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
    "max_context",
    "target_lag_count",
    "covariate_lag_count",
    "embedding_lag_count",
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
    target_lag_count: int | None = None
    covariate_lag_count: int | None = None
    embedding_lag_count: int | None = None


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
    val_real_mae: float
    val_real_rmse: float
    val_real_mape: float
    ranking_score: float
    target_lag_count: int | None = None
    covariate_lag_count: int | None = None
    embedding_lag_count: int | None = None
    best_epoch: int | None = None
    test_loss: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_mape: float | None = None
    test_real_mae: float | None = None
    test_real_rmse: float | None = None
    test_real_mape: float | None = None
    horizon_metrics: list["HorizonTrialMetrics"] | None = None


@dataclass(frozen=True)
class SplitMetricsBundle:
    train: RollingMetrics
    val: RollingMetrics
    test: RollingMetrics


@dataclass(frozen=True)
class PerHorizonTuneBatchResult:
    candidate_index: int
    parent_index: int
    horizon: int
    tuning_size_index: int
    text_attn_layers: list[int]
    epochs: int
    gate_lr: float
    text_attn_lr: float
    gate_logit_clamp: float | None
    tune_batch_size: int
    max_context: int | None
    target_lag_count: int | None
    covariate_lag_count: int | None
    embedding_lag_count: int | None
    best_epoch: int | None
    before: SplitMetricsBundle
    after: SplitMetricsBundle
    ranking_score: float
    is_best_for_horizon: bool = False


@dataclass(frozen=True)
class PerHorizonTuneBatchExecutionArtifacts:
    result: PerHorizonTuneBatchResult
    assigned_device: str | None = None


@dataclass(frozen=True)
class HorizonTrialMetrics:
    horizon: int
    best_epoch: int | None
    val_loss: float
    val_mae: float
    val_rmse: float
    val_mape: float
    val_real_mae: float
    val_real_rmse: float
    val_real_mape: float
    test_loss: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    test_mape: float | None = None
    test_real_mae: float | None = None
    test_real_rmse: float | None = None
    test_real_mape: float | None = None


@dataclass(frozen=True)
class SummaryMetrics:
    mae: float
    rmse: float
    mape: float
    real_mae: float
    real_rmse: float
    real_mape: float


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


@dataclass(frozen=True)
class CovariateLagFamily:
    prefix: str
    base_lag_count: int
    available_lag_count: int


@dataclass(frozen=True)
class LagSearchMetadata:
    target_lag_prefix: str
    base_target_lag_count: int | None
    available_target_lag_count: int
    covariate_families: list[CovariateLagFamily]
    shared_covariate_lag_count: int
    embedding_lag_prefix: str | None
    base_embedding_lag_count: int | None
    available_embedding_lag_count: int


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


def _delete_checkpoint_file(checkpoint_path: str | Path) -> None:
    Path(checkpoint_path).unlink(missing_ok=True)


def _delete_checkpoint_files(
    checkpoint_paths: dict[int, str],
    *,
    keep_trial_indices: set[int],
) -> int:
    deleted = 0
    for trial_index, checkpoint_path in list(checkpoint_paths.items()):
        if trial_index in keep_trial_indices:
            continue
        _delete_checkpoint_file(checkpoint_path)
        del checkpoint_paths[trial_index]
        deleted += 1
    return deleted


def _clone_state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[name] = value.detach().cpu().clone()
        else:
            cloned[name] = copy.deepcopy(value)
    return cloned


def _is_text_mixing_state_key(state_key: str) -> bool:
    return (
        state_key.endswith(".alpha")
        or ".text_head_projs." in state_key
        or ".text_head_q_norms." in state_key
        or ".text_head_k_norms." in state_key
        or ".text_shared_proj." in state_key
        or ".text_q_norm." in state_key
        or ".text_k_norm." in state_key
        or ".text_attn_linears." in state_key
    )


def _clone_text_mixing_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, Any]:
    text_mixing_state_dict = {
        name: value
        for name, value in model.state_dict().items()
        if _is_text_mixing_state_key(name)
    }
    if not text_mixing_state_dict:
        raise RuntimeError("No text-mixing parameters were found in the model state_dict.")
    return _clone_state_dict_to_cpu(text_mixing_state_dict)


def _load_text_mixing_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    if incompatible_keys.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys while loading text-mixing checkpoint bundle: "
            + ", ".join(sorted(incompatible_keys.unexpected_keys))
        )


def _top_trial_indices(results: list[RandomSearchTrialResult], *, top_k: int) -> set[int]:
    top_limit = min(top_k, len(results))
    ranked_results = sorted(results, key=trial_sort_key)
    return {result.trial_index for result in ranked_results[:top_limit]}


def _metrics_from_triplet(metrics: tuple[tuple[float, float, float], tuple[float, float, float]]) -> SummaryMetrics:
    return SummaryMetrics(
        mae=metrics[0][0],
        rmse=metrics[0][1],
        mape=metrics[0][2],
        real_mae=metrics[1][0],
        real_rmse=metrics[1][1],
        real_mape=metrics[1][2],
    )


def _mean_summary_metrics(metrics_list: list[SummaryMetrics]) -> SummaryMetrics:
    return SummaryMetrics(
        mae=float(np.mean([metrics.mae for metrics in metrics_list])),
        rmse=float(np.mean([metrics.rmse for metrics in metrics_list])),
        mape=float(np.mean([metrics.mape for metrics in metrics_list])),
        real_mae=float(np.mean([metrics.real_mae for metrics in metrics_list])),
        real_rmse=float(np.mean([metrics.real_rmse for metrics in metrics_list])),
        real_mape=float(np.mean([metrics.real_mape for metrics in metrics_list])),
    )


def _mean_rolling_metrics(metrics_list: list[RollingMetrics]) -> RollingMetrics:
    return RollingMetrics(
        loss=float(np.mean([metrics.loss for metrics in metrics_list])),
        mae=float(np.mean([metrics.mae for metrics in metrics_list])),
        rmse=float(np.mean([metrics.rmse for metrics in metrics_list])),
        mape=float(np.mean([metrics.mape for metrics in metrics_list])),
        real_mae=float(np.mean([metrics.real_mae for metrics in metrics_list])),
        real_rmse=float(np.mean([metrics.real_rmse for metrics in metrics_list])),
        real_mape=float(np.mean([metrics.real_mape for metrics in metrics_list])),
    )


def _read_csv_header(path: str | Path) -> list[str]:
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"Dataset file is empty: {resolved_path}") from exc
    return header


def _contiguous_lag_count(lag_values: set[int]) -> int:
    max_count = 0
    while (max_count + 1) in lag_values:
        max_count += 1
    return max_count


def _feature_family_name(feature_name: str) -> str:
    match = re.fullmatch(r"(.+)_lag(\d+)", feature_name)
    if match is None:
        return feature_name
    return match.group(1)


def _infer_lag_prefix_from_columns(
    columns: list[str],
    *,
    label: str,
) -> tuple[str, list[int]]:
    if not columns:
        raise ValueError(f"{label} must contain at least one lagged column.")

    prefixes: set[str] = set()
    lag_values: list[int] = []
    for column in columns:
        match = re.fullmatch(r"(.+)_lag(\d+)", column)
        if match is None:
            raise ValueError(
                f"{label} must contain only lagged columns named like '<prefix>_lagN'. "
                f"Got: {column!r}"
            )
        prefixes.add(match.group(1))
        lag_values.append(int(match.group(2)))

    if len(prefixes) != 1:
        raise ValueError(
            f"{label} must map to exactly one lagged column family. Got prefixes: {sorted(prefixes)}"
        )
    return next(iter(prefixes)), lag_values


def _infer_lag_prefix_from_template(template: str, *, label: str) -> str:
    if "{lag}" not in template:
        raise ValueError(f"{label} must include the '{{lag}}' placeholder.")
    example_column = template.format(lag=1)
    match = re.fullmatch(r"(.+)_lag(\d+)", example_column)
    if match is None:
        raise ValueError(
            f"{label} must format to '<prefix>_lagN'. Got: {example_column!r}"
        )
    return match.group(1)


def discover_lag_search_metadata(base_run_cfg: DataConfig) -> LagSearchMetadata:
    header = _read_csv_header(base_run_cfg.data_path)

    target_lag_prefix_candidates = [base_run_cfg.target_column]
    if base_run_cfg.target_mode == "target_differencing":
        target_lag_prefix_candidates.insert(0, "target_diff")

    resolved_target_lag_prefix: str | None = None
    available_target_lags: set[int] = set()
    base_target_lag_count: int | None = None
    for target_lag_prefix in target_lag_prefix_candidates:
        target_lag_pattern = re.compile(rf"^{re.escape(target_lag_prefix)}_lag(\d+)$")
        base_target_lag_columns = [
            column
            for column in base_run_cfg.numeric_features
            if target_lag_pattern.fullmatch(column) is not None
        ]
        if not base_target_lag_columns:
            continue
        resolved_target_lag_prefix = target_lag_prefix
        base_target_lag_count = len(base_target_lag_columns)
        available_target_lags = {
            int(match.group(1))
            for column in header
            if (match := target_lag_pattern.fullmatch(column)) is not None
        }
        break
    if resolved_target_lag_prefix is None:
        resolved_target_lag_prefix = (
            "target_diff" if base_run_cfg.target_mode == "target_differencing" else base_run_cfg.target_column
        )

    configured_covariate_lags: dict[str, set[int]] = {}
    covariate_order: list[str] = []
    raw_covariate_families: set[str] = set()
    for feature_name in base_run_cfg.numeric_features:
        family_name = _feature_family_name(feature_name)
        if family_name == resolved_target_lag_prefix:
            continue
        if family_name not in configured_covariate_lags:
            configured_covariate_lags[family_name] = set()
            covariate_order.append(family_name)
        match = re.fullmatch(r"(.+)_lag(\d+)", feature_name)
        if match is None:
            raw_covariate_families.add(family_name)
            continue
        configured_covariate_lags[family_name].add(int(match.group(2)))

    covariate_families: list[CovariateLagFamily] = []
    for prefix in covariate_order:
        covariate_lag_pattern = re.compile(rf"^{re.escape(prefix)}_lag(\d+)$")
        available_lag_count = _contiguous_lag_count(
            {
                int(match.group(1))
                for column in header
                if (match := covariate_lag_pattern.fullmatch(column)) is not None
            }
        )
        configured_lags = configured_covariate_lags[prefix]
        base_lag_count = (
            available_lag_count
            if prefix in raw_covariate_families
            else _contiguous_lag_count(configured_lags)
        )
        covariate_families.append(
            CovariateLagFamily(
                prefix=prefix,
                base_lag_count=base_lag_count,
                available_lag_count=available_lag_count,
            )
        )

    shared_covariate_lag_count = min(
        (family.available_lag_count for family in covariate_families),
        default=0,
    )

    embedding_lag_prefix: str | None
    base_embedding_lag_count: int | None
    available_embedding_lag_count: int
    if base_run_cfg.embedding_columns is not None:
        embedding_lag_prefix, base_embedding_lags = _infer_lag_prefix_from_columns(
            list(base_run_cfg.embedding_columns),
            label="embedding_columns",
        )
        base_embedding_lag_count = len(base_embedding_lags)
    elif base_run_cfg.embedding_column_template is not None:
        embedding_lag_prefix = _infer_lag_prefix_from_template(
            base_run_cfg.embedding_column_template,
            label="embedding_column_template",
        )
        base_embedding_lag_count = (
            len(base_run_cfg.embedding_lags) if base_run_cfg.embedding_lags else None
        )
    else:
        embedding_lag_prefix = None
        base_embedding_lag_count = None

    if embedding_lag_prefix is None:
        available_embedding_lag_count = 0
    else:
        embedding_lag_pattern = re.compile(rf"^{re.escape(embedding_lag_prefix)}_lag(\d+)$")
        available_embedding_lag_count = _contiguous_lag_count(
            {
                int(match.group(1))
                for column in header
                if (match := embedding_lag_pattern.fullmatch(column)) is not None
            }
        )

    return LagSearchMetadata(
        target_lag_prefix=resolved_target_lag_prefix,
        base_target_lag_count=base_target_lag_count,
        available_target_lag_count=_contiguous_lag_count(available_target_lags),
        covariate_families=covariate_families,
        shared_covariate_lag_count=shared_covariate_lag_count,
        embedding_lag_prefix=embedding_lag_prefix,
        base_embedding_lag_count=base_embedding_lag_count,
        available_embedding_lag_count=available_embedding_lag_count,
    )


def _validate_requested_lag_counts(
    *,
    requested_counts: list[int] | None,
    available_count: int,
    search_key: str,
    family_label: str,
) -> None:
    if requested_counts is None:
        return
    invalid_counts = sorted({count for count in requested_counts if count > available_count})
    if invalid_counts:
        raise ValueError(
            f"{search_key} requested {family_label} lag counts {invalid_counts}, but only "
            f"contiguous lag1..lag{available_count} are available."
        )


def validate_lag_search_space(
    base_run_cfg: DataConfig,
    search_space_cfg: SearchSpaceConfig,
    lag_search_metadata: LagSearchMetadata,
) -> None:
    if search_space_cfg.target_lag_count is not None:
        if lag_search_metadata.base_target_lag_count is None:
            raise ValueError(
                "search_space.target_lag_count requires base config numeric_features to include "
                f"lagged target columns matching '{base_run_cfg.target_column}_lagN'."
            )
        _validate_requested_lag_counts(
            requested_counts=search_space_cfg.target_lag_count,
            available_count=lag_search_metadata.available_target_lag_count,
            search_key="search_space.target_lag_count",
            family_label=f"target columns for '{base_run_cfg.target_column}'",
        )

    if search_space_cfg.covariate_lag_count is not None:
        if not lag_search_metadata.covariate_families:
            if any(count != 0 for count in search_space_cfg.covariate_lag_count):
                raise ValueError(
                    "search_space.covariate_lag_count requires base config numeric_features to include "
                    "at least one non-target lagged numeric feature family."
                )
        else:
            invalid_counts = sorted(
                {
                    count
                    for count in search_space_cfg.covariate_lag_count
                    if count > lag_search_metadata.shared_covariate_lag_count
                }
            )
            if invalid_counts:
                raise ValueError(
                    "search_space.covariate_lag_count requested values "
                    f"{invalid_counts}, but the shortest non-target lag family only supports "
                    f"contiguous lags up to lag{lag_search_metadata.shared_covariate_lag_count}."
                )

    if search_space_cfg.embedding_lag_count is not None:
        if lag_search_metadata.embedding_lag_prefix is None:
            raise ValueError(
                "search_space.embedding_lag_count requires the base config to define one lagged "
                "embedding column family."
            )
        _validate_requested_lag_counts(
            requested_counts=search_space_cfg.embedding_lag_count,
            available_count=lag_search_metadata.available_embedding_lag_count,
            search_key="search_space.embedding_lag_count",
            family_label=f"embedding columns for '{lag_search_metadata.embedding_lag_prefix}'",
        )


def _resolve_target_lag_numeric_features(
    numeric_features: list[str],
    *,
    target_lag_count: int,
    lag_search_metadata: LagSearchMetadata,
) -> list[str]:
    existing_target_lag_columns = {
        f"{lag_search_metadata.target_lag_prefix}_lag{lag}"
        for lag in range(1, lag_search_metadata.available_target_lag_count + 1)
    }
    resolved_target_columns = [
        f"{lag_search_metadata.target_lag_prefix}_lag{lag}"
        for lag in range(1, target_lag_count + 1)
    ]

    resolved_numeric_features: list[str] = []
    inserted_target_block = False
    for feature in numeric_features:
        if feature in existing_target_lag_columns:
            if not inserted_target_block:
                resolved_numeric_features.extend(resolved_target_columns)
                inserted_target_block = True
            continue
        resolved_numeric_features.append(feature)

    if not inserted_target_block:
        raise ValueError(
            "Unable to resolve target lag search because the base config does not contain "
            f"any '{lag_search_metadata.target_lag_prefix}_lagN' numeric feature."
        )
    return resolved_numeric_features


def _resolve_covariate_lag_numeric_features(
    numeric_features: list[str],
    *,
    covariate_lag_count: int,
    lag_search_metadata: LagSearchMetadata,
) -> list[str]:
    covariate_prefixes = {family.prefix for family in lag_search_metadata.covariate_families}
    resolved_numeric_features: list[str] = []
    inserted_covariates: set[str] = set()

    for feature_name in numeric_features:
        family_name = _feature_family_name(feature_name)
        if family_name in covariate_prefixes:
            if family_name not in inserted_covariates:
                resolved_numeric_features.extend(
                    f"{family_name}_lag{lag}" for lag in range(1, covariate_lag_count + 1)
                )
                inserted_covariates.add(family_name)
            continue
        resolved_numeric_features.append(feature_name)

    return resolved_numeric_features


def _resolve_embedding_columns(
    *,
    embedding_lag_count: int,
    lag_search_metadata: LagSearchMetadata,
) -> list[str]:
    if lag_search_metadata.embedding_lag_prefix is None:
        raise ValueError("Embedding lag resolution requires a discovered embedding lag prefix.")
    return [
        f"{lag_search_metadata.embedding_lag_prefix}_lag{lag}"
        for lag in range(1, embedding_lag_count + 1)
    ]


def _trial_data_cache_key(
    *,
    target_lag_count: int | None,
    covariate_lag_count: int | None,
    embedding_lag_count: int | None,
) -> tuple[int | None, int | None, int | None]:
    return (target_lag_count, covariate_lag_count, embedding_lag_count)


def trial_spec_target_lag_count(run_cfg: DataConfig) -> int | None:
    target_lag_prefix_candidates = [run_cfg.target_column]
    if run_cfg.target_mode == "target_differencing":
        target_lag_prefix_candidates.insert(0, "target_diff")

    for target_lag_prefix in target_lag_prefix_candidates:
        target_lag_pattern = re.compile(rf"^{re.escape(target_lag_prefix)}_lag(\d+)$")
        count = _contiguous_lag_count(
            {
                int(match.group(1))
                for feature in run_cfg.numeric_features
                if (match := target_lag_pattern.fullmatch(feature)) is not None
            }
        )
        if count > 0:
            return count
    return None


def trial_spec_embedding_lag_count(run_cfg: DataConfig) -> int | None:
    if run_cfg.embedding_columns is not None:
        return _contiguous_lag_count(
            {
                int(match.group(2))
                for column in run_cfg.embedding_columns
                if (match := re.fullmatch(r"(.+)_lag(\d+)", column)) is not None
                and int(match.group(2)) > 0
            }
        ) or None
    return _contiguous_lag_count({lag for lag in run_cfg.embedding_lags if lag > 0}) or None


def trial_spec_covariate_lag_count(run_cfg: DataConfig) -> int | None:
    target_lag_prefix_candidates = [run_cfg.target_column]
    if run_cfg.target_mode == "target_differencing":
        target_lag_prefix_candidates.insert(0, "target_diff")
    target_prefixes = set(target_lag_prefix_candidates)

    family_lag_counts: list[int] = []
    family_seen: set[str] = set()
    for feature in run_cfg.numeric_features:
        family_name = _feature_family_name(feature)
        if family_name in target_prefixes or family_name in family_seen:
            continue
        family_seen.add(family_name)
        lag_count = _contiguous_lag_count(
            {
                int(match.group(2))
                for candidate in run_cfg.numeric_features
                if (match := re.fullmatch(r"(.+)_lag(\d+)", candidate)) is not None
                and match.group(1) == family_name
            }
        )
        family_lag_counts.append(lag_count)

    if not family_lag_counts:
        return 0
    return min(family_lag_counts)


def _serialize_summary_metrics(metrics: SummaryMetrics | None, *, label: str | None = None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    payload = {
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "mape": metrics.mape,
        "real_mae": metrics.real_mae,
        "real_rmse": metrics.real_rmse,
        "real_mape": metrics.real_mape,
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
        "val_real_mae": metrics.val_real_mae,
        "val_real_rmse": metrics.val_real_rmse,
        "val_real_mape": metrics.val_real_mape,
        "test_loss": metrics.test_loss,
        "test_mae": metrics.test_mae,
        "test_rmse": metrics.test_rmse,
        "test_mape": metrics.test_mape,
        "test_real_mae": metrics.test_real_mae,
        "test_real_rmse": metrics.test_real_rmse,
        "test_real_mape": metrics.test_real_mape,
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


def _format_summary_metrics_line(prefix: str, metrics: SummaryMetrics) -> str:
    return (
        f"{prefix} | mae={metrics.mae:.6f} | "
        f"rmse={metrics.rmse:.6f} | "
        f"mape={metrics.mape:.6f}% | "
        f"real_mae={metrics.real_mae:.6f} | "
        f"real_rmse={metrics.real_rmse:.6f} | "
        f"real_mape={metrics.real_mape:.6f}%"
    )


def _format_per_horizon_baseline_lines(horizon_result: HorizonBaselineResult) -> list[str]:
    prefix = f"baseline_horizon={horizon_result.horizon}"
    lines = [
        _format_summary_metrics_line(f"{prefix} | val_no_text", horizon_result.val_no_text),
        _format_summary_metrics_line(f"{prefix} | val_with_text", horizon_result.val_with_text),
        _format_summary_metrics_line(f"{prefix} | val_pca", horizon_result.val_pca),
        _format_summary_metrics_line(f"{prefix} | test_no_text", horizon_result.test_no_text),
        _format_summary_metrics_line(f"{prefix} | test_with_text", horizon_result.test_with_text),
        _format_summary_metrics_line(f"{prefix} | test_pca", horizon_result.test_pca),
    ]
    lines.append(
        (
            f"{prefix} | val_truncate | unavailable"
            if horizon_result.val_truncate is None
            else (
                f"{prefix} | val_truncate | label={horizon_result.val_truncate_label} | "
                f"mae={horizon_result.val_truncate.mae:.6f} | "
                f"rmse={horizon_result.val_truncate.rmse:.6f} | "
                f"mape={horizon_result.val_truncate.mape:.6f}% | "
                f"real_mae={horizon_result.val_truncate.real_mae:.6f} | "
                f"real_rmse={horizon_result.val_truncate.real_rmse:.6f} | "
                f"real_mape={horizon_result.val_truncate.real_mape:.6f}%"
            )
        )
    )
    lines.append(
        (
            f"{prefix} | test_truncate | unavailable"
            if horizon_result.test_truncate is None
            else (
                f"{prefix} | test_truncate | label={horizon_result.test_truncate_label} | "
                f"mae={horizon_result.test_truncate.mae:.6f} | "
                f"rmse={horizon_result.test_truncate.rmse:.6f} | "
                f"mape={horizon_result.test_truncate.mape:.6f}% | "
                f"real_mae={horizon_result.test_truncate.real_mae:.6f} | "
                f"real_rmse={horizon_result.test_truncate.real_rmse:.6f} | "
                f"real_mape={horizon_result.test_truncate.real_mape:.6f}%"
            )
        )
    )
    return lines


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
                val_no_text = _metrics_from_triplet(
                    evaluate_fresh_baseline_metric(
                        base_run_cfg,
                        data_splits=data_splits,
                        horizon=horizon,
                        split_name="val",
                        baseline_kind="no_text",
                        label="Baseline val (no text attn)",
                    )
                )
                val_with_text = _metrics_from_triplet(
                    evaluate_fresh_baseline_metric(
                        base_run_cfg,
                        data_splits=data_splits,
                        horizon=horizon,
                        split_name="val",
                        baseline_kind="with_text",
                        label="Baseline val (with text attn)",
                    )
                )
                val_pca = _metrics_from_triplet(
                    evaluate_fresh_baseline_metric(
                        base_run_cfg,
                        data_splits=data_splits,
                        horizon=horizon,
                        split_name="val",
                        baseline_kind="pca",
                        label="Baseline val (PCA)",
                    )
                )
                val_truncate_result = evaluate_fresh_baseline_metric(
                    base_run_cfg,
                    data_splits=data_splits,
                    horizon=horizon,
                    split_name="val",
                    baseline_kind="truncate_text",
                    label="Baseline val (text truncate)",
                )

                test_no_text = _metrics_from_triplet(
                    evaluate_fresh_baseline_metric(
                        base_run_cfg,
                        data_splits=data_splits,
                        horizon=horizon,
                        split_name="test",
                        baseline_kind="no_text",
                        label="Baseline test (no text attn)",
                    )
                )
                test_with_text = _metrics_from_triplet(
                    evaluate_fresh_baseline_metric(
                        base_run_cfg,
                        data_splits=data_splits,
                        horizon=horizon,
                        split_name="test",
                        baseline_kind="with_text",
                        label="Baseline test (with text attn)",
                    )
                )
                test_pca = _metrics_from_triplet(
                    evaluate_fresh_baseline_metric(
                        base_run_cfg,
                        data_splits=data_splits,
                        horizon=horizon,
                        split_name="test",
                        baseline_kind="pca",
                        label="Baseline test (PCA)",
                    )
                )
                test_truncate_result = evaluate_fresh_baseline_metric(
                    base_run_cfg,
                    data_splits=data_splits,
                    horizon=horizon,
                    split_name="test",
                    baseline_kind="truncate_text",
                    label="Baseline test (text truncate)",
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
    *,
    lag_search_metadata: LagSearchMetadata | None = None,
) -> dict[str, list[object]]:
    if lag_search_metadata is not None:
        validate_lag_search_space(base_run_cfg, search_space_cfg, lag_search_metadata)
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
        "max_context": search_space_cfg.max_context if search_space_cfg.max_context is not None else [base_run_cfg.tuning.max_context],
        "target_lag_count": search_space_cfg.target_lag_count if search_space_cfg.target_lag_count is not None else [None],
        "covariate_lag_count": (
            search_space_cfg.covariate_lag_count
            if search_space_cfg.covariate_lag_count is not None
            else [None]
        ),
        "embedding_lag_count": (
            search_space_cfg.embedding_lag_count
            if search_space_cfg.embedding_lag_count is not None
            else [None]
        ),
    }


def build_tune_batch_size_choices(
    base_run_cfg: DataConfig,
    search_space_cfg: SearchSpaceConfig,
) -> list[int]:
    if search_space_cfg.tune_batch_size is None:
        return [base_run_cfg.tuning.tune_batch_size]
    return list(search_space_cfg.tune_batch_size)


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
        **{
            **{field_name: resolved_values[field_name] for field_name in SEARCH_FIELD_NAMES},
            "tune_batch_size": int(search_choices["_base_tune_batch_size"][0]),
        }
    )


def sample_trial_specs(
    base_run_cfg: DataConfig,
    random_search_cfg: RandomSearchConfig,
    *,
    lag_search_metadata: LagSearchMetadata | None = None,
) -> tuple[list[RandomSearchTrialSpec], int]:
    search_choices = build_search_choices(
        base_run_cfg,
        random_search_cfg.search_space,
        lag_search_metadata=lag_search_metadata,
    )
    search_choices["_base_tune_batch_size"] = [base_run_cfg.tuning.tune_batch_size]
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
    *,
    lag_search_metadata: LagSearchMetadata | None = None,
) -> DataConfig:
    resolved_numeric_features = list(base_run_cfg.numeric_features)
    resolved_embedding_columns = (
        None if base_run_cfg.embedding_columns is None else list(base_run_cfg.embedding_columns)
    )
    resolved_embedding_lags = list(base_run_cfg.embedding_lags)

    if trial_spec.target_lag_count is not None:
        if lag_search_metadata is None:
            raise ValueError("target_lag_count resolution requires lag_search_metadata.")
        resolved_numeric_features = _resolve_target_lag_numeric_features(
            resolved_numeric_features,
            target_lag_count=trial_spec.target_lag_count,
            lag_search_metadata=lag_search_metadata,
        )

    if trial_spec.covariate_lag_count is not None:
        if lag_search_metadata is None:
            raise ValueError("covariate_lag_count resolution requires lag_search_metadata.")
        resolved_numeric_features = _resolve_covariate_lag_numeric_features(
            resolved_numeric_features,
            covariate_lag_count=trial_spec.covariate_lag_count,
            lag_search_metadata=lag_search_metadata,
        )

    if trial_spec.embedding_lag_count is not None:
        if lag_search_metadata is None:
            raise ValueError("embedding_lag_count resolution requires lag_search_metadata.")
        resolved_embedding_columns = _resolve_embedding_columns(
            embedding_lag_count=trial_spec.embedding_lag_count,
            lag_search_metadata=lag_search_metadata,
        )
        resolved_embedding_lags = []

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
    return replace(
        base_run_cfg,
        numeric_features=resolved_numeric_features,
        embedding_columns=resolved_embedding_columns,
        embedding_lags=resolved_embedding_lags,
        model=updated_model_cfg,
        tuning=updated_tuning_cfg,
    )


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
        f"target_lag_count={trial_spec_target_lag_count(trial_cfg)} | "
        f"covariate_lag_count={trial_spec_covariate_lag_count(trial_cfg)} | "
        f"embedding_lag_count={trial_spec_embedding_lag_count(trial_cfg)} | "
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
                val_real_mae=val_metrics.real_mae,
                val_real_rmse=val_metrics.real_rmse,
                val_real_mape=val_metrics.real_mape,
            )
        )
        checkpoint_bundle[horizon] = _clone_text_mixing_state_dict_to_cpu(prepared_trial.reg.model)
        print(
            f"Horizon {horizon:02d} complete | best_epoch={fine_tune_outcome.best_epoch} | "
            f"val_mae={val_metrics.mae:.6f} | val_rmse={val_metrics.rmse:.6f} | "
            f"val_mape={val_metrics.mape:.6f}% | "
            f"val_real_mae={val_metrics.real_mae:.6f} | "
            f"val_real_rmse={val_metrics.real_rmse:.6f} | "
            f"val_real_mape={val_metrics.real_mape:.6f}%"
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
        target_lag_count=trial_spec_target_lag_count(trial_cfg),
        covariate_lag_count=trial_spec_covariate_lag_count(trial_cfg),
        embedding_lag_count=trial_spec_embedding_lag_count(trial_cfg),
        val_loss=aggregate_val_metrics.loss,
        val_mae=aggregate_val_metrics.mae,
        val_rmse=aggregate_val_metrics.rmse,
        val_mape=aggregate_val_metrics.mape,
        val_real_mae=aggregate_val_metrics.real_mae,
        val_real_rmse=aggregate_val_metrics.real_rmse,
        val_real_mape=aggregate_val_metrics.real_mape,
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
        f"val_mape={aggregate_val_metrics.mape:.6f}% | "
        f"val_real_mae={aggregate_val_metrics.real_mae:.6f} | "
        f"val_real_rmse={aggregate_val_metrics.real_rmse:.6f} | "
        f"val_real_mape={aggregate_val_metrics.real_mape:.6f}%"
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
    trial_data_splits: dict[int, FineTuneDataSplits],
    logs_dir: Path,
    checkpoints_dir: Path,
    top_k: int,
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
                data_splits=trial_data_splits[trial_index],
                trial_log_path=trial_log_path,
                checkpoint_path=checkpoint_path,
                assigned_device=None,
                tee_stdout=True,
            )
            results.append(artifacts.result)
            checkpoint_paths[trial_index] = artifacts.checkpoint_path
            _delete_checkpoint_files(
                checkpoint_paths,
                keep_trial_indices=_top_trial_indices(results, top_k=top_k),
            )
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
                data_splits=trial_data_splits[next_trial_index],
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
                _delete_checkpoint_files(
                    checkpoint_paths,
                    keep_trial_indices=_top_trial_indices(results, top_k=top_k),
                )
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
        "target_lag_count": result.target_lag_count,
        "covariate_lag_count": result.covariate_lag_count,
        "embedding_lag_count": result.embedding_lag_count,
        "val_loss": result.val_loss,
        "val_mae": result.val_mae,
        "val_rmse": result.val_rmse,
        "val_mape": result.val_mape,
        "val_real_mae": result.val_real_mae,
        "val_real_rmse": result.val_real_rmse,
        "val_real_mape": result.val_real_mape,
        "ranking_score": result.ranking_score,
        "test_loss": result.test_loss,
        "test_mae": result.test_mae,
        "test_rmse": result.test_rmse,
        "test_mape": result.test_mape,
        "test_real_mae": result.test_real_mae,
        "test_real_rmse": result.test_real_rmse,
        "test_real_mape": result.test_real_mape,
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
        "target_lag_count",
        "covariate_lag_count",
        "embedding_lag_count",
        "val_loss",
        "val_mae",
        "val_rmse",
        "val_mape",
        "val_real_mae",
        "val_real_rmse",
        "val_real_mape",
        "ranking_score",
        "test_loss",
        "test_mae",
        "test_rmse",
        "test_mape",
        "test_real_mae",
        "test_real_rmse",
        "test_real_mape",
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
            f"test_rmse={metrics.rmse:.6f} | test_mape={metrics.mape:.6f}% | "
            f"test_real_mae={metrics.real_mae:.6f} | "
            f"test_real_rmse={metrics.real_rmse:.6f} | "
            f"test_real_mape={metrics.real_mape:.6f}%\n"
        )
        if horizon_metrics is not None:
            for horizon_metric in horizon_metrics:
                f.write(
                    f"horizon={horizon_metric.horizon} | best_epoch={horizon_metric.best_epoch} | "
                    f"test_loss={horizon_metric.test_loss:.6f} | "
                    f"test_mae={horizon_metric.test_mae:.6f} | "
                    f"test_rmse={horizon_metric.test_rmse:.6f} | "
                    f"test_mape={horizon_metric.test_mape:.6f}% | "
                    f"test_real_mae={horizon_metric.test_real_mae:.6f} | "
                    f"test_real_rmse={horizon_metric.test_real_rmse:.6f} | "
                    f"test_real_mape={horizon_metric.test_real_mape:.6f}%\n"
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
            f"mape={shared_baseline.val_no_text.mape:.6f}% | "
            f"real_mae={shared_baseline.val_no_text.real_mae:.6f} | "
            f"real_rmse={shared_baseline.val_no_text.real_rmse:.6f} | "
            f"real_mape={shared_baseline.val_no_text.real_mape:.6f}%"
        ),
        (
            f"val_with_text | mae={shared_baseline.val_with_text.mae:.6f} | "
            f"rmse={shared_baseline.val_with_text.rmse:.6f} | "
            f"mape={shared_baseline.val_with_text.mape:.6f}% | "
            f"real_mae={shared_baseline.val_with_text.real_mae:.6f} | "
            f"real_rmse={shared_baseline.val_with_text.real_rmse:.6f} | "
            f"real_mape={shared_baseline.val_with_text.real_mape:.6f}%"
        ),
        (
            f"val_pca | mae={shared_baseline.val_pca.mae:.6f} | "
            f"rmse={shared_baseline.val_pca.rmse:.6f} | "
            f"mape={shared_baseline.val_pca.mape:.6f}% | "
            f"real_mae={shared_baseline.val_pca.real_mae:.6f} | "
            f"real_rmse={shared_baseline.val_pca.real_rmse:.6f} | "
            f"real_mape={shared_baseline.val_pca.real_mape:.6f}%"
        ),
        (
            "val_truncate | unavailable"
            if shared_baseline.val_truncate is None
            else (
                f"val_truncate | label={shared_baseline.val_truncate_label} | "
                f"mae={shared_baseline.val_truncate.mae:.6f} | "
                f"rmse={shared_baseline.val_truncate.rmse:.6f} | "
                f"mape={shared_baseline.val_truncate.mape:.6f}% | "
                f"real_mae={shared_baseline.val_truncate.real_mae:.6f} | "
                f"real_rmse={shared_baseline.val_truncate.real_rmse:.6f} | "
                f"real_mape={shared_baseline.val_truncate.real_mape:.6f}%"
            )
        ),
        (
            f"test_no_text | mae={shared_baseline.test_no_text.mae:.6f} | "
            f"rmse={shared_baseline.test_no_text.rmse:.6f} | "
            f"mape={shared_baseline.test_no_text.mape:.6f}% | "
            f"real_mae={shared_baseline.test_no_text.real_mae:.6f} | "
            f"real_rmse={shared_baseline.test_no_text.real_rmse:.6f} | "
            f"real_mape={shared_baseline.test_no_text.real_mape:.6f}%"
        ),
        (
            f"test_with_text | mae={shared_baseline.test_with_text.mae:.6f} | "
            f"rmse={shared_baseline.test_with_text.rmse:.6f} | "
            f"mape={shared_baseline.test_with_text.mape:.6f}% | "
            f"real_mae={shared_baseline.test_with_text.real_mae:.6f} | "
            f"real_rmse={shared_baseline.test_with_text.real_rmse:.6f} | "
            f"real_mape={shared_baseline.test_with_text.real_mape:.6f}%"
        ),
        (
            f"test_pca | mae={shared_baseline.test_pca.mae:.6f} | "
            f"rmse={shared_baseline.test_pca.rmse:.6f} | "
            f"mape={shared_baseline.test_pca.mape:.6f}% | "
            f"real_mae={shared_baseline.test_pca.real_mae:.6f} | "
            f"real_rmse={shared_baseline.test_pca.real_rmse:.6f} | "
            f"real_mape={shared_baseline.test_pca.real_mape:.6f}%"
        ),
        (
            "test_truncate | unavailable"
            if shared_baseline.test_truncate is None
            else (
                f"test_truncate | label={shared_baseline.test_truncate_label} | "
                f"mae={shared_baseline.test_truncate.mae:.6f} | "
                f"rmse={shared_baseline.test_truncate.rmse:.6f} | "
                f"mape={shared_baseline.test_truncate.mape:.6f}% | "
                f"real_mae={shared_baseline.test_truncate.real_mae:.6f} | "
                f"real_rmse={shared_baseline.test_truncate.real_rmse:.6f} | "
                f"real_mape={shared_baseline.test_truncate.real_mape:.6f}%"
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
            f"val_real_mae={best_trial.val_real_mae:.6f} | "
            f"val_real_rmse={best_trial.val_real_rmse:.6f} | "
            f"val_real_mape={best_trial.val_real_mape:.6f}% | "
            f"test_mae={best_trial.test_mae:.6f} | "
            f"test_rmse={best_trial.test_rmse:.6f} | "
            f"test_mape={best_trial.test_mape:.6f}% | "
            f"test_real_mae={best_trial.test_real_mae:.6f} | "
            f"test_real_rmse={best_trial.test_real_rmse:.6f} | "
            f"test_real_mape={best_trial.test_real_mape:.6f}%"
        ),
        (
            f"params | text_attn_layers={best_trial.text_attn_layers} | "
            f"epochs={best_trial.epochs} | gate_lr={best_trial.gate_lr} | "
            f"text_attn_lr={best_trial.text_attn_lr} | "
            f"gate_logit_clamp={best_trial.gate_logit_clamp} | "
            f"tune_batch_size={best_trial.tune_batch_size} | "
            f"max_context={best_trial.max_context} | "
            f"target_lag_count={best_trial.target_lag_count} | "
            f"covariate_lag_count={best_trial.covariate_lag_count} | "
            f"embedding_lag_count={best_trial.embedding_lag_count}"
        ),
        "",
    ]
    if shared_baseline.per_horizon is not None:
        lines.append("Shared Baseline By Horizon")
        for horizon_result in shared_baseline.per_horizon:
            lines.extend(_format_per_horizon_baseline_lines(horizon_result))
        lines.append("")
    if best_trial.horizon_metrics is not None:
        for metrics in best_trial.horizon_metrics:
            lines.append(
                (
                    f"best_trial_horizon={metrics.horizon} | best_epoch={metrics.best_epoch} | "
                    f"val_mae={metrics.val_mae:.6f} | val_rmse={metrics.val_rmse:.6f} | "
                    f"val_real_mae={metrics.val_real_mae:.6f} | val_real_rmse={metrics.val_real_rmse:.6f} | "
                    f"test_mae={metrics.test_mae:.6f} | test_rmse={metrics.test_rmse:.6f} | "
                    f"test_real_mae={metrics.test_real_mae:.6f} | test_real_rmse={metrics.test_real_rmse:.6f}"
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
                f"val_real_mae={result.val_real_mae:.6f} | "
                f"val_real_rmse={result.val_real_rmse:.6f} | "
                f"val_real_mape={result.val_real_mape:.6f}% | "
                f"test_mae={result.test_mae:.6f} | test_rmse={result.test_rmse:.6f} | "
                f"test_mape={result.test_mape:.6f}% | "
                f"test_real_mae={result.test_real_mae:.6f} | "
                f"test_real_rmse={result.test_real_rmse:.6f} | "
                f"test_real_mape={result.test_real_mape:.6f}% | text_attn_layers={result.text_attn_layers} | "
                f"epochs={result.epochs} | gate_lr={result.gate_lr} | "
                f"text_attn_lr={result.text_attn_lr} | gate_logit_clamp={result.gate_logit_clamp} | "
                f"tune_batch_size={result.tune_batch_size} | max_context={result.max_context} | "
                f"target_lag_count={result.target_lag_count} | "
                f"covariate_lag_count={result.covariate_lag_count} | "
                f"embedding_lag_count={result.embedding_lag_count}"
            )
        )
        if result.horizon_metrics is not None:
            for metrics in result.horizon_metrics:
                lines.append(
                    (
                        f"trial_index={result.trial_index} horizon={metrics.horizon} | "
                        f"best_epoch={metrics.best_epoch} | val_mae={metrics.val_mae:.6f} | "
                        f"val_rmse={metrics.val_rmse:.6f} | "
                        f"val_real_mae={metrics.val_real_mae:.6f} | val_real_rmse={metrics.val_real_rmse:.6f} | "
                        f"test_mae={metrics.test_mae:.6f} | test_rmse={metrics.test_rmse:.6f} | "
                        f"test_real_mae={metrics.test_real_mae:.6f} | test_real_rmse={metrics.test_real_rmse:.6f}"
                    )
                )

    with (output_dir / "summary.log").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _evaluate_all_splits(
    prepared_trial: PreparedFineTuneTrial,
    *,
    tuning_cfg,
) -> SplitMetricsBundle:
    return SplitMetricsBundle(
        train=evaluate_prepared_split(
            prepared_trial,
            split_name="train",
            use_text=True,
            tuning_cfg=tuning_cfg,
        ),
        val=evaluate_prepared_split(
            prepared_trial,
            split_name="val",
            use_text=True,
            tuning_cfg=tuning_cfg,
        ),
        test=evaluate_prepared_split(
            prepared_trial,
            split_name="test",
            use_text=True,
            tuning_cfg=tuning_cfg,
        ),
    )


def _execute_per_horizon_tune_batch_candidate(
    trial_cfg: DataConfig,
    *,
    candidate_index: int,
    parent_index: int,
    horizon: int,
    tuning_size_index: int,
    data_splits: FineTuneDataSplits,
) -> PerHorizonTuneBatchResult:
    print(
        f"\n== Shared parameter set {parent_index} | horizon {horizon} | "
        f"tuning_batch_size {tuning_size_index} ({trial_cfg.tuning.tune_batch_size}) ==\n"
        f"Used params | text_attn_layers={trial_cfg.model.text_attn_layers} | "
        f"epochs={trial_cfg.tuning.epochs} | gate_lr={trial_cfg.tuning.gate_lr} | "
        f"text_attn_lr={trial_cfg.tuning.text_attn_lr} | "
        f"gate_logit_clamp={trial_cfg.tuning.gate_logit_clamp} | "
        f"max_context={trial_cfg.tuning.max_context} | "
        f"target_lag_count={trial_spec_target_lag_count(trial_cfg)} | "
        f"covariate_lag_count={trial_spec_covariate_lag_count(trial_cfg)} | "
        f"embedding_lag_count={trial_spec_embedding_lag_count(trial_cfg)}"
    )
    set_random_seeds(trial_cfg.seed)
    prepared_trial = prepare_fine_tune_trial(
        trial_cfg,
        data_splits=data_splits,
        horizon=horizon,
    )
    before = _evaluate_all_splits(prepared_trial, tuning_cfg=trial_cfg.tuning)
    print(
        "Before tuning | "
        f"train_mae={before.train.mae:.6f} | val_mae={before.val.mae:.6f} | "
        f"test_mae={before.test.mae:.6f}"
    )
    fine_tune_outcome = fine_tune_prepared_trial(prepared_trial, tuning_cfg=trial_cfg.tuning)
    after = _evaluate_all_splits(prepared_trial, tuning_cfg=trial_cfg.tuning)
    ranking_score = objective_score(trial_cfg.tuning.early_stopping_metric, after.val)
    print(
        "After tuning | "
        f"best_epoch={fine_tune_outcome.best_epoch} | objective={ranking_score:.6f} | "
        f"train_mae={after.train.mae:.6f} | val_mae={after.val.mae:.6f} | "
        f"test_mae={after.test.mae:.6f}"
    )
    return PerHorizonTuneBatchResult(
        candidate_index=candidate_index,
        parent_index=parent_index,
        horizon=horizon,
        tuning_size_index=tuning_size_index,
        text_attn_layers=list(trial_cfg.model.text_attn_layers),
        epochs=trial_cfg.tuning.epochs,
        gate_lr=trial_cfg.tuning.gate_lr,
        text_attn_lr=trial_cfg.tuning.text_attn_lr,
        gate_logit_clamp=trial_cfg.tuning.gate_logit_clamp,
        tune_batch_size=trial_cfg.tuning.tune_batch_size,
        max_context=trial_cfg.tuning.max_context,
        target_lag_count=trial_spec_target_lag_count(trial_cfg),
        covariate_lag_count=trial_spec_covariate_lag_count(trial_cfg),
        embedding_lag_count=trial_spec_embedding_lag_count(trial_cfg),
        best_epoch=fine_tune_outcome.best_epoch,
        before=before,
        after=after,
        ranking_score=ranking_score,
    )


def _execute_per_horizon_candidate_with_logging(
    trial_cfg: DataConfig,
    *,
    candidate_index: int,
    parent_index: int,
    horizon: int,
    tuning_size_index: int,
    data_splits: FineTuneDataSplits,
    log_path: str | Path,
    assigned_device: str | None,
    tee_stdout: bool,
) -> PerHorizonTuneBatchExecutionArtifacts:
    resolved_trial_cfg = trial_cfg
    if assigned_device is not None and assigned_device != trial_cfg.model.device:
        resolved_trial_cfg = replace(
            trial_cfg,
            model=replace(trial_cfg.model, device=assigned_device),
        )

    log_path = Path(log_path)
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        streams: tuple[Any, ...] = (sys.stdout, log_file) if tee_stdout else (log_file,)
        tee_writer = _TeeWriter(*streams)
        with contextlib.redirect_stdout(tee_writer), contextlib.redirect_stderr(tee_writer):
            if assigned_device is not None:
                print(f"Assigned device | {assigned_device}")
            result = _execute_per_horizon_tune_batch_candidate(
                resolved_trial_cfg,
                candidate_index=candidate_index,
                parent_index=parent_index,
                horizon=horizon,
                tuning_size_index=tuning_size_index,
                data_splits=data_splits,
            )
    return PerHorizonTuneBatchExecutionArtifacts(
        result=result,
        assigned_device=assigned_device,
    )


def _run_per_horizon_candidate_batch(
    candidate_cfgs: dict[int, DataConfig],
    *,
    candidate_metadata: dict[int, tuple[int, int, int]],
    candidate_data_splits: dict[int, FineTuneDataSplits],
    logs_dir: Path,
    max_parallel_trials: int,
    gpu_ids: list[int] | None,
) -> list[PerHorizonTuneBatchResult]:
    candidate_items = list(candidate_cfgs.items())
    if not candidate_items:
        return []

    if max_parallel_trials <= 1:
        results: list[PerHorizonTuneBatchResult] = []
        for candidate_index, candidate_cfg in candidate_items:
            parent_index, horizon, tuning_size_index = candidate_metadata[candidate_index]
            artifacts = _execute_per_horizon_candidate_with_logging(
                candidate_cfg,
                candidate_index=candidate_index,
                parent_index=parent_index,
                horizon=horizon,
                tuning_size_index=tuning_size_index,
                data_splits=candidate_data_splits[candidate_index],
                log_path=logs_dir / f"candidate_{candidate_index:04d}.log",
                assigned_device=None,
                tee_stdout=True,
            )
            results.append(artifacts.result)
        return results

    device_slots = _build_parallel_device_slots(
        max_parallel_trials=max_parallel_trials,
        gpu_ids=gpu_ids,
        base_device=candidate_items[0][1].model.device,
    )
    print(
        "Parallel candidate execution enabled | "
        f"max_parallel_trials={max_parallel_trials} | device_slots={device_slots}"
    )
    results: list[PerHorizonTuneBatchResult] = []
    mp_context = multiprocessing.get_context("spawn")
    candidate_iter = iter(candidate_items)

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(device_slots),
        mp_context=mp_context,
    ) as executor:
        future_states: dict[concurrent.futures.Future[PerHorizonTuneBatchExecutionArtifacts], tuple[int, str | None]] = {}

        def submit_next(assigned_device: str | None) -> bool:
            try:
                next_candidate_index, next_candidate_cfg = next(candidate_iter)
            except StopIteration:
                return False
            parent_index, horizon, tuning_size_index = candidate_metadata[next_candidate_index]
            future = executor.submit(
                _execute_per_horizon_candidate_with_logging,
                next_candidate_cfg,
                candidate_index=next_candidate_index,
                parent_index=parent_index,
                horizon=horizon,
                tuning_size_index=tuning_size_index,
                data_splits=candidate_data_splits[next_candidate_index],
                log_path=str(logs_dir / f"candidate_{next_candidate_index:04d}.log"),
                assigned_device=assigned_device,
                tee_stdout=False,
            )
            future_states[future] = (next_candidate_index, assigned_device)
            print(
                f"Started candidate {next_candidate_index:04d} | parent={parent_index} | "
                f"horizon={horizon} | tuning_size_index={tuning_size_index} | "
                f"device={assigned_device or next_candidate_cfg.model.device or 'auto'}"
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
                submitted_candidate_index, assigned_device = future_states.pop(future)
                artifacts = future.result()
                result = artifacts.result
                if result.candidate_index != submitted_candidate_index:
                    raise RuntimeError(
                        "Parallel candidate bookkeeping mismatch: "
                        f"submitted candidate {submitted_candidate_index} returned {result.candidate_index}."
                    )
                results.append(result)
                print(
                    f"Completed candidate {result.candidate_index:04d} | parent={result.parent_index} | "
                    f"horizon={result.horizon} | tune_batch_size={result.tune_batch_size} | "
                    f"covariate_lag_count={result.covariate_lag_count} | "
                    f"objective={result.ranking_score:.6f} | val_mae={result.after.val.mae:.6f} | "
                    f"device={artifacts.assigned_device or assigned_device or 'auto'}"
                )
                submit_next(assigned_device)
    return results


def _per_horizon_candidate_sort_key(
    result: PerHorizonTuneBatchResult,
) -> tuple[float, float, int, int]:
    return (
        result.ranking_score,
        result.after.val.mae,
        result.parent_index,
        result.tuning_size_index,
    )


def _metric_to_dict(metrics: RollingMetrics) -> dict[str, float]:
    return {
        "loss": metrics.loss,
        "mae": metrics.mae,
        "rmse": metrics.rmse,
        "mape": metrics.mape,
        "real_mae": metrics.real_mae,
        "real_rmse": metrics.real_rmse,
        "real_mape": metrics.real_mape,
    }


def _split_bundle_to_dict(bundle: SplitMetricsBundle) -> dict[str, dict[str, float]]:
    return {
        "train": _metric_to_dict(bundle.train),
        "val": _metric_to_dict(bundle.val),
        "test": _metric_to_dict(bundle.test),
    }


def _serialize_per_horizon_candidate_result(
    result: PerHorizonTuneBatchResult,
    *,
    rank_in_horizon: int | None = None,
) -> dict[str, Any]:
    payload = {
        "candidate_index": result.candidate_index,
        "parent_index": result.parent_index,
        "horizon": result.horizon,
        "tuning_size_index": result.tuning_size_index,
        "text_attn_layers": result.text_attn_layers,
        "epochs": result.epochs,
        "gate_lr": result.gate_lr,
        "text_attn_lr": result.text_attn_lr,
        "gate_logit_clamp": result.gate_logit_clamp,
        "tune_batch_size": result.tune_batch_size,
        "max_context": result.max_context,
        "target_lag_count": result.target_lag_count,
        "covariate_lag_count": result.covariate_lag_count,
        "embedding_lag_count": result.embedding_lag_count,
        "best_epoch": result.best_epoch,
        "ranking_score": result.ranking_score,
        "is_best_for_horizon": result.is_best_for_horizon,
        "before": _split_bundle_to_dict(result.before),
        "after": _split_bundle_to_dict(result.after),
    }
    if rank_in_horizon is not None:
        payload["rank_in_horizon"] = rank_in_horizon
    return payload


def _candidate_csv_row(
    result: PerHorizonTuneBatchResult,
    *,
    rank_in_horizon: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "rank_in_horizon": rank_in_horizon,
        "candidate_index": result.candidate_index,
        "parent_index": result.parent_index,
        "horizon": result.horizon,
        "tuning_size_index": result.tuning_size_index,
        "is_best_for_horizon": result.is_best_for_horizon,
        "text_attn_layers": json.dumps(result.text_attn_layers),
        "epochs": result.epochs,
        "gate_lr": result.gate_lr,
        "text_attn_lr": result.text_attn_lr,
        "gate_logit_clamp": result.gate_logit_clamp,
        "tune_batch_size": result.tune_batch_size,
        "max_context": result.max_context,
        "target_lag_count": result.target_lag_count,
        "covariate_lag_count": result.covariate_lag_count,
        "embedding_lag_count": result.embedding_lag_count,
        "best_epoch": result.best_epoch,
        "ranking_score": result.ranking_score,
    }
    for phase_name, bundle in (("before", result.before), ("after", result.after)):
        for split_name in ("train", "val", "test"):
            metrics = getattr(bundle, split_name)
            for metric_name, metric_value in _metric_to_dict(metrics).items():
                row[f"{phase_name}_{split_name}_{metric_name}"] = metric_value
    return row


def _write_per_horizon_trial_results_csv(
    output_dir: Path,
    *,
    ranked_by_horizon: dict[int, list[PerHorizonTuneBatchResult]],
) -> None:
    fieldnames = [
        "rank_in_horizon",
        "candidate_index",
        "parent_index",
        "horizon",
        "tuning_size_index",
        "is_best_for_horizon",
        "text_attn_layers",
        "epochs",
        "gate_lr",
        "text_attn_lr",
        "gate_logit_clamp",
        "tune_batch_size",
        "max_context",
        "target_lag_count",
        "covariate_lag_count",
        "embedding_lag_count",
        "best_epoch",
        "ranking_score",
    ]
    for phase_name in ("before", "after"):
        for split_name in ("train", "val", "test"):
            for metric_name in ("loss", "mae", "rmse", "mape", "real_mae", "real_rmse", "real_mape"):
                fieldnames.append(f"{phase_name}_{split_name}_{metric_name}")

    with (output_dir / "trial_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for horizon in sorted(ranked_by_horizon):
            for rank, result in enumerate(ranked_by_horizon[horizon], start=1):
                writer.writerow(_candidate_csv_row(result, rank_in_horizon=rank))


def _parent_params_to_dict(result: PerHorizonTuneBatchResult) -> dict[str, Any]:
    return {
        "text_attn_layers": result.text_attn_layers,
        "epochs": result.epochs,
        "gate_lr": result.gate_lr,
        "text_attn_lr": result.text_attn_lr,
        "gate_logit_clamp": result.gate_logit_clamp,
        "max_context": result.max_context,
        "target_lag_count": result.target_lag_count,
        "covariate_lag_count": result.covariate_lag_count,
        "embedding_lag_count": result.embedding_lag_count,
    }


def _write_per_horizon_summary_json(
    output_dir: Path,
    *,
    random_search_cfg: RandomSearchConfig,
    objective_metric: str,
    parent_total_combinations: int,
    parent_specs: list[RandomSearchTrialSpec],
    tune_batch_size_choices: list[int],
    results: list[PerHorizonTuneBatchResult],
    ranked_by_horizon: dict[int, list[PerHorizonTuneBatchResult]],
    shared_baseline: SharedBaselineResult,
) -> None:
    results_by_parent: dict[int, list[PerHorizonTuneBatchResult]] = {}
    for result in sorted(results, key=lambda item: (item.parent_index, item.horizon, item.tuning_size_index)):
        results_by_parent.setdefault(result.parent_index, []).append(result)

    summary = {
        "base_config": random_search_cfg.base_config,
        "base_dataset": random_search_cfg.base_dataset,
        "objective_metric": objective_metric,
        "max_parallel_trials": random_search_cfg.max_parallel_trials,
        "gpu_ids": random_search_cfg.gpu_ids,
        "baseline_evaluations_skipped_per_trial": True,
        "shared_baseline_evaluated": True,
        "shared_baseline": _serialize_shared_baseline_result(shared_baseline),
        "shared_param_trials_requested": random_search_cfg.trials,
        "shared_param_trials_evaluated": len(parent_specs),
        "parent_total_combinations": parent_total_combinations,
        "tune_batch_size_choices": tune_batch_size_choices,
        "candidate_evaluations": len(results),
        "top_k_per_horizon": min(random_search_cfg.top_k, max((len(items) for items in ranked_by_horizon.values()), default=0)),
        "best_by_horizon": {
            str(horizon): _serialize_per_horizon_candidate_result(ranked_results[0], rank_in_horizon=1)
            for horizon, ranked_results in sorted(ranked_by_horizon.items())
            if ranked_results
        },
        "top_by_horizon": {
            str(horizon): [
                _serialize_per_horizon_candidate_result(result, rank_in_horizon=rank)
                for rank, result in enumerate(ranked_results[: random_search_cfg.top_k], start=1)
            ]
            for horizon, ranked_results in sorted(ranked_by_horizon.items())
        },
        "per_parent_results": [
            {
                "parent_index": parent_index,
                "used_params": _parent_params_to_dict(parent_results[0]),
                "candidates": [
                    _serialize_per_horizon_candidate_result(result)
                    for result in parent_results
                ],
            }
            for parent_index, parent_results in sorted(results_by_parent.items())
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _format_metric_triplet(label: str, metrics: RollingMetrics) -> str:
    return (
        f"{label}_loss={metrics.loss:.6f} | "
        f"{label}_mae={metrics.mae:.6f} | "
        f"{label}_rmse={metrics.rmse:.6f} | "
        f"{label}_mape={metrics.mape:.6f}% | "
        f"{label}_real_mae={metrics.real_mae:.6f} | "
        f"{label}_real_rmse={metrics.real_rmse:.6f} | "
        f"{label}_real_mape={metrics.real_mape:.6f}%"
    )


def _format_candidate_summary_line(result: PerHorizonTuneBatchResult) -> str:
    return (
        f"tuning_batch_size {result.tuning_size_index}/{result.tune_batch_size} | "
        f"candidate_index={result.candidate_index} | best_for_horizon={str(result.is_best_for_horizon).lower()} | "
        f"best_epoch={result.best_epoch} | objective={result.ranking_score:.6f} | "
        f"{_format_metric_triplet('before_train', result.before.train)} | "
        f"{_format_metric_triplet('before_val', result.before.val)} | "
        f"{_format_metric_triplet('before_test', result.before.test)} | "
        f"{_format_metric_triplet('after_train', result.after.train)} | "
        f"{_format_metric_triplet('after_val', result.after.val)} | "
        f"{_format_metric_triplet('after_test', result.after.test)}"
    )


def _write_per_horizon_summary_log(
    output_dir: Path,
    *,
    random_search_cfg: RandomSearchConfig,
    objective_metric: str,
    parent_total_combinations: int,
    parent_specs: list[RandomSearchTrialSpec],
    tune_batch_size_choices: list[int],
    results: list[PerHorizonTuneBatchResult],
    ranked_by_horizon: dict[int, list[PerHorizonTuneBatchResult]],
    shared_baseline: SharedBaselineResult,
) -> None:
    results_by_parent_horizon: dict[int, dict[int, list[PerHorizonTuneBatchResult]]] = {}
    for result in sorted(results, key=lambda item: (item.parent_index, item.horizon, item.tuning_size_index)):
        results_by_parent_horizon.setdefault(result.parent_index, {}).setdefault(result.horizon, []).append(result)

    lines = [
        "Random Search Summary",
        f"base_config: {random_search_cfg.base_config}",
        f"base_dataset: {random_search_cfg.base_dataset}",
        f"objective_metric: {objective_metric}",
        f"max_parallel_trials: {random_search_cfg.max_parallel_trials}",
        f"gpu_ids: {random_search_cfg.gpu_ids}",
        "baseline_evaluations_skipped_per_trial: true",
        "shared_baseline_evaluated: true",
        f"shared_param_trials_requested: {random_search_cfg.trials}",
        f"shared_param_trials_evaluated: {len(parent_specs)}",
        f"parent_total_combinations: {parent_total_combinations}",
        f"tune_batch_size_choices: {tune_batch_size_choices}",
        f"candidate_evaluations: {len(results)}",
        "",
        "Shared Baseline",
        (
            f"params | text_attn_layers={shared_baseline.text_attn_layers} | "
            f"max_context={shared_baseline.max_context}"
        ),
    ]
    if shared_baseline.per_horizon is not None:
        lines.append("Shared Baseline By Horizon")
        for horizon_result in shared_baseline.per_horizon:
            lines.extend(_format_per_horizon_baseline_lines(horizon_result))
    lines.append("")

    for parent_index in sorted(results_by_parent_horizon):
        first_result = next(iter(next(iter(results_by_parent_horizon[parent_index].values()))))
        lines.append(
            f"Shared parameter set {parent_index}: Used params | "
            f"text_attn_layers={first_result.text_attn_layers} | "
            f"epochs={first_result.epochs} | gate_lr={first_result.gate_lr} | "
            f"text_attn_lr={first_result.text_attn_lr} | "
            f"gate_logit_clamp={first_result.gate_logit_clamp} | "
            f"max_context={first_result.max_context} | "
            f"target_lag_count={first_result.target_lag_count} | "
            f"embedding_lag_count={first_result.embedding_lag_count}"
        )
        for horizon in sorted(results_by_parent_horizon[parent_index]):
            lines.append(f"Model horizon {horizon}:")
            for result in results_by_parent_horizon[parent_index][horizon]:
                lines.append(_format_candidate_summary_line(result))
            lines.append("")

    lines.append("Best By Horizon")
    for horizon, ranked_results in sorted(ranked_by_horizon.items()):
        best = ranked_results[0]
        lines.append(
            f"horizon={horizon} | parent_index={best.parent_index} | "
            f"tune_batch_size={best.tune_batch_size} | best_epoch={best.best_epoch} | "
            f"objective={best.ranking_score:.6f} | val_mae={best.after.val.mae:.6f} | "
            f"test_mae={best.after.test.mae:.6f}"
        )
    lines.append("")

    for horizon, ranked_results in sorted(ranked_by_horizon.items()):
        top_limit = min(random_search_cfg.top_k, len(ranked_results))
        lines.append(f"Top {top_limit} Candidates For Horizon {horizon}")
        for rank, result in enumerate(ranked_results[:top_limit], start=1):
            lines.append(
                f"{rank}. candidate_index={result.candidate_index} | parent_index={result.parent_index} | "
                f"tune_batch_size={result.tune_batch_size} | best_epoch={result.best_epoch} | "
                f"objective={result.ranking_score:.6f} | val_mae={result.after.val.mae:.6f} | "
                f"test_mae={result.after.test.mae:.6f}"
            )
        lines.append("")

    with (output_dir / "summary.log").open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_best_fine_tune_configs_by_horizon(
    output_dir: Path,
    *,
    candidate_cfgs: dict[int, DataConfig],
    ranked_by_horizon: dict[int, list[PerHorizonTuneBatchResult]],
) -> dict[int, DataConfig]:
    best_cfgs: dict[int, DataConfig] = {}
    for horizon, ranked_results in sorted(ranked_by_horizon.items()):
        best = ranked_results[0]
        best_cfg = candidate_cfgs[best.candidate_index]
        best_cfgs[horizon] = best_cfg
        with (output_dir / f"best_fine_tune_config_horizon_{horizon:02d}.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(best_cfg), f, sort_keys=False)
    if 1 in best_cfgs:
        with (output_dir / "best_fine_tune_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(best_cfgs[1]), f, sort_keys=False)
    return best_cfgs


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
    lag_search_requested = (
        random_search_cfg.search_space.target_lag_count is not None
        or random_search_cfg.search_space.covariate_lag_count is not None
        or random_search_cfg.search_space.embedding_lag_count is not None
    )
    lag_search_metadata = (
        discover_lag_search_metadata(base_run_cfg)
        if lag_search_requested
        else None
    )

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
    if lag_search_requested:
        print(
            "Lag search enabled | "
            f"target_lag_count_choices={random_search_cfg.search_space.target_lag_count} | "
            f"covariate_lag_count_choices={random_search_cfg.search_space.covariate_lag_count} | "
            f"embedding_lag_count_choices={random_search_cfg.search_space.embedding_lag_count}"
        )

    parent_specs, parent_total_combinations = sample_trial_specs(
        base_run_cfg,
        random_search_cfg,
        lag_search_metadata=lag_search_metadata,
    )
    tune_batch_size_choices = build_tune_batch_size_choices(base_run_cfg, random_search_cfg.search_space)
    if random_search_cfg.trials > parent_total_combinations:
        print(
            f"Requested {random_search_cfg.trials} trials but search space has only "
            f"{parent_total_combinations} parent combinations; clamping to {parent_total_combinations}."
        )
    candidate_count = len(parent_specs) * base_run_cfg.prediction_window * len(tune_batch_size_choices)
    print(
        f"Random search space | parent_total_combinations={parent_total_combinations} | "
        f"running_shared_param_sets={len(parent_specs)} | "
        f"tune_batch_size_choices={tune_batch_size_choices} | "
        f"candidate_evaluations={candidate_count} | seed={random_search_cfg.seed}"
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

    parent_cfgs: dict[int, DataConfig] = {}
    parent_data_splits: dict[int, FineTuneDataSplits] = {}
    data_splits_cache: dict[tuple[int | None, int | None, int | None], FineTuneDataSplits] = {
        _trial_data_cache_key(target_lag_count=None, covariate_lag_count=None, embedding_lag_count=None): data_splits
    }
    for parent_index, parent_spec in enumerate(parent_specs, start=1):
        parent_cfg = resolve_trial_config(
            base_run_cfg,
            parent_spec,
            lag_search_metadata=lag_search_metadata,
        )
        parent_cfgs[parent_index] = parent_cfg
        data_key = _trial_data_cache_key(
            target_lag_count=parent_spec.target_lag_count,
            covariate_lag_count=parent_spec.covariate_lag_count,
            embedding_lag_count=parent_spec.embedding_lag_count,
        )
        if data_key not in data_splits_cache:
            data_splits_cache[data_key] = load_and_split_fine_tune_data(parent_cfg)
        parent_data_splits[parent_index] = data_splits_cache[data_key]

    candidate_cfgs: dict[int, DataConfig] = {}
    candidate_metadata: dict[int, tuple[int, int, int]] = {}
    candidate_data_splits: dict[int, FineTuneDataSplits] = {}
    candidate_index = 0
    for parent_index, parent_cfg in parent_cfgs.items():
        for horizon in range(1, base_run_cfg.prediction_window + 1):
            for tuning_size_index, tune_batch_size in enumerate(tune_batch_size_choices, start=1):
                candidate_index += 1
                candidate_cfgs[candidate_index] = replace(
                    parent_cfg,
                    tuning=replace(
                        parent_cfg.tuning,
                        tune_batch_size=tune_batch_size,
                    ),
                )
                candidate_metadata[candidate_index] = (parent_index, horizon, tuning_size_index)
                candidate_data_splits[candidate_index] = parent_data_splits[parent_index]

    results = _run_per_horizon_candidate_batch(
        candidate_cfgs,
        candidate_metadata=candidate_metadata,
        candidate_data_splits=candidate_data_splits,
        logs_dir=logs_dir,
        max_parallel_trials=random_search_cfg.max_parallel_trials,
        gpu_ids=random_search_cfg.gpu_ids,
    )

    if not results:
        raise RuntimeError("Random search did not execute any candidate evaluations.")

    results_by_horizon: dict[int, list[PerHorizonTuneBatchResult]] = {}
    for result in results:
        results_by_horizon.setdefault(result.horizon, []).append(result)
    ranked_by_horizon = {
        horizon: sorted(horizon_results, key=_per_horizon_candidate_sort_key)
        for horizon, horizon_results in sorted(results_by_horizon.items())
    }
    best_candidate_indices = {
        ranked_results[0].candidate_index
        for ranked_results in ranked_by_horizon.values()
        if ranked_results
    }
    results = [
        replace(result, is_best_for_horizon=result.candidate_index in best_candidate_indices)
        for result in results
    ]
    results_by_candidate = {result.candidate_index: result for result in results}
    ranked_by_horizon = {
        horizon: [results_by_candidate[result.candidate_index] for result in ranked_results]
        for horizon, ranked_results in ranked_by_horizon.items()
    }

    print("\n== Random Search Best Candidates By Horizon ==")
    for horizon, ranked_results in sorted(ranked_by_horizon.items()):
        best = ranked_results[0]
        print(
            f"horizon={horizon} | parent_index={best.parent_index} | "
            f"candidate_index={best.candidate_index} | tune_batch_size={best.tune_batch_size} | "
            f"best_epoch={best.best_epoch} | objective={best.ranking_score:.6f} | "
            f"val_mae={best.after.val.mae:.6f} | test_mae={best.after.test.mae:.6f}"
        )

    best_run_cfgs = _write_best_fine_tune_configs_by_horizon(
        output_dir,
        candidate_cfgs=candidate_cfgs,
        ranked_by_horizon=ranked_by_horizon,
    )
    _write_per_horizon_trial_results_csv(output_dir, ranked_by_horizon=ranked_by_horizon)
    _write_per_horizon_summary_json(
        output_dir,
        random_search_cfg=random_search_cfg,
        objective_metric=base_run_cfg.tuning.early_stopping_metric,
        parent_total_combinations=parent_total_combinations,
        parent_specs=parent_specs,
        tune_batch_size_choices=tune_batch_size_choices,
        results=results,
        ranked_by_horizon=ranked_by_horizon,
        shared_baseline=shared_baseline,
    )
    _write_per_horizon_summary_log(
        output_dir,
        random_search_cfg=random_search_cfg,
        objective_metric=base_run_cfg.tuning.early_stopping_metric,
        parent_total_combinations=parent_total_combinations,
        parent_specs=parent_specs,
        tune_batch_size_choices=tune_batch_size_choices,
        results=results,
        ranked_by_horizon=ranked_by_horizon,
        shared_baseline=shared_baseline,
    )
    first_best = ranked_by_horizon[min(ranked_by_horizon)][0]
    return {
        "output_dir": str(output_dir),
        "logs_dir": str(logs_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "candidate_results": results,
        "ranked_by_horizon": ranked_by_horizon,
        "shared_baseline": shared_baseline,
        "best_test_metrics": RollingMetrics(
            loss=first_best.after.test.loss,
            mae=first_best.after.test.mae,
            rmse=first_best.after.test.rmse,
            mape=first_best.after.test.mape,
            real_mae=first_best.after.test.real_mae,
            real_rmse=first_best.after.test.real_rmse,
            real_mape=first_best.after.test.real_mape,
        ),
        "best_run_cfg": best_run_cfgs.get(1, next(iter(best_run_cfgs.values()))),
        "best_run_cfgs_by_horizon": best_run_cfgs,
        "shared_param_trial_specs": parent_specs,
        "tune_batch_size_choices": tune_batch_size_choices,
        "parent_total_combinations": parent_total_combinations,
        "total_combinations": parent_total_combinations,
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
