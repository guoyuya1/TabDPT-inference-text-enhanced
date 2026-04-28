"""Feature search runner for TabDPT baseline without text attention."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import json
import multiprocessing
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fine_tuning.eval_fine_tune import _format_dual_metrics, evaluate_rolling  # noqa: E402
from fine_tuning.feature_search_config import FeatureSearchDataConfig, load_feature_search_config  # noqa: E402
from fine_tuning.load_dataset import validate_direct_mode_numeric_features  # noqa: E402

try:  # pragma: no cover - import path depends on execution mode
    from .tabdpt_hpo_baseline import (  # type: ignore[attr-defined]  # noqa: E402
        FineTuneDataSplits,
        HorizonResult,
        SummaryMetrics,
        _TeeWriter,
        _build_history_and_eval_split,
        _level_eval_context,
        _make_tabdpt,
        _apply_target_differencing_mode,
        _mean_summary_metrics,
        _reset_compiler_state,
        _select_horizon_splits,
        _serialize_summary_metrics,
        _summary_metrics_from_dual,
        maybe_normalize_fine_tune_splits,
        prediction_horizons,
        preprocess_features,
        split_fine_tune_data,
    )
except ImportError:  # pragma: no cover
    from tabdpt_hpo_baseline import (  # noqa: E402
        FineTuneDataSplits,
        HorizonResult,
        SummaryMetrics,
        _TeeWriter,
        _build_history_and_eval_split,
        _level_eval_context,
        _make_tabdpt,
        _apply_target_differencing_mode,
        _mean_summary_metrics,
        _reset_compiler_state,
        _select_horizon_splits,
        _serialize_summary_metrics,
        _summary_metrics_from_dual,
        maybe_normalize_fine_tune_splits,
        prediction_horizons,
        preprocess_features,
        split_fine_tune_data,
    )


LAG_FEATURE_PATTERN = re.compile(r"(.+)_lag(\d+)$")


@dataclass(frozen=True)
class CovariateLagFamily:
    prefix: str
    base_lag_count: int
    available_lag_count: int


@dataclass(frozen=True)
class FeatureLagMetadata:
    target_lag_prefix: str
    base_target_lag_count: int | None
    available_target_lag_count: int
    covariate_families: list[CovariateLagFamily]
    shared_covariate_lag_count: int


@dataclass(frozen=True)
class FeatureSetResult:
    max_context: int | None
    target_lag_count: int
    covariate_lag_count: int
    numeric_features: list[str]
    val: SummaryMetrics
    test: SummaryMetrics
    per_horizon: list[HorizonResult]


@dataclass(frozen=True)
class FeatureSetJob:
    target_lag_count: int
    covariate_lag_count: int
    numeric_features: list[str]


@dataclass(frozen=True)
class FeatureSetExecutionArtifacts:
    results: list[FeatureSetResult]
    assigned_device: str | None


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (SCRIPT_DIR / path).resolve()


def _extract_gpu_id_from_device(device: str | None) -> int | None:
    if not isinstance(device, str) or not device.startswith("cuda:"):
        return None
    _, _, gpu_id = device.partition(":")
    if not gpu_id.isdigit():
        raise ValueError(f"Unsupported CUDA device specifier: {device!r}")
    return int(gpu_id)


def _build_parallel_device_slots(
    *,
    max_parallel_trials_per_gpu: int,
    gpu_ids: list[int] | None,
    base_device: str | None,
) -> list[str | None]:
    if max_parallel_trials_per_gpu <= 0:
        raise ValueError("max_parallel_trials_per_gpu must be positive.")

    normalized_base_device = base_device.lower() if isinstance(base_device, str) else None
    if normalized_base_device == "cpu":
        return ["cpu"] * max_parallel_trials_per_gpu

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
            f"cuda:{gpu_id}"
            for gpu_id in resolved_gpu_ids
            for _ in range(max_parallel_trials_per_gpu)
        ]

    fallback_device = base_device if base_device not in {None, "auto"} else None
    return [fallback_device] * max_parallel_trials_per_gpu


def _read_csv_header(path: str) -> list[str]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header is None:
        raise ValueError(f"CSV file has no header row: {path}")
    return header


def load_feature_search_arrays(
    run_cfg: FeatureSearchDataConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    validate_direct_mode_numeric_features(run_cfg.numeric_features)

    df = pd.read_csv(run_cfg.data_path)
    timestamps: np.ndarray | None = None
    if run_cfg.date_column and run_cfg.date_column in df.columns:
        df[run_cfg.date_column] = pd.to_datetime(df[run_cfg.date_column])
        df = df.sort_values(run_cfg.date_column).reset_index(drop=True)
        timestamps = df[run_cfg.date_column].to_numpy()

    if run_cfg.max_rows is not None:
        df = df.head(run_cfg.max_rows).reset_index(drop=True)
        if timestamps is not None:
            timestamps = timestamps[: run_cfg.max_rows]

    for col in [*run_cfg.numeric_features, run_cfg.target_column]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {run_cfg.data_path}")

    X = df[run_cfg.numeric_features].astype(np.float32).to_numpy()
    y = df[run_cfg.target_column].astype(np.float32).to_numpy()
    text = np.zeros((len(df), 0, 0), dtype=np.float32)
    y_level: np.ndarray | None = None

    if run_cfg.target_mode == "target_differencing":
        X, y, text, timestamps, y_level = _apply_target_differencing_mode(
            X=X,
            y=y,
            text=text,
            timestamps=timestamps,
            numeric_features=run_cfg.numeric_features,
            target_column=run_cfg.target_column,
        )

    return X, y, text, timestamps, y_level


def load_and_split_feature_search_data(run_cfg: FeatureSearchDataConfig) -> FineTuneDataSplits:
    X, y, text, timestamps, y_level = load_feature_search_arrays(run_cfg)
    return split_fine_tune_data(run_cfg, X=X, y=y, text=text, timestamps=timestamps, y_level=y_level)


def _contiguous_lag_count(lag_values: set[int]) -> int:
    count = 0
    while count + 1 in lag_values:
        count += 1
    return count


def _feature_family_name(feature_name: str) -> str:
    match = LAG_FEATURE_PATTERN.fullmatch(feature_name)
    if match is None:
        return feature_name
    return match.group(1)


def _available_lag_count_for_prefix(header: list[str], prefix: str) -> int:
    return _contiguous_lag_count(
        {
            int(match.group(2))
            for column in header
            if (match := LAG_FEATURE_PATTERN.fullmatch(column)) is not None and match.group(1) == prefix
        }
    )


def discover_feature_lag_metadata(run_cfg: FeatureSearchDataConfig) -> FeatureLagMetadata:
    header = _read_csv_header(run_cfg.data_path)
    target_lag_prefix = "target_diff" if run_cfg.target_mode == "target_differencing" else run_cfg.target_column
    available_target_lag_count = _available_lag_count_for_prefix(header, target_lag_prefix)

    configured_covariate_lags: dict[str, set[int]] = {}
    covariate_order: list[str] = []
    raw_covariate_families: set[str] = set()
    for feature_name in run_cfg.numeric_features:
        family_name = _feature_family_name(feature_name)
        if family_name == target_lag_prefix:
            continue
        if family_name not in configured_covariate_lags:
            configured_covariate_lags[family_name] = set()
            covariate_order.append(family_name)
        match = LAG_FEATURE_PATTERN.fullmatch(feature_name)
        if match is None:
            raw_covariate_families.add(family_name)
            continue
        configured_covariate_lags[family_name].add(int(match.group(2)))

    covariate_families: list[CovariateLagFamily] = []
    for prefix in covariate_order:
        available_lag_count = _available_lag_count_for_prefix(header, prefix)
        configured_lags = configured_covariate_lags[prefix]
        base_lag_count = available_lag_count if prefix in raw_covariate_families else _contiguous_lag_count(configured_lags)
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
    return FeatureLagMetadata(
        target_lag_prefix=target_lag_prefix,
        base_target_lag_count=available_target_lag_count,
        available_target_lag_count=available_target_lag_count,
        covariate_families=covariate_families,
        shared_covariate_lag_count=shared_covariate_lag_count,
    )


def validate_feature_search_config(run_cfg: FeatureSearchDataConfig) -> FeatureLagMetadata:
    metadata = discover_feature_lag_metadata(run_cfg)

    invalid_target_counts = sorted(
        {
            count
            for count in run_cfg.feature_search.target_lag_count
            if count > metadata.available_target_lag_count
        }
    )
    if invalid_target_counts:
        raise ValueError(
            "feature_search.target_lag_count requested values "
            f"{invalid_target_counts}, but only contiguous target lags up to "
            f"lag{metadata.available_target_lag_count} are available."
        )

    if not metadata.covariate_families:
        if any(count != 0 for count in run_cfg.feature_search.covariate_lag_count):
            raise ValueError(
                "feature_search.covariate_lag_count requires at least one non-target lagged numeric feature family."
            )
        return metadata

    invalid_covariate_counts = sorted(
        {
            count
            for count in run_cfg.feature_search.covariate_lag_count
            if count > metadata.shared_covariate_lag_count
        }
    )
    if invalid_covariate_counts:
        raise ValueError(
            "feature_search.covariate_lag_count requested values "
            f"{invalid_covariate_counts}, but the shortest non-target lag family only supports "
            f"contiguous lags up to lag{metadata.shared_covariate_lag_count}."
        )
    return metadata


def resolve_feature_search_numeric_features(
    run_cfg: FeatureSearchDataConfig,
    *,
    target_lag_count: int,
    covariate_lag_count: int,
    lag_metadata: FeatureLagMetadata,
) -> list[str]:
    covariate_prefixes = {family.prefix for family in lag_metadata.covariate_families}
    resolved_numeric_features: list[str] = []
    inserted_covariates: set[str] = set()
    saw_target_family = False

    for feature_name in run_cfg.numeric_features:
        family_name = _feature_family_name(feature_name)
        if family_name == lag_metadata.target_lag_prefix:
            saw_target_family = True
            if target_lag_count > 0:
                resolved_numeric_features.extend(
                    f"{lag_metadata.target_lag_prefix}_lag{lag}" for lag in range(1, target_lag_count + 1)
                )
            continue

        if family_name in covariate_prefixes:
            if family_name not in inserted_covariates:
                resolved_numeric_features.extend(
                    f"{family_name}_lag{lag}" for lag in range(1, covariate_lag_count + 1)
                )
                inserted_covariates.add(family_name)
            continue

        resolved_numeric_features.append(feature_name)

    if not saw_target_family:
        raise ValueError(
            "Unable to resolve target lag block because the base config does not contain "
            f"any '{lag_metadata.target_lag_prefix}_lagN' numeric features."
        )
    return resolved_numeric_features


def _serialize_lag_metadata(metadata: FeatureLagMetadata) -> dict[str, Any]:
    return {
        "target_lag_prefix": metadata.target_lag_prefix,
        "base_target_lag_count": metadata.base_target_lag_count,
        "available_target_lag_count": metadata.available_target_lag_count,
        "shared_covariate_lag_count": metadata.shared_covariate_lag_count,
        "covariate_families": [
            {
                "prefix": family.prefix,
                "base_lag_count": family.base_lag_count,
                "available_lag_count": family.available_lag_count,
            }
            for family in metadata.covariate_families
        ],
    }


def _feature_set_sort_key(result: FeatureSetResult) -> tuple[float, int, int]:
    return (result.val.real_mae, result.target_lag_count, result.covariate_lag_count)


def _print_feature_set_summary(result: FeatureSetResult) -> None:
    print(
        "Feature-set mean metrics | "
        f"max_context={result.max_context} | "
        f"target_lag_count={result.target_lag_count} | "
        f"covariate_lag_count={result.covariate_lag_count}"
    )
    _format_dual_metrics(
        "val_no_text",
        (result.val.mae, result.val.rmse, result.val.mape),
        (result.val.real_mae, result.val.real_rmse, result.val.real_mape),
    )
    _format_dual_metrics(
        "test_no_text",
        (result.test.mae, result.test.rmse, result.test.mape),
        (result.test.real_mae, result.test.real_rmse, result.test.real_mape),
    )


def _print_rankings(results: list[FeatureSetResult], *, max_context: int | None) -> None:
    print(f"\nRanking for max_context={max_context}")
    ranked = sorted(
        (result for result in results if result.max_context == max_context),
        key=_feature_set_sort_key,
    )
    for rank, result in enumerate(ranked, start=1):
        print(
            f"{rank:02d}. target_lag_count={result.target_lag_count} | "
            f"covariate_lag_count={result.covariate_lag_count} | "
            f"val_real_mae={result.val.real_mae:.6f} | "
            f"test_real_mae={result.test.real_mae:.6f}"
        )


def evaluate_feature_set(
    run_cfg: FeatureSearchDataConfig,
    data_splits: FineTuneDataSplits,
    *,
    target_lag_count: int,
    covariate_lag_count: int,
    max_context_values: list[int | None],
) -> list[FeatureSetResult]:
    feature_search_cfg = run_cfg.feature_search
    per_max_context_horizons: dict[int | None, list[HorizonResult]] = {
        max_context: [] for max_context in max_context_values
    }

    for horizon in prediction_horizons(run_cfg):
        print(
            f"\n-- Horizon {horizon}/{run_cfg.prediction_window} | "
            f"target_lag_count={target_lag_count} | "
            f"covariate_lag_count={covariate_lag_count} --"
        )
        raw_splits = _select_horizon_splits(
            data_splits,
            horizon=horizon,
            include_calendar_features=feature_search_cfg.include_calendar_features,
            include_seasonal_features=feature_search_cfg.include_seasonal_features,
        )
        normalized_splits, target_scaler = maybe_normalize_fine_tune_splits(
            raw_splits,
            normalize=feature_search_cfg.normalize,
        )

        prepared: dict[int | None, dict[str, Any]] = {}
        for max_context in max_context_values:
            _reset_compiler_state()
            reg = _make_tabdpt(run_cfg)
            reg.fit(normalized_splits.X_context, normalized_splits.y_context)
            prepared[max_context] = {
                "reg": reg,
                "feature_splits": FineTuneDataSplits(
                    X_context=preprocess_features(reg, normalized_splits.X_context, reduction_mode=None, reduction_payload=None),
                    y_context=normalized_splits.y_context,
                    text_context=normalized_splits.text_context,
                    X_train=preprocess_features(reg, normalized_splits.X_train, reduction_mode=None, reduction_payload=None),
                    y_train=normalized_splits.y_train,
                    text_train=normalized_splits.text_train,
                    X_val=preprocess_features(reg, normalized_splits.X_val, reduction_mode=None, reduction_payload=None),
                    y_val=normalized_splits.y_val,
                    text_val=normalized_splits.text_val,
                    X_test=preprocess_features(reg, normalized_splits.X_test, reduction_mode=None, reduction_payload=None),
                    y_test=normalized_splits.y_test,
                    text_test=normalized_splits.text_test,
                    y_level_context=raw_splits.y_level_context,
                    y_level_train=raw_splits.y_level_train,
                    y_level_val=raw_splits.y_level_val,
                    y_level_test=raw_splits.y_level_test,
                ),
                "metrics": {},
            }

        for max_context, prepared_item in prepared.items():
            reg = prepared_item["reg"]
            feature_splits = prepared_item["feature_splits"]
            for split_name in ("val", "test"):
                X_hist, y_hist, text_hist, X_eval, y_eval, text_eval = _build_history_and_eval_split(
                    feature_splits,
                    split_name=split_name,
                )
                _, _, _, _, y_eval_real, _ = _build_history_and_eval_split(raw_splits, split_name=split_name)
                _, previous_level = _level_eval_context(raw_splits, split_name=split_name)
                y_eval_level = raw_splits.y_level_val if split_name == "val" else raw_splits.y_level_test
                prepared_item["metrics"][split_name] = evaluate_rolling(
                    reg,
                    X_context_proc=X_hist,
                    y_context=y_hist,
                    text_context=text_hist,
                    X_eval_proc=X_eval,
                    y_eval=y_eval,
                    y_eval_real=y_eval_real,
                    text_eval=text_eval,
                    use_text=False,
                    label=(
                        f"{split_name} h={horizon} max_context={max_context} "
                        f"target_lag_count={target_lag_count} covariate_lag_count={covariate_lag_count}"
                    ),
                    max_context=max_context,
                    target_scaler=target_scaler,
                    horizon=horizon,
                    y_eval_level=y_eval_level,
                    previous_level=previous_level,
                )

            val_summary = _summary_metrics_from_dual(prepared_item["metrics"]["val"])
            test_summary = _summary_metrics_from_dual(prepared_item["metrics"]["test"])
            print(
                f"max_context={max_context} | "
                f"target_lag_count={target_lag_count} | "
                f"covariate_lag_count={covariate_lag_count}"
            )
            _format_dual_metrics(
                "val_no_text",
                (val_summary.mae, val_summary.rmse, val_summary.mape),
                (val_summary.real_mae, val_summary.real_rmse, val_summary.real_mape),
            )
            _format_dual_metrics(
                "test_no_text",
                (test_summary.mae, test_summary.rmse, test_summary.mape),
                (test_summary.real_mae, test_summary.real_rmse, test_summary.real_mape),
            )
            per_max_context_horizons[max_context].append(
                HorizonResult(
                    horizon=horizon,
                    val=val_summary,
                    test=test_summary,
                )
            )

    return [
        FeatureSetResult(
            max_context=max_context,
            target_lag_count=target_lag_count,
            covariate_lag_count=covariate_lag_count,
            numeric_features=list(run_cfg.numeric_features),
            val=_mean_summary_metrics([item.val for item in per_horizon]),
            test=_mean_summary_metrics([item.test for item in per_horizon]),
            per_horizon=per_horizon,
        )
        for max_context, per_horizon in per_max_context_horizons.items()
    ]


def _execute_feature_set_with_logging(
    run_cfg: FeatureSearchDataConfig,
    *,
    data_splits: FineTuneDataSplits,
    target_lag_count: int,
    covariate_lag_count: int,
    max_context_values: list[int | None],
    feature_set_log_path: str | Path,
    assigned_device: str | None,
    tee_stdout: bool,
) -> FeatureSetExecutionArtifacts:
    resolved_run_cfg = run_cfg
    if assigned_device is not None and assigned_device != run_cfg.model.device:
        resolved_run_cfg = replace(
            run_cfg,
            model=replace(run_cfg.model, device=assigned_device),
        )

    feature_set_log_path = Path(feature_set_log_path)
    with feature_set_log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        streams: tuple[Any, ...] = (sys.stdout, log_file) if tee_stdout else (log_file,)
        tee_writer = _TeeWriter(*streams)
        with contextlib.redirect_stdout(tee_writer), contextlib.redirect_stderr(tee_writer):
            if assigned_device is not None:
                print(f"Assigned device | {assigned_device}")
            print("\n" + "=" * 16 + " Feature set " + "=" * 16)
            print(
                f"target_lag_count={target_lag_count} | "
                f"covariate_lag_count={covariate_lag_count}"
            )
            print(f"numeric_features={resolved_run_cfg.numeric_features}")
            print(
                "Split sizes | "
                f"context={len(data_splits.y_context)} train={len(data_splits.y_train)} "
                f"val={len(data_splits.y_val)} test={len(data_splits.y_test)}"
            )
            results = evaluate_feature_set(
                resolved_run_cfg,
                data_splits,
                target_lag_count=target_lag_count,
                covariate_lag_count=covariate_lag_count,
                max_context_values=max_context_values,
            )
            for result in results:
                _print_feature_set_summary(result)

    return FeatureSetExecutionArtifacts(results=results, assigned_device=assigned_device)


def _run_feature_set_jobs(
    run_cfg: FeatureSearchDataConfig,
    *,
    feature_set_jobs: list[FeatureSetJob],
    max_context_values: list[int | None],
    output_dir: Path,
) -> list[FeatureSetResult]:
    if not feature_set_jobs:
        return []

    feature_set_logs_dir = output_dir / "feature_set_logs"
    feature_set_logs_dir.mkdir(parents=True, exist_ok=True)

    if (
        run_cfg.feature_search.max_parallel_trialsper_gpu <= 1
        and (run_cfg.feature_search.gpu_ids is None or len(run_cfg.feature_search.gpu_ids) <= 1)
    ):
        results: list[FeatureSetResult] = []
        for job_index, job in enumerate(feature_set_jobs, start=1):
            feature_cfg = replace(run_cfg, numeric_features=job.numeric_features)
            data_splits = load_and_split_feature_search_data(feature_cfg)
            artifacts = _execute_feature_set_with_logging(
                feature_cfg,
                data_splits=data_splits,
                target_lag_count=job.target_lag_count,
                covariate_lag_count=job.covariate_lag_count,
                max_context_values=max_context_values,
                feature_set_log_path=feature_set_logs_dir / f"feature_set_{job_index:03d}.log",
                assigned_device=None,
                tee_stdout=True,
            )
            results.extend(artifacts.results)
        return results

    device_slots = _build_parallel_device_slots(
        max_parallel_trials_per_gpu=run_cfg.feature_search.max_parallel_trialsper_gpu,
        gpu_ids=run_cfg.feature_search.gpu_ids,
        base_device=run_cfg.model.device,
    )
    print(
        "Parallel feature-set execution enabled | "
        f"max_parallel_trialsper_gpu={run_cfg.feature_search.max_parallel_trialsper_gpu} | "
        f"gpu_ids={run_cfg.feature_search.gpu_ids} | "
        f"device_slots={device_slots}"
    )

    results: list[FeatureSetResult] = []
    mp_context = multiprocessing.get_context("spawn")
    job_iter = iter(enumerate(feature_set_jobs, start=1))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(device_slots),
        mp_context=mp_context,
    ) as executor:
        future_states: dict[
            concurrent.futures.Future[FeatureSetExecutionArtifacts],
            tuple[int, FeatureSetJob, str | None],
        ] = {}

        def submit_next(assigned_device: str | None) -> bool:
            try:
                job_index, job = next(job_iter)
            except StopIteration:
                return False

            feature_cfg = replace(run_cfg, numeric_features=job.numeric_features)
            data_splits = load_and_split_feature_search_data(feature_cfg)
            future = executor.submit(
                _execute_feature_set_with_logging,
                feature_cfg,
                data_splits=data_splits,
                target_lag_count=job.target_lag_count,
                covariate_lag_count=job.covariate_lag_count,
                max_context_values=max_context_values,
                feature_set_log_path=str(feature_set_logs_dir / f"feature_set_{job_index:03d}.log"),
                assigned_device=assigned_device,
                tee_stdout=False,
            )
            future_states[future] = (job_index, job, assigned_device)
            print(
                f"Started feature_set_{job_index:03d} | "
                f"device={assigned_device or feature_cfg.model.device or 'auto'} | "
                f"target_lag_count={job.target_lag_count} | "
                f"covariate_lag_count={job.covariate_lag_count}"
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
                job_index, job, assigned_device = future_states.pop(future)
                artifacts = future.result()
                results.extend(artifacts.results)
                print(
                    f"Completed feature_set_{job_index:03d} | "
                    f"device={artifacts.assigned_device or assigned_device or 'auto'} | "
                    f"target_lag_count={job.target_lag_count} | "
                    f"covariate_lag_count={job.covariate_lag_count}"
                )
                submit_next(assigned_device)

    return results


def _write_summary_json(
    output_dir: Path,
    *,
    config_path: Path,
    run_cfg: FeatureSearchDataConfig,
    lag_metadata: FeatureLagMetadata,
    results: list[FeatureSetResult],
) -> None:
    max_context_values = list(run_cfg.feature_search.max_context)
    payload = {
        "config": str(config_path),
        "prediction_window": run_cfg.prediction_window,
        "feature_search": {
            "max_context": max_context_values,
            "target_lag_count": list(run_cfg.feature_search.target_lag_count),
            "covariate_lag_count": list(run_cfg.feature_search.covariate_lag_count),
            "include_calendar_features": run_cfg.feature_search.include_calendar_features,
            "include_seasonal_features": run_cfg.feature_search.include_seasonal_features,
            "normalize": run_cfg.feature_search.normalize,
            "max_parallel_trialsper_gpu": run_cfg.feature_search.max_parallel_trialsper_gpu,
            "gpu_ids": run_cfg.feature_search.gpu_ids,
        },
        "lag_metadata": _serialize_lag_metadata(lag_metadata),
        "results_by_max_context": [
            {
                "max_context": max_context,
                "rankings": [
                    {
                        "rank": rank,
                        "target_lag_count": result.target_lag_count,
                        "covariate_lag_count": result.covariate_lag_count,
                        "val": _serialize_summary_metrics(result.val),
                        "test": _serialize_summary_metrics(result.test),
                    }
                    for rank, result in enumerate(
                        sorted(
                            (item for item in results if item.max_context == max_context),
                            key=_feature_set_sort_key,
                        ),
                        start=1,
                    )
                ],
                "feature_sets": [
                    {
                        "target_lag_count": result.target_lag_count,
                        "covariate_lag_count": result.covariate_lag_count,
                        "numeric_features": result.numeric_features,
                        "val": _serialize_summary_metrics(result.val),
                        "test": _serialize_summary_metrics(result.test),
                        "per_horizon": [
                            {
                                "horizon": horizon_result.horizon,
                                "val": _serialize_summary_metrics(horizon_result.val),
                                "test": _serialize_summary_metrics(horizon_result.test),
                            }
                            for horizon_result in result.per_horizon
                        ],
                    }
                    for result in sorted(
                        (item for item in results if item.max_context == max_context),
                        key=_feature_set_sort_key,
                    )
                ],
            }
            for max_context in max_context_values
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_feature_search(
    config_path: str,
    *,
    dataset: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    resolved_config_path = _resolve_config_path(config_path)
    run_cfg = load_feature_search_config(str(resolved_config_path), dataset)
    lag_metadata = validate_feature_search_config(run_cfg)

    config_stem = resolved_config_path.stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(output_root) if output_root is not None else Path("results") / "feature_search"
    output_dir = output_base / f"{config_stem}_feature_search_baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    log_path = output_dir / "summary.log"
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        tee_writer = _TeeWriter(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee_writer), contextlib.redirect_stderr(tee_writer):
            print(
                "Feature search setup | "
                f"config={resolved_config_path} | "
                f"prediction_window={run_cfg.prediction_window}"
            )
            print(
                "Feature flags | "
                f"calendar_features_enabled={run_cfg.feature_search.include_calendar_features} | "
                f"seasonal_features_enabled={run_cfg.feature_search.include_seasonal_features} | "
                f"normalization_enabled={run_cfg.feature_search.normalize}"
            )
            print(
                "Search space | "
                f"max_context={run_cfg.feature_search.max_context} | "
                f"target_lag_count={run_cfg.feature_search.target_lag_count} | "
                f"covariate_lag_count={run_cfg.feature_search.covariate_lag_count}"
            )
            print(
                "Parallelism | "
                f"max_parallel_trialsper_gpu={run_cfg.feature_search.max_parallel_trialsper_gpu} | "
                f"gpu_ids={run_cfg.feature_search.gpu_ids}"
            )
            print(
                "Lag metadata | "
                f"target_lag_prefix={lag_metadata.target_lag_prefix} | "
                f"available_target_lag_count={lag_metadata.available_target_lag_count} | "
                f"shared_covariate_lag_count={lag_metadata.shared_covariate_lag_count}"
            )
            for family in lag_metadata.covariate_families:
                print(
                    "Covariate family | "
                    f"prefix={family.prefix} | "
                    f"base_lag_count={family.base_lag_count} | "
                    f"available_lag_count={family.available_lag_count}"
                )

            feature_set_jobs: list[FeatureSetJob] = []
            for target_lag_count in run_cfg.feature_search.target_lag_count:
                for covariate_lag_count in run_cfg.feature_search.covariate_lag_count:
                    resolved_numeric_features = resolve_feature_search_numeric_features(
                        run_cfg,
                        target_lag_count=target_lag_count,
                        covariate_lag_count=covariate_lag_count,
                        lag_metadata=lag_metadata,
                    )
                    feature_set_jobs.append(
                        FeatureSetJob(
                            target_lag_count=target_lag_count,
                            covariate_lag_count=covariate_lag_count,
                            numeric_features=resolved_numeric_features,
                        )
                    )

            results = _run_feature_set_jobs(
                run_cfg,
                feature_set_jobs=feature_set_jobs,
                max_context_values=list(run_cfg.feature_search.max_context),
                output_dir=output_dir,
            )

            print("\nSummary")
            for max_context in run_cfg.feature_search.max_context:
                _print_rankings(results, max_context=max_context)

    _write_summary_json(
        output_dir,
        config_path=resolved_config_path,
        run_cfg=run_cfg,
        lag_metadata=lag_metadata,
        results=results,
    )
    return {
        "output_dir": str(output_dir),
        "summary_log": str(log_path),
        "summary_json": str(output_dir / "summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TabDPT baseline feature search without text attention.")
    parser.add_argument("--config", required=True, help="Path to the feature-search YAML config.")
    parser.add_argument("--dataset", default=None, help="Optional dataset override for multi-dataset configs.")
    parser.add_argument("--output-root", default=None, help="Optional output root. Defaults to results/feature_search.")
    args = parser.parse_args()

    result = run_feature_search(
        args.config,
        dataset=args.dataset,
        output_root=args.output_root,
    )
    print(f"Saved summary.log to: {result['summary_log']}")
    print(f"Saved summary.json to: {result['summary_json']}")


if __name__ == "__main__":
    main()
