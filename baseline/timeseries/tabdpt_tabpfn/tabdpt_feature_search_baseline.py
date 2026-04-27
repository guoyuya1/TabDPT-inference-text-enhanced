"""Feature search runner for TabDPT baseline without text attention."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fine_tuning.eval_fine_tune import _format_dual_metrics, evaluate_rolling  # noqa: E402
from fine_tuning.feature_search_config import FeatureSearchDataConfig, load_feature_search_config  # noqa: E402

try:  # pragma: no cover - import path depends on execution mode
    from .tabdpt_hpo_baseline import (  # type: ignore[attr-defined]  # noqa: E402
        FineTuneDataSplits,
        HorizonResult,
        SummaryMetrics,
        _TeeWriter,
        _build_history_and_eval_split,
        _level_eval_context,
        _make_tabdpt,
        _mean_summary_metrics,
        _reset_compiler_state,
        _select_horizon_splits,
        _serialize_summary_metrics,
        _summary_metrics_from_dual,
        load_and_split_fine_tune_data,
        maybe_normalize_fine_tune_splits,
        prediction_horizons,
        preprocess_features,
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
        _mean_summary_metrics,
        _reset_compiler_state,
        _select_horizon_splits,
        _serialize_summary_metrics,
        _summary_metrics_from_dual,
        load_and_split_fine_tune_data,
        maybe_normalize_fine_tune_splits,
        prediction_horizons,
        preprocess_features,
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


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (SCRIPT_DIR / path).resolve()


def _read_csv_header(path: str) -> list[str]:
    with Path(path).expanduser().open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if header is None:
        raise ValueError(f"CSV file has no header row: {path}")
    return header


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
        raise ValueError(
            "feature_search.covariate_lag_count requires at least one non-target lagged numeric feature family."
        )

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
    target_inserted = False
    inserted_covariates: set[str] = set()

    for feature_name in run_cfg.numeric_features:
        family_name = _feature_family_name(feature_name)
        if family_name == lag_metadata.target_lag_prefix:
            if not target_inserted:
                resolved_numeric_features.extend(
                    f"{lag_metadata.target_lag_prefix}_lag{lag}" for lag in range(1, target_lag_count + 1)
                )
                target_inserted = True
            continue

        if family_name in covariate_prefixes:
            if family_name not in inserted_covariates:
                resolved_numeric_features.extend(
                    f"{family_name}_lag{lag}" for lag in range(1, covariate_lag_count + 1)
                )
                inserted_covariates.add(family_name)
            continue

        resolved_numeric_features.append(feature_name)

    if not target_inserted:
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
    output_base = Path(output_root) if output_root is not None else Path("results") / "hpo"
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

            results: list[FeatureSetResult] = []
            for target_lag_count in run_cfg.feature_search.target_lag_count:
                for covariate_lag_count in run_cfg.feature_search.covariate_lag_count:
                    resolved_numeric_features = resolve_feature_search_numeric_features(
                        run_cfg,
                        target_lag_count=target_lag_count,
                        covariate_lag_count=covariate_lag_count,
                        lag_metadata=lag_metadata,
                    )
                    feature_cfg = replace(run_cfg, numeric_features=resolved_numeric_features)
                    data_splits = load_and_split_fine_tune_data(feature_cfg)
                    print(
                        "\n"
                        + "=" * 16
                        + " Feature set "
                        + "=" * 16
                    )
                    print(
                        f"target_lag_count={target_lag_count} | "
                        f"covariate_lag_count={covariate_lag_count}"
                    )
                    print(f"numeric_features={resolved_numeric_features}")
                    print(
                        "Split sizes | "
                        f"context={len(data_splits.y_context)} train={len(data_splits.y_train)} "
                        f"val={len(data_splits.y_val)} test={len(data_splits.y_test)}"
                    )
                    feature_results = evaluate_feature_set(
                        feature_cfg,
                        data_splits,
                        target_lag_count=target_lag_count,
                        covariate_lag_count=covariate_lag_count,
                        max_context_values=list(run_cfg.feature_search.max_context),
                    )
                    for result in feature_results:
                        _print_feature_set_summary(result)
                    results.extend(feature_results)

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
    parser.add_argument("--output-root", default=None, help="Optional output root. Defaults to results/hpo.")
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
