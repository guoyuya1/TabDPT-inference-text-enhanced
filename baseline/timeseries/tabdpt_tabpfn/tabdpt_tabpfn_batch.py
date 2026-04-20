"""Fixed-horizon batch-mode TabDPT and TabPFN baselines."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
for path in (SCRIPT_DIR, REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fine_tuning.batch_mode_utils import (
    BatchSlotMetrics,
    batch_forecast_records,
    build_terminal_available_batch_queries,
    evaluate_batch_slot_metrics,
    predict_estimator_batch,
    predict_tabdpt_batch,
    records_to_csv_rows,
    resolve_batch_horizon,
)
from tabdpt_tabpfn import (
    _format_dual_metrics,
    _format_metrics,
    build_history_and_eval_split,
    build_horizon_splits,
    evaluate_rolling_estimator,
    evaluate_rolling_tabdpt,
    inverse_transform_targets,
    load_dataframe,
    load_numeric_arrays,
    make_tabdpt,
    make_tabpfn,
    normalize_target_splits,
    parse_cfg,
    preprocess_features,
    read_cfg,
)


def _model_rows(rows: list[dict[str, object]], *, model_name: str) -> list[dict[str, object]]:
    for row in rows:
        row["model_name"] = model_name
    return rows


def _future_support(X_context_proc, X_train_proc, X_val_proc, X_test_proc, splits) -> tuple[np.ndarray, np.ndarray]:
    X_support = np.concatenate((X_context_proc, X_train_proc, X_val_proc, X_test_proc), axis=0)
    y_support = np.concatenate((splits.y_context, splits.y_train, splits.y_val, splits.y_test), axis=0)
    return X_support, y_support


def _call_quietly(fn, /, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def run_fixed_horizon_batch(cfg: dict, *, use_tabpfn: bool, output_csv: str | None = None) -> dict[str, object]:
    batch_horizon = resolve_batch_horizon(int(cfg["prediction_window"]))

    df = load_dataframe(cfg)
    timestamps = df[cfg["date_column"]].reset_index(drop=True) if cfg.get("date_column") else None
    if timestamps is None:
        raise ValueError("Batch mode requires cfg['date_column'] so future target calendars can be generated.")

    X_raw, y_raw = load_numeric_arrays(cfg)
    raw_feature_splits = build_horizon_splits(X_raw, y_raw, cfg, horizon=batch_horizon, timestamps=timestamps)
    splits, target_scaler = normalize_target_splits(raw_feature_splits, cfg["normalizer"])

    tabdpt = make_tabdpt(cfg)
    tabdpt.fit(splits.X_context, splits.y_context)

    X_context_proc = preprocess_features(tabdpt, splits.X_context)
    X_train_proc = preprocess_features(tabdpt, splits.X_train)
    X_val_proc = preprocess_features(tabdpt, splits.X_val)
    X_test_proc = preprocess_features(tabdpt, splits.X_test)

    val_hist = build_history_and_eval_split(
        X_context_proc,
        X_train_proc,
        X_val_proc,
        X_test_proc,
        splits.y_context,
        splits.y_train,
        splits.y_val,
        splits.y_test,
        "val",
        batch_horizon,
    )
    val_hist_real = build_history_and_eval_split(
        raw_feature_splits.X_context,
        raw_feature_splits.X_train,
        raw_feature_splits.X_val,
        raw_feature_splits.X_test,
        raw_feature_splits.y_context,
        raw_feature_splits.y_train,
        raw_feature_splits.y_val,
        raw_feature_splits.y_test,
        "val",
        batch_horizon,
    )
    test_hist = build_history_and_eval_split(
        X_context_proc,
        X_train_proc,
        X_val_proc,
        X_test_proc,
        splits.y_context,
        splits.y_train,
        splits.y_val,
        splits.y_test,
        "test",
        batch_horizon,
    )
    test_hist_real = build_history_and_eval_split(
        raw_feature_splits.X_context,
        raw_feature_splits.X_train,
        raw_feature_splits.X_val,
        raw_feature_splits.X_test,
        raw_feature_splits.y_context,
        raw_feature_splits.y_train,
        raw_feature_splits.y_val,
        raw_feature_splits.y_test,
        "test",
        batch_horizon,
    )

    tabdpt_val_metrics = _call_quietly(
        evaluate_rolling_tabdpt,
        tabdpt,
        X_context_proc=val_hist[0],
        y_context=val_hist[1],
        X_eval_proc=val_hist[2],
        y_eval=val_hist[3],
        y_eval_real=val_hist_real[3],
        max_context=cfg["max_context"],
        horizon=batch_horizon,
        target_scaler=target_scaler,
        label="Val TabDPT",
    )
    tabdpt_test_metrics = _call_quietly(
        evaluate_rolling_tabdpt,
        tabdpt,
        X_context_proc=test_hist[0],
        y_context=test_hist[1],
        X_eval_proc=test_hist[2],
        y_eval=test_hist[3],
        y_eval_real=test_hist_real[3],
        max_context=cfg["max_context"],
        horizon=batch_horizon,
        target_scaler=target_scaler,
        label="Test TabDPT",
    )
    tabpfn_val_metrics = None
    tabpfn_test_metrics = None
    if use_tabpfn:
        tabpfn_val_metrics = _call_quietly(
            evaluate_rolling_estimator,
            make_tabpfn(cfg),
            X_context=val_hist[0],
            y_context=val_hist[1],
            X_eval=val_hist[2],
            y_eval=val_hist[3],
            y_eval_real=val_hist_real[3],
            max_context=cfg["max_context"],
            horizon=batch_horizon,
            target_scaler=target_scaler,
            label="Val TabPFN",
        )
        tabpfn_test_metrics = _call_quietly(
            evaluate_rolling_estimator,
            make_tabpfn(cfg),
            X_context=test_hist[0],
            y_context=test_hist[1],
            X_eval=test_hist[2],
            y_eval=test_hist[3],
            y_eval_real=test_hist_real[3],
            max_context=cfg["max_context"],
            horizon=batch_horizon,
            target_scaler=target_scaler,
            label="Test TabPFN",
        )

    tabdpt_val_slot_metrics = evaluate_batch_slot_metrics(
        batch_horizon=batch_horizon,
        X_history=val_hist[0],
        y_history=val_hist[1],
        X_eval=val_hist[2],
        y_eval=val_hist[3],
        y_eval_real=val_hist_real[3],
        max_context=cfg["max_context"],
        predict_fn=lambda X_support, y_support, _text_support, X_query, _text_query: predict_tabdpt_batch(
            tabdpt,
            X_support_proc=X_support,
            y_support=y_support,
            X_query_proc=X_query,
            max_context=cfg["max_context"],
            use_text=False,
        ),
        inverse_transform_fn=lambda arr: inverse_transform_targets(arr, target_scaler),
    )
    tabdpt_test_slot_metrics = evaluate_batch_slot_metrics(
        batch_horizon=batch_horizon,
        X_history=test_hist[0],
        y_history=test_hist[1],
        X_eval=test_hist[2],
        y_eval=test_hist[3],
        y_eval_real=test_hist_real[3],
        max_context=cfg["max_context"],
        predict_fn=lambda X_support, y_support, _text_support, X_query, _text_query: predict_tabdpt_batch(
            tabdpt,
            X_support_proc=X_support,
            y_support=y_support,
            X_query_proc=X_query,
            max_context=cfg["max_context"],
            use_text=False,
        ),
        inverse_transform_fn=lambda arr: inverse_transform_targets(arr, target_scaler),
    )
    tabpfn_val_slot_metrics = None
    tabpfn_test_slot_metrics = None
    if use_tabpfn:
        tabpfn_val_slot_metrics = evaluate_batch_slot_metrics(
            batch_horizon=batch_horizon,
            X_history=val_hist[0],
            y_history=val_hist[1],
            X_eval=val_hist[2],
            y_eval=val_hist[3],
            y_eval_real=val_hist_real[3],
            max_context=cfg["max_context"],
            predict_fn=lambda X_support, y_support, _text_support, X_query, _text_query: predict_estimator_batch(
                make_tabpfn(cfg),
                X_support=X_support,
                y_support=y_support,
                X_query=X_query,
                max_context=cfg["max_context"],
            ),
            inverse_transform_fn=lambda arr: inverse_transform_targets(arr, target_scaler),
        )
        tabpfn_test_slot_metrics = evaluate_batch_slot_metrics(
            batch_horizon=batch_horizon,
            X_history=test_hist[0],
            y_history=test_hist[1],
            X_eval=test_hist[2],
            y_eval=test_hist[3],
            y_eval_real=test_hist_real[3],
            max_context=cfg["max_context"],
            predict_fn=lambda X_support, y_support, _text_support, X_query, _text_query: predict_estimator_batch(
                make_tabpfn(cfg),
                X_support=X_support,
                y_support=y_support,
                X_query=X_query,
                max_context=cfg["max_context"],
            ),
            inverse_transform_fn=lambda arr: inverse_transform_targets(arr, target_scaler),
        )
    print(f"\n{'=' * 16} Horizon {batch_horizon}/{cfg['prediction_window']} {'=' * 16}")
    _format_dual_metrics("Val TabDPT", tabdpt_val_metrics[0], tabdpt_val_metrics[1])
    _format_dual_metrics("Test TabDPT", tabdpt_test_metrics[0], tabdpt_test_metrics[1])
    if use_tabpfn and tabpfn_val_metrics is not None and tabpfn_test_metrics is not None:
        _format_dual_metrics("Val TabPFN", tabpfn_val_metrics[0], tabpfn_val_metrics[1])
        _format_dual_metrics("Test TabPFN", tabpfn_test_metrics[0], tabpfn_test_metrics[1])

    query_bundle = build_terminal_available_batch_queries(
        X_raw=X_raw,
        y_raw=y_raw,
        text=None,
        timestamps=pd.Series(pd.to_datetime(timestamps)),
        calendar_frequency=cfg.get("calendar_frequency"),
        seasonality_k=int(cfg.get("seasonality_k", 3)),
        seasonality_L=cfg.get("seasonality_L"),
        horizon=batch_horizon,
    )
    X_query_proc = preprocess_features(tabdpt, query_bundle.X_query_raw)
    X_support_proc, y_support = _future_support(X_context_proc, X_train_proc, X_val_proc, X_test_proc, splits)

    tabdpt_future_normalized = predict_tabdpt_batch(
        tabdpt,
        X_support_proc=X_support_proc,
        y_support=y_support,
        X_query_proc=X_query_proc,
        max_context=cfg["max_context"],
        use_text=False,
    )
    tabdpt_future_real = inverse_transform_targets(tabdpt_future_normalized, target_scaler)
    tabdpt_forecasts = batch_forecast_records(
        predictions_real=tabdpt_future_real,
        query_bundle=query_bundle,
        model_name="tabdpt",
    )

    tabpfn_forecasts = None
    if use_tabpfn:
        tabpfn_future_normalized = predict_estimator_batch(
            make_tabpfn(cfg),
            X_support=X_support_proc,
            y_support=y_support,
            X_query=X_query_proc,
            max_context=cfg["max_context"],
        )
        tabpfn_future_real = inverse_transform_targets(tabpfn_future_normalized, target_scaler)
        tabpfn_forecasts = batch_forecast_records(
            predictions_real=tabpfn_future_real,
            query_bundle=query_bundle,
            model_name="tabpfn",
        )

    if output_csv is not None:
        rows: list[dict[str, object]] = []
        rows.extend(
            _model_rows(
                records_to_csv_rows(
                    forecast_records=[],
                    slot_metrics=tabdpt_val_slot_metrics,
                    split_name="val",
                    overall_normalized=tabdpt_val_metrics[0],
                    overall_real=tabdpt_val_metrics[1],
                ),
                model_name="tabdpt",
            )
        )
        rows.extend(
            _model_rows(
                records_to_csv_rows(
                    forecast_records=[],
                    slot_metrics=tabdpt_test_slot_metrics,
                    split_name="test",
                    overall_normalized=tabdpt_test_metrics[0],
                    overall_real=tabdpt_test_metrics[1],
                ),
                model_name="tabdpt",
            )
        )
        rows.extend(
            records_to_csv_rows(
                forecast_records=tabdpt_forecasts,
                slot_metrics=[],
                split_name="final_available",
            )
        )
        if use_tabpfn and tabpfn_val_metrics is not None and tabpfn_test_metrics is not None:
            rows.extend(
                _model_rows(
                    records_to_csv_rows(
                        forecast_records=[],
                        slot_metrics=tabpfn_val_slot_metrics or [],
                        split_name="val",
                        overall_normalized=tabpfn_val_metrics[0],
                        overall_real=tabpfn_val_metrics[1],
                    ),
                    model_name="tabpfn",
                )
            )
            rows.extend(
                _model_rows(
                    records_to_csv_rows(
                        forecast_records=[],
                        slot_metrics=tabpfn_test_slot_metrics or [],
                        split_name="test",
                        overall_normalized=tabpfn_test_metrics[0],
                        overall_real=tabpfn_test_metrics[1],
                    ),
                    model_name="tabpfn",
                )
            )
            rows.extend(
                records_to_csv_rows(
                    forecast_records=tabpfn_forecasts or [],
                    slot_metrics=[],
                    split_name="final_available",
                )
            )
        output_path = Path(output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\noutput_csv={output_path}")

    summary: dict[str, tuple[float, float, float]] = {
        "val_tabdpt": tabdpt_val_metrics[1],
        "test_tabdpt": tabdpt_test_metrics[1],
    }
    if use_tabpfn and tabpfn_val_metrics is not None and tabpfn_test_metrics is not None:
        summary["val_tabpfn"] = tabpfn_val_metrics[1]
        summary["test_tabpfn"] = tabpfn_test_metrics[1]

    print("\nSummary")
    for name in sorted(summary):
        _format_metrics(name, *summary[name])

    return {
        "batch_horizon": batch_horizon,
        "summary": summary,
        "tabdpt_val_metrics": tabdpt_val_metrics,
        "tabdpt_test_metrics": tabdpt_test_metrics,
        "tabdpt_val_slot_metrics": tabdpt_val_slot_metrics,
        "tabdpt_test_slot_metrics": tabdpt_test_slot_metrics,
        "tabdpt_forecasts": tabdpt_forecasts,
        "tabpfn_val_metrics": tabpfn_val_metrics,
        "tabpfn_test_metrics": tabpfn_test_metrics,
        "tabpfn_val_slot_metrics": tabpfn_val_slot_metrics,
        "tabpfn_test_slot_metrics": tabpfn_test_slot_metrics,
        "tabpfn_forecasts": tabpfn_forecasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-horizon batch-mode TabDPT and TabPFN baselines."
    )
    parser.add_argument("--config", default="tabular.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--tabpfn-model-path")
    parser.add_argument("--tabpfn-cache-dir")
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--output-csv", help="Optional CSV path for overall metrics, slot metrics, and forecasts.")
    args = parser.parse_args()

    config_file, raw_cfgs = read_cfg(args.config, args.dataset)
    summaries: list[tuple[str, dict[str, tuple[float, float, float]]]] = []
    for dataset_name, raw_cfg in raw_cfgs.items():
        cfg = parse_cfg(raw_cfg, args)
        np.random.seed(cfg["seed"])
        print(dataset_name)
        print(f"config={config_file}")
        print(f"prediction_window={cfg['prediction_window']}")
        print(f"max_context={cfg['max_context']}")
        if not args.skip_tabpfn:
            if cfg["tabpfn"]["model_path"]:
                print(f"tabpfn_model_path={cfg['tabpfn']['model_path']}")
            else:
                print(f"tabpfn_cache_dir={cfg['tabpfn']['cache_dir']}")
        result = run_fixed_horizon_batch(cfg, use_tabpfn=not args.skip_tabpfn, output_csv=args.output_csv)
        summaries.append((dataset_name, result["summary"]))
        print()

    print("summary")
    for dataset_name, summary in summaries:
        print(dataset_name)
        for name in sorted(summary):
            _format_metrics(name, *summary[name])


if __name__ == "__main__":
    main()
