from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from .batch_mode_utils import (
        BatchSlotMetrics,
        batch_forecast_records,
        build_terminal_available_batch_queries,
        evaluate_batch_slot_metrics,
        records_to_csv_rows,
        resolve_batch_horizon,
        predict_tabdpt_batch,
    )
    from .eval_fine_tune import _format_dual_metrics
    from .fine_tune_configs import load_fine_tune_config
    from .fine_tune_dpt import (
        _prepared_eval_inputs,
        evaluate_rolling,
        fine_tune_prepared_trial,
        inverse_transform_targets,
        load_fine_tune_arrays,
        load_and_split_fine_tune_data,
        prepare_fine_tune_trial,
        preprocess_features,
        prediction_horizons,
        set_random_seeds,
    )
except ImportError:
    from batch_mode_utils import (
        BatchSlotMetrics,
        batch_forecast_records,
        build_terminal_available_batch_queries,
        evaluate_batch_slot_metrics,
        records_to_csv_rows,
        resolve_batch_horizon,
        predict_tabdpt_batch,
    )
    from eval_fine_tune import _format_dual_metrics
    from fine_tune_configs import load_fine_tune_config
    from fine_tune_dpt import (
        _prepared_eval_inputs,
        evaluate_rolling,
        fine_tune_prepared_trial,
        inverse_transform_targets,
        load_fine_tune_arrays,
        load_and_split_fine_tune_data,
        prepare_fine_tune_trial,
        preprocess_features,
        prediction_horizons,
        set_random_seeds,
    )


def _load_batch_dataframe(run_cfg) -> pd.DataFrame:
    df = pd.read_csv(run_cfg.data_path)
    if run_cfg.date_column:
        if run_cfg.date_column not in df.columns:
            raise ValueError(f"Missing date column {run_cfg.date_column!r} in {run_cfg.data_path}.")
        df[run_cfg.date_column] = pd.to_datetime(df[run_cfg.date_column])
        df = df.sort_values(run_cfg.date_column).reset_index(drop=True)
    else:
        raise ValueError("Batch mode requires a date_column so future target calendars can be generated.")
    if run_cfg.max_rows is not None:
        df = df.head(run_cfg.max_rows).reset_index(drop=True)
    return df


def _print_slot_metrics(header: str, metrics: list[BatchSlotMetrics]) -> None:
    print(f"\n{header}")
    for metric in metrics:
        _format_dual_metrics(
            f"slot {metric.slot:02d} (n={metric.count})",
            metric.normalized,
            metric.real,
        )


def _future_support_from_prepared(prepared_trial) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_support = np.concatenate(
        (
            prepared_trial.X_context_proc,
            prepared_trial.X_train_proc,
            prepared_trial.X_val_proc,
            prepared_trial.X_test_proc,
        ),
        axis=0,
    )
    y_support = np.concatenate(
        (
            prepared_trial.splits.y_context,
            prepared_trial.splits.y_train,
            prepared_trial.splits.y_val,
            prepared_trial.splits.y_test,
        ),
        axis=0,
    )
    text_support = np.concatenate(
        (
            prepared_trial.splits.text_context,
            prepared_trial.splits.text_train,
            prepared_trial.splits.text_val,
            prepared_trial.splits.text_test,
        ),
        axis=0,
    )
    return X_support, y_support, text_support


def _call_quietly(fn, /, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def run_fixed_horizon_batch(
    run_cfg,
    *,
    horizon: int | None = None,
) -> dict[str, object]:
    batch_horizon = resolve_batch_horizon(horizon if horizon is not None else run_cfg.prediction_window)
    print(f"\n{'=' * 16} Horizon {batch_horizon}/{run_cfg.prediction_window} {'=' * 16}")

    data_splits = load_and_split_fine_tune_data(run_cfg)
    prepared_trial = prepare_fine_tune_trial(run_cfg, data_splits=data_splits, horizon=batch_horizon)
    fine_tune_outcome = fine_tune_prepared_trial(prepared_trial, tuning_cfg=run_cfg.tuning)
    reg = prepared_trial.reg

    val_inputs = _prepared_eval_inputs(prepared_trial, split_name="val")
    val_metrics = _call_quietly(
        evaluate_rolling,
        reg,
        X_context_proc=val_inputs[0],
        y_context=val_inputs[1],
        text_context=val_inputs[2],
        X_eval_proc=val_inputs[3],
        y_eval=val_inputs[4],
        y_eval_real=val_inputs[5],
        text_eval=val_inputs[6],
        use_text=True,
        label="Val (with text attn)",
        max_context=run_cfg.tuning.max_context,
        target_scaler=prepared_trial.target_scaler,
        horizon=batch_horizon,
    )
    test_inputs = _prepared_eval_inputs(prepared_trial, split_name="test")
    test_metrics = _call_quietly(
        evaluate_rolling,
        reg,
        X_context_proc=test_inputs[0],
        y_context=test_inputs[1],
        text_context=test_inputs[2],
        X_eval_proc=test_inputs[3],
        y_eval=test_inputs[4],
        y_eval_real=test_inputs[5],
        text_eval=test_inputs[6],
        use_text=True,
        label="Test (with text attn)",
        max_context=run_cfg.tuning.max_context,
        target_scaler=prepared_trial.target_scaler,
        horizon=batch_horizon,
    )

    val_slot_metrics = evaluate_batch_slot_metrics(
        batch_horizon=batch_horizon,
        X_history=val_inputs[0],
        y_history=val_inputs[1],
        X_eval=val_inputs[3],
        y_eval=val_inputs[4],
        y_eval_real=val_inputs[5],
        max_context=run_cfg.tuning.max_context,
        text_history=val_inputs[2],
        text_eval=val_inputs[6],
        predict_fn=lambda X_support, y_support, text_support, X_query, text_query: predict_tabdpt_batch(
            reg,
            X_support_proc=X_support,
            y_support=y_support,
            X_query_proc=X_query,
            max_context=run_cfg.tuning.max_context,
            use_text=True,
            text_support=text_support,
            text_query=text_query,
        ),
        inverse_transform_fn=lambda arr: inverse_transform_targets(arr, prepared_trial.target_scaler),
    )
    test_slot_metrics = evaluate_batch_slot_metrics(
        batch_horizon=batch_horizon,
        X_history=test_inputs[0],
        y_history=test_inputs[1],
        X_eval=test_inputs[3],
        y_eval=test_inputs[4],
        y_eval_real=test_inputs[5],
        max_context=run_cfg.tuning.max_context,
        text_history=test_inputs[2],
        text_eval=test_inputs[6],
        predict_fn=lambda X_support, y_support, text_support, X_query, text_query: predict_tabdpt_batch(
            reg,
            X_support_proc=X_support,
            y_support=y_support,
            X_query_proc=X_query,
            max_context=run_cfg.tuning.max_context,
            use_text=True,
            text_support=text_support,
            text_query=text_query,
        ),
        inverse_transform_fn=lambda arr: inverse_transform_targets(arr, prepared_trial.target_scaler),
    )
    _print_slot_metrics("Validation Slot Metrics", val_slot_metrics)
    _print_slot_metrics("Test Slot Metrics", test_slot_metrics)
    print("\n== Overall Metrics ==")
    _format_dual_metrics("Val (with text attn)", val_metrics[0], val_metrics[1])
    _format_dual_metrics("Test (with text attn)", test_metrics[0], test_metrics[1])

    X_raw, y_raw, text_raw, timestamps, _ = load_fine_tune_arrays(run_cfg)
    if timestamps is None:
        raise ValueError("Batch mode requires timestamps to generate future target calendars.")
    query_bundle = build_terminal_available_batch_queries(
        X_raw=X_raw,
        y_raw=y_raw,
        text=text_raw,
        timestamps=pd.Series(pd.to_datetime(timestamps)),
        calendar_frequency=run_cfg.calendar_frequency,
        seasonality_k=run_cfg.seasonality_k,
        seasonality_L=run_cfg.seasonality_L,
        horizon=batch_horizon,
    )
    X_query_proc = preprocess_features(
        reg,
        query_bundle.X_query_raw,
        reduction_mode=None,
        reduction_payload=None,
    )
    X_support_proc, y_support, text_support = _future_support_from_prepared(prepared_trial)
    future_preds_normalized = predict_tabdpt_batch(
        reg,
        X_support_proc=X_support_proc,
        y_support=y_support,
        X_query_proc=X_query_proc,
        max_context=run_cfg.tuning.max_context,
        use_text=True,
        text_support=text_support,
        text_query=query_bundle.text_query,
    )
    future_preds_real = inverse_transform_targets(future_preds_normalized, prepared_trial.target_scaler)
    forecast_records = batch_forecast_records(
        predictions_real=future_preds_real,
        query_bundle=query_bundle,
        model_name="tuned_text",
    )

    print("\n== Final Available Batch Forecast ==")
    for record in forecast_records:
        print(
            f"slot {record.slot:02d} | source_index={record.source_index} "
            f"source_ts={record.source_timestamp} -> target_ts={record.target_timestamp} "
            f"| prediction={record.prediction_real:.4f}"
        )

    csv_rows: list[dict[str, object]] = []
    csv_rows.extend(
        records_to_csv_rows(
            forecast_records=[],
            slot_metrics=val_slot_metrics,
            split_name="val",
            overall_normalized=val_metrics[0],
            overall_real=val_metrics[1],
        )
    )
    csv_rows.extend(
        records_to_csv_rows(
            forecast_records=[],
            slot_metrics=test_slot_metrics,
            split_name="test",
            overall_normalized=test_metrics[0],
            overall_real=test_metrics[1],
        )
    )
    csv_rows.extend(
        records_to_csv_rows(
            forecast_records=forecast_records,
            slot_metrics=[],
            split_name="final_available",
        )
    )
    for row in csv_rows:
        row["horizon"] = batch_horizon

    return {
        "batch_horizon": batch_horizon,
        "best_epoch": fine_tune_outcome.best_epoch,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_slot_metrics": val_slot_metrics,
        "test_slot_metrics": test_slot_metrics,
        "forecast_records": forecast_records,
        "csv_rows": csv_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-horizon batch-mode fine-tuning with one model per batch horizon."
    )
    parser.add_argument("--dataset", help="Dataset key in the config file.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    parser.add_argument("--output-csv", help="Optional CSV path for overall metrics, slot metrics, and forecasts.")
    args = parser.parse_args()

    run_cfg = load_fine_tune_config(args.config, args.dataset)
    set_random_seeds(run_cfg.seed)

    if args.dataset:
        print(f"Using dataset '{args.dataset}' from {args.config}")
    else:
        print(f"Using dataset config {args.config}")

    horizon_results = [
        run_fixed_horizon_batch(run_cfg, horizon=horizon)
        for horizon in prediction_horizons(run_cfg)
    ]

    if len(horizon_results) > 1:
        print("\n================ Mean Summary Across Horizons ================")
        mean_val_normalized = tuple(
            float(np.mean([result["val_metrics"][0][idx] for result in horizon_results])) for idx in range(3)
        )
        mean_val_real = tuple(
            float(np.mean([result["val_metrics"][1][idx] for result in horizon_results])) for idx in range(3)
        )
        mean_test_normalized = tuple(
            float(np.mean([result["test_metrics"][0][idx] for result in horizon_results])) for idx in range(3)
        )
        mean_test_real = tuple(
            float(np.mean([result["test_metrics"][1][idx] for result in horizon_results])) for idx in range(3)
        )
        _format_dual_metrics("Mean val (with text attn)", mean_val_normalized, mean_val_real)
        _format_dual_metrics("Mean test (with text attn)", mean_test_normalized, mean_test_real)
        print(
            "Best epochs by horizon: "
            + ", ".join(f"h{result['batch_horizon']}={result['best_epoch']}" for result in horizon_results)
        )

    if args.output_csv:
        output_path = Path(args.output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        csv_rows = [row for result in horizon_results for row in result["csv_rows"]]
        pd.DataFrame(csv_rows).to_csv(output_path, index=False)
        print(f"\noutput_csv={output_path}")


if __name__ == "__main__":
    main()
