"""Chronos2 baseline with config-driven dataset selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def read_cfg(config_path: str, dataset: str | None) -> dict:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parent / config_file

    with open(config_file, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if isinstance(raw_cfg, dict) and any(key in raw_cfg for key in ("data_path", "target_column", "target")):
        dataset_name = dataset or config_file.stem
        return {dataset_name: dict(raw_cfg)}

    dataset_map = raw_cfg.get("datasets") if isinstance(raw_cfg, dict) else None
    if dataset_map is None:
        dataset_map = raw_cfg
    if dataset is None:
        return {name: dict(cfg) for name, cfg in dataset_map.items()}
    return {dataset: dict(dataset_map[dataset])}


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="chronos2.yaml")
parser.add_argument("--dataset")
args = parser.parse_args()

all_cfgs = read_cfg(args.config, args.dataset)
summary_rows = []

for dataset_name, cfg in all_cfgs.items():
    print(dataset_name)

    feature_columns = []
    for key in ("numeric_features", "features", "feature_columns"):
        value = cfg.get(key)
        if value is not None:
            feature_columns = list(value)
            break

    date_column = cfg["date_column"]
    target_column = str(cfg.get("target_column") or cfg.get("target"))
    selected_columns = [date_column, *feature_columns, target_column]
    selected_columns = list(dict.fromkeys(selected_columns))

    df = pd.read_csv(cfg["data_path"])
    df = df.sort_values(date_column).reset_index(drop=True)[selected_columns]

    if cfg.get("max_rows") is not None:
        df = df.iloc[: int(cfg["max_rows"])].copy()
    else:
        df = df.copy()

    df["item_id"] = 1

    n_rows = len(df)
    train_end = int(n_rows * float(cfg.get("train_ratio") or 0.0))
    val_end = train_end + int(n_rows * float(cfg.get("val_ratio") or 0.0))
    if not val_end and cfg.get("fit_rows") is not None:
        val_end = int(cfg["fit_rows"])
        train_end = val_end

    zero_shot_predictor = TimeSeriesPredictor(
        target=target_column,
        prediction_length=1,
    )
    zero_shot_predictor.fit(
        TimeSeriesDataFrame.from_data_frame(
            df.iloc[:val_end], id_column="item_id", timestamp_column=date_column
        ),
        hyperparameters={
            "Chronos2": {"model_path": "autogluon/chronos-2"}
        },
        enable_ensemble=False,
    )

    zero_shot_preds = []
    for i in range(val_end, len(df)):
        hist_df = df.iloc[:i].copy()
        hist_ts = TimeSeriesDataFrame.from_data_frame(
            hist_df, id_column="item_id", timestamp_column=date_column
        )
        fcst = zero_shot_predictor.predict(hist_ts, model="Chronos2")
        y_pred = fcst["mean"].iloc[-1]
        y_true = df.loc[i, target_column]
        zero_shot_preds.append({"index": i, "date": df.loc[i, date_column], "y_true": y_true, "y_pred": y_pred})

    zero_shot_preds_df = pd.DataFrame(zero_shot_preds)
    zero_shot_err = zero_shot_preds_df["y_true"] - zero_shot_preds_df["y_pred"]
    zero_shot_mae = zero_shot_err.abs().mean()
    zero_shot_rmse = np.sqrt(np.mean(zero_shot_err ** 2))
    zero_shot_mape = np.mean(np.abs(zero_shot_err) / (np.abs(zero_shot_preds_df["y_true"]) + 1e-8)) * 100

    print("baseline_1_zero_shot")
    print("MAE:", zero_shot_mae)
    print("RMSE:", zero_shot_rmse)
    print("MAPE (%):", zero_shot_mape)

    fine_tuned_predictor = TimeSeriesPredictor(
        target=target_column,
        prediction_length=1,
    )
    fine_tuned_predictor.fit(
        TimeSeriesDataFrame.from_data_frame(
            df.iloc[:train_end], id_column="item_id", timestamp_column=date_column
        ),
        tuning_data=TimeSeriesDataFrame.from_data_frame(
            df.iloc[train_end:val_end], id_column="item_id", timestamp_column=date_column
        ),
        hyperparameters={
            "Chronos2": {
                "model_path": "autogluon/chronos-2",
                "fine_tune": True,
                "eval_during_fine_tune": True,
            }
        },
        enable_ensemble=False,
    )

    fine_tuned_preds = []
    for i in range(val_end, len(df)):
        hist_df = df.iloc[:i].copy()
        hist_ts = TimeSeriesDataFrame.from_data_frame(
            hist_df, id_column="item_id", timestamp_column=date_column
        )
        fcst = fine_tuned_predictor.predict(hist_ts, model="Chronos2")
        y_pred = fcst["mean"].iloc[-1]
        y_true = df.loc[i, target_column]
        fine_tuned_preds.append({"index": i, "date": df.loc[i, date_column], "y_true": y_true, "y_pred": y_pred})

    fine_tuned_preds_df = pd.DataFrame(fine_tuned_preds)
    fine_tuned_err = fine_tuned_preds_df["y_true"] - fine_tuned_preds_df["y_pred"]
    fine_tuned_mae = fine_tuned_err.abs().mean()
    fine_tuned_rmse = np.sqrt(np.mean(fine_tuned_err ** 2))
    fine_tuned_mape = np.mean(np.abs(fine_tuned_err) / (np.abs(fine_tuned_preds_df["y_true"]) + 1e-8)) * 100

    print("baseline_2_fine_tuned")
    print("MAE:", fine_tuned_mae)
    print("RMSE:", fine_tuned_rmse)
    print("MAPE (%):", fine_tuned_mape)

    summary_rows.append(
        {
            "dataset": dataset_name,
            "baseline_1_zero_shot_mae": zero_shot_mae,
            "baseline_1_zero_shot_rmse": zero_shot_rmse,
            "baseline_2_fine_tuned_mae": fine_tuned_mae,
            "baseline_2_fine_tuned_rmse": fine_tuned_rmse,
        }
    )

print("summary")
for row in summary_rows:
    print(
        row["dataset"],
        "| baseline_1_zero_shot MAE:",
        row["baseline_1_zero_shot_mae"],
        "RMSE:",
        row["baseline_1_zero_shot_rmse"],
        "| baseline_2_fine_tuned MAE:",
        row["baseline_2_fine_tuned_mae"],
        "RMSE:",
        row["baseline_2_fine_tuned_rmse"],
    )
