"""Chronos2 baseline with config-driven dataset selection."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def build_normalizer(normalizer_name: str | None):
    match normalizer_name:
        case None:
            return None
        case "standard":
            return StandardScaler()
        case "minmax":
            return MinMaxScaler()
        case "robust":
            return RobustScaler()
        case _:
            raise ValueError(f"Unsupported normalizer: {normalizer_name}")


def normalize_by_train_split(
    df: pd.DataFrame,
    train_end: int,
    columns: list[str],
    normalizer_name: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if normalizer_name is None or not columns:
        return df.copy(), {}
    if train_end <= 0:
        raise ValueError("train_end must be positive when normalization is enabled")

    normalized_df = df.copy()
    scalers: dict[str, object] = {}

    for column in columns:
        full_series = pd.to_numeric(df[column], errors="coerce").astype(np.float64)
        normalized_df[column] = full_series
        train_series = full_series.iloc[:train_end]
        valid_train_mask = train_series.notna()
        if not valid_train_mask.any():
            print(
                f"Skipping normalization for {column}: "
                "no numeric training values after coercion."
            )
            normalized_df.loc[:, column] = full_series
            continue

        coerced_count = int(df[column].notna().sum() - full_series.notna().sum())
        if coerced_count:
            print(
                f"Column {column}: coerced {coerced_count} non-numeric value(s) "
                "to NaN before normalization."
            )

        scaler = build_normalizer(normalizer_name)
        train_values = train_series.loc[valid_train_mask].to_numpy(dtype=np.float64).reshape(-1, 1)
        scaler.fit(train_values)
        normalized_values = full_series.copy()
        valid_full_mask = full_series.notna()
        normalized_values.loc[valid_full_mask] = scaler.transform(
            full_series.loc[valid_full_mask].to_numpy(dtype=np.float64).reshape(-1, 1)
        ).reshape(-1)
        normalized_df.loc[:, column] = normalized_values
        scalers[column] = scaler

    return normalized_df, scalers


def inverse_transform_target(value: float, target_scaler: object | None) -> float:
    if target_scaler is None:
        return float(value)
    return float(target_scaler.inverse_transform(np.array([[value]], dtype=np.float64))[0, 0])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mae = np.abs(err).mean()
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(np.abs(err) / (np.abs(y_true) + 1e-8)) * 100
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape": float(mape),
    }


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
    frequency = cfg.get("frequency")
    target_column = str(cfg.get("target_column") or cfg.get("target"))
    normalizer_name = cfg.get("normalizer")
    selected_columns = [date_column, *feature_columns, target_column]
    selected_columns = list(dict.fromkeys(selected_columns))

    raw_df = pd.read_csv(cfg["data_path"])
    raw_df = raw_df.sort_values(date_column).reset_index(drop=True)[selected_columns]

    if cfg.get("max_rows") is not None:
        raw_df = raw_df.iloc[: int(cfg["max_rows"])].copy()
    else:
        raw_df = raw_df.copy()

    raw_df["item_id"] = 1

    n_rows = len(raw_df)
    train_end = int(n_rows * float(cfg.get("train_ratio") or 0.0))
    val_end = train_end + int(n_rows * float(cfg.get("val_ratio") or 0.0))
    prediction_window = int(cfg.get("prediction_window") or 6)
    if not val_end and cfg.get("fit_rows") is not None:
        val_end = int(cfg["fit_rows"])
        train_end = val_end

    normalized_columns = list(dict.fromkeys([*feature_columns, target_column]))
    df, scalers = normalize_by_train_split(raw_df, train_end, normalized_columns, normalizer_name)
    target_scaler = scalers.get(target_column)

    zero_shot_predictor = TimeSeriesPredictor(
        target=target_column,
        prediction_length=prediction_window,
        freq=frequency,
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
        available_horizon = min(prediction_window, len(df) - i)
        for horizon_offset in range(available_horizon):
            target_idx = i + horizon_offset
            y_pred_norm = float(fcst["mean"].iloc[horizon_offset])
            y_pred = inverse_transform_target(y_pred_norm, target_scaler)
            y_true = raw_df.loc[target_idx, target_column]
            y_true_norm = float(df.loc[target_idx, target_column])
            zero_shot_preds.append(
                {
                    "index": target_idx,
                    "date": raw_df.loc[target_idx, date_column],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_true_norm": y_true_norm,
                    "y_pred_norm": y_pred_norm,
                }
            )

    zero_shot_preds_df = pd.DataFrame(zero_shot_preds)
    zero_shot_metrics = compute_metrics(
        zero_shot_preds_df["y_true"].to_numpy(dtype=np.float64),
        zero_shot_preds_df["y_pred"].to_numpy(dtype=np.float64),
    )
    zero_shot_norm_metrics = compute_metrics(
        zero_shot_preds_df["y_true_norm"].to_numpy(dtype=np.float64),
        zero_shot_preds_df["y_pred_norm"].to_numpy(dtype=np.float64),
    )

    print("baseline_1_zero_shot")
    print("MAE:", zero_shot_metrics["mae"])
    print("RMSE:", zero_shot_metrics["rmse"])
    print("MAPE (%):", zero_shot_metrics["mape"])
    print("Normalized MAE:", zero_shot_norm_metrics["mae"])
    print("Normalized RMSE:", zero_shot_norm_metrics["rmse"])
    print("Normalized MAPE (%):", zero_shot_norm_metrics["mape"])

    fine_tuned_predictor = TimeSeriesPredictor(
        target=target_column,
        prediction_length=prediction_window,
        freq=frequency,
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
        available_horizon = min(prediction_window, len(df) - i)
        for horizon_offset in range(available_horizon):
            target_idx = i + horizon_offset
            y_pred_norm = float(fcst["mean"].iloc[horizon_offset])
            y_pred = inverse_transform_target(y_pred_norm, target_scaler)
            y_true = raw_df.loc[target_idx, target_column]
            y_true_norm = float(df.loc[target_idx, target_column])
            fine_tuned_preds.append(
                {
                    "index": target_idx,
                    "date": raw_df.loc[target_idx, date_column],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "y_true_norm": y_true_norm,
                    "y_pred_norm": y_pred_norm,
                }
            )

    fine_tuned_preds_df = pd.DataFrame(fine_tuned_preds)
    fine_tuned_metrics = compute_metrics(
        fine_tuned_preds_df["y_true"].to_numpy(dtype=np.float64),
        fine_tuned_preds_df["y_pred"].to_numpy(dtype=np.float64),
    )
    fine_tuned_norm_metrics = compute_metrics(
        fine_tuned_preds_df["y_true_norm"].to_numpy(dtype=np.float64),
        fine_tuned_preds_df["y_pred_norm"].to_numpy(dtype=np.float64),
    )

    print("baseline_2_fine_tuned")
    print("MAE:", fine_tuned_metrics["mae"])
    print("RMSE:", fine_tuned_metrics["rmse"])
    print("MAPE (%):", fine_tuned_metrics["mape"])
    print("Normalized MAE:", fine_tuned_norm_metrics["mae"])
    print("Normalized RMSE:", fine_tuned_norm_metrics["rmse"])
    print("Normalized MAPE (%):", fine_tuned_norm_metrics["mape"])

    summary_rows.append(
        {
            "dataset": dataset_name,
            "baseline_1_zero_shot_mae": zero_shot_metrics["mae"],
            "baseline_1_zero_shot_rmse": zero_shot_metrics["rmse"],
            "baseline_1_zero_shot_norm_mae": zero_shot_norm_metrics["mae"],
            "baseline_1_zero_shot_norm_rmse": zero_shot_norm_metrics["rmse"],
            "baseline_2_fine_tuned_mae": fine_tuned_metrics["mae"],
            "baseline_2_fine_tuned_rmse": fine_tuned_metrics["rmse"],
            "baseline_2_fine_tuned_norm_mae": fine_tuned_norm_metrics["mae"],
            "baseline_2_fine_tuned_norm_rmse": fine_tuned_norm_metrics["rmse"],
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
        "Norm_MAE:",
        row["baseline_1_zero_shot_norm_mae"],
        "Norm_RMSE:",
        row["baseline_1_zero_shot_norm_rmse"],
        "| baseline_2_fine_tuned MAE:",
        row["baseline_2_fine_tuned_mae"],
        "RMSE:",
        row["baseline_2_fine_tuned_rmse"],
        "Norm_MAE:",
        row["baseline_2_fine_tuned_norm_mae"],
        "Norm_RMSE:",
        row["baseline_2_fine_tuned_norm_rmse"],
    )
