from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from autogluon.common import space
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
parser.add_argument("--config", default="dlinear.yaml")
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
        train_end = int(cfg["fit_rows"])
        val_end = train_end

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()

    train_data = TimeSeriesDataFrame.from_data_frame(train_df, id_column="item_id", timestamp_column=date_column)
    val_data = TimeSeriesDataFrame.from_data_frame(val_df, id_column="item_id", timestamp_column=date_column)

    predictor = TimeSeriesPredictor(
        target=target_column,
        prediction_length=1,
        eval_metric="MAE",
    )

    # HPO with only training data and early stopping based on validation data
    predictor.fit(
        train_data,
        tuning_data=val_data,
        hyperparameters={
            "DLinear": {
                "context_length": space.Int(2, 20),
                "hidden_dimension": space.Int(8, 64),
                "kernel_size": space.Int(3, 25),
                "lr": space.Real(1e-4, 3e-3, log=True),
                "batch_size": space.Categorical(32, 64, 128),
                "max_epochs": space.Int(20, 200),
            }
        },
        hyperparameter_tune_kwargs={
            "num_trials": 20,
            "searcher": "random",
            "scheduler": "local",
        },
        enable_ensemble=False,
    )

    summary = predictor.fit_summary()
    best_model = summary["model_best"]
    best_hps = summary["model_hyperparams"][best_model]

    print("Best model:", best_model)
    print("Best hps:", best_hps)

    dataset_tag = str(dataset_name).replace("/", "_").replace(" ", "_")
    preds = []
    for i in range(val_end, len(df)):
        hist_df = df.iloc[:i].copy()
        hist_df["item_id"] = 1
        hist_ts = TimeSeriesDataFrame.from_data_frame(hist_df, id_column="item_id", timestamp_column=date_column)

        wf_predictor = TimeSeriesPredictor(
            target=target_column,
            prediction_length=1,
            eval_metric="MAE",
            path=f"ag_walkforward_{dataset_tag}_{i}",
        )
        wf_predictor.fit(
            hist_ts,
            hyperparameters={"DLinear": best_hps},
            enable_ensemble=False,
        )

        fcst = wf_predictor.predict(hist_ts, model="DLinear")
        pred_value = fcst["mean"].iloc[-1]
        true_value = df.loc[i, target_column]

        preds.append({"index": i, "date": df.loc[i, date_column], "y_true": true_value, "y_pred": pred_value})

    preds_df = pd.DataFrame(preds)
    err = preds_df["y_true"] - preds_df["y_pred"]
    mae = err.abs().mean()
    rmse = np.sqrt(np.mean(err ** 2))
    mape = np.mean(np.abs(err) / (np.abs(preds_df["y_true"]) + 1e-8)) * 100

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE (%):", mape)

    summary_rows.append(
        {
            "dataset": dataset_name,
            "mae": mae,
            "rmse": rmse,
        }
    )

print("summary")
for row in summary_rows:
    print(
        row["dataset"],
        "| MAE:",
        row["mae"],
        "RMSE:",
        row["rmse"],
    )
