"""Rolling TabDPT and TabPFN baselines."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, REPO_ROOT / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fine_tuning.eval_fine_tune import (  # noqa: E402
    _build_rolling_train_step,
    _compute_metrics,
    _format_metrics,
    evaluate_rolling,
)
from fine_tuning.load_dataset import (  # noqa: E402
    build_causal_fixed_origin_horizon_splits,
    build_direct_multi_horizon_dataset,
    select_direct_horizon_targets,
    validate_direct_mode_numeric_features,
)
from fine_tuning.split_ts import time_split  # noqa: E402
from tabdpt import TabDPTRegressor  # noqa: E402


@dataclass(frozen=True)
class BaselineDataSplits:
    X_context: np.ndarray
    y_context: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def resolve_config_path(config_path: str) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (SCRIPT_DIR / path).resolve()


def repo_path(path_str: str | None) -> str | None:
    if path_str is None:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def resolve_device(device: str | None) -> str:
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def read_cfg(config_path: str, dataset: str | None) -> tuple[Path, dict[str, dict]]:
    config_file = resolve_config_path(config_path)
    with config_file.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    if isinstance(raw_cfg, dict) and "data_path" in raw_cfg:
        dataset_name = dataset or config_file.stem
        return config_file, {dataset_name: dict(raw_cfg)}

    datasets = raw_cfg.get("datasets", raw_cfg)
    if dataset is None:
        return config_file, {name: dict(cfg) for name, cfg in datasets.items()}
    if dataset not in datasets:
        available = ", ".join(sorted(str(name) for name in datasets.keys()))
        raise ValueError(
            f"Dataset '{dataset}' not found in {config_file}. "
            f"Available datasets: {available}"
        )
    return config_file, {dataset: dict(datasets[dataset])}


def parse_cfg(raw_cfg: dict, args: argparse.Namespace) -> dict:
    cfg = dict(raw_cfg)
    cfg["data_path"] = repo_path(cfg["data_path"])
    cfg["numeric_features"] = list(
        cfg.get("numeric_features") or cfg.get("features") or cfg.get("feature_columns") or []
    )
    cfg["target_column"] = str(cfg.get("target_column") or cfg.get("target"))
    cfg["prediction_window"] = int(cfg.get("prediction_window", 1))
    cfg["max_context"] = cfg.get("max_context")
    cfg["seed"] = int(cfg.get("seed", 0))
    cfg["calendar_frequency"] = cfg.get("calendar_frequency")
    cfg["seasonality_k"] = int(cfg.get("seasonality_k", 3))
    cfg["seasonality_L"] = cfg.get("seasonality_L")
    if cfg["seasonality_k"] <= 0:
        raise ValueError("seasonality_k must be positive.")

    cfg["tabdpt"] = {
        "device": cfg.get("tabdpt", {}).get("device", "auto"),
        "model_weight_path": repo_path(cfg.get("tabdpt", {}).get("model_weight_path")),
        "use_flash": bool(cfg.get("tabdpt", {}).get("use_flash", True)),
        "compile_model": bool(cfg.get("tabdpt", {}).get("compile_model", True)),
    }
    cfg["tabpfn"] = {
        "device": cfg.get("tabpfn", {}).get("device", "auto"),
        "model_path": repo_path(cfg.get("tabpfn", {}).get("model_path")),
        "cache_dir": repo_path(cfg.get("tabpfn", {}).get("cache_dir", "models/tabpfn")),
    }

    if args.tabpfn_model_path:
        cfg["tabpfn"]["model_path"] = repo_path(args.tabpfn_model_path)
    if args.tabpfn_cache_dir:
        cfg["tabpfn"]["cache_dir"] = repo_path(args.tabpfn_cache_dir)
    if cfg["tabpfn"]["cache_dir"]:
        Path(cfg["tabpfn"]["cache_dir"]).mkdir(parents=True, exist_ok=True)

    return cfg


def load_dataframe(cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["data_path"])
    if cfg.get("date_column") and cfg["date_column"] in df.columns:
        df[cfg["date_column"]] = pd.to_datetime(df[cfg["date_column"]])
        df = df.sort_values(cfg["date_column"]).reset_index(drop=True)
    if cfg.get("max_rows") is not None:
        df = df.head(int(cfg["max_rows"])).reset_index(drop=True)
    return df


def load_numeric_arrays(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load pre-lagged feature rows and the one-step target column."""
    df = load_dataframe(cfg)
    validate_direct_mode_numeric_features(cfg["numeric_features"])
    X = df[cfg["numeric_features"]].astype(np.float32).to_numpy()
    if X.shape[1] == 0:
        X = np.zeros((len(df), 1), dtype=np.float32)
    y = df[cfg["target_column"]].astype(np.float32).to_numpy()
    return X, y


def _base_covariate_columns(cfg: dict, df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for feature in cfg["numeric_features"]:
        if feature.endswith("_lag") and feature.rsplit("_lag", 1)[0] in df.columns:
            base = feature.rsplit("_lag", 1)[0]
        else:
            base = feature.split("_lag", 1)[0]
        if base in df.columns and base != cfg["target_column"] and base not in cols:
            cols.append(base)
    return cols


def build_lagged_arrays(
    df: pd.DataFrame,
    cfg: dict,
    target_lag_days: int,
    covariate_lag_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    if target_lag_days < 0 or covariate_lag_days < 0:
        raise ValueError("Lag days must be non-negative.")

    target_col = cfg["target_column"]
    covariate_cols = _base_covariate_columns(cfg, df)
    y_series = df[target_col].astype(np.float32)
    if max(target_lag_days, covariate_lag_days) == 0:
        return load_numeric_arrays(cfg)

    feature_blocks: list[pd.Series] = []
    for lag in range(1, target_lag_days + 1):
        feature_blocks.append(y_series.shift(lag))
    for col in covariate_cols:
        cov_series = df[col].astype(np.float32)
        for lag in range(1, covariate_lag_days + 1):
            feature_blocks.append(cov_series.shift(lag))

    # If no covariate columns are available, keep target-lag-only features valid.
    if not feature_blocks:
        raise ValueError("No lagged features could be built from the provided config/data.")

    X_df = pd.concat(feature_blocks, axis=1)
    valid_mask = X_df.notna().all(axis=1)
    X = X_df[valid_mask].to_numpy(dtype=np.float32)
    y = y_series[valid_mask].to_numpy(dtype=np.float32)
    if len(X) <= cfg["prediction_window"]:
        raise ValueError("Not enough rows left after lagging for the configured prediction window.")
    return X, y


def build_direct_horizon_dataset(
    X_rows: np.ndarray,
    y_targets: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep the same lagged feature row and shift the scalar target forward.

    Example with lag=5 in row form:
    - row 1: y1,y2,y3,y4,y5 -> y6
    - row 2: y2,y3,y4,y5,y6 -> y7

    Horizon 2 keeps the same feature rows but changes the target:
    - row 1: y1,y2,y3,y4,y5 -> y7
    - row 2: y2,y3,y4,y5,y6 -> y8
    """
    target_shift = horizon - 1
    num_rows = len(y_targets) - target_shift
    X_horizon = X_rows[:num_rows]
    y_horizon = y_targets[target_shift:target_shift + num_rows]
    return X_horizon, y_horizon


def select_splits_for_horizon(
    splits: BaselineDataSplits,
    horizon: int,
) -> BaselineDataSplits:
    X_context, y_context, _ = select_direct_horizon_targets(
        splits.X_context,
        splits.y_context,
        np.zeros((len(splits.y_context), 0, 0), dtype=np.float32),
        horizon=horizon,
    )
    X_train, y_train, _ = select_direct_horizon_targets(
        splits.X_train,
        splits.y_train,
        np.zeros((len(splits.y_train), 0, 0), dtype=np.float32),
        horizon=horizon,
    )
    X_val, y_val, _ = select_direct_horizon_targets(
        splits.X_val,
        splits.y_val,
        np.zeros((len(splits.y_val), 0, 0), dtype=np.float32),
        horizon=horizon,
    )
    X_test, y_test, _ = select_direct_horizon_targets(
        splits.X_test,
        splits.y_test,
        np.zeros((len(splits.y_test), 0, 0), dtype=np.float32),
        horizon=horizon,
    )
    return BaselineDataSplits(
        X_context=X_context,
        y_context=y_context,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


def build_horizon_splits(
    X_rows: np.ndarray,
    y_targets: np.ndarray,
    cfg: dict,
    horizon: int,
    timestamps: pd.Series | None = None,
) -> BaselineDataSplits:
    dummy_text = np.zeros((len(y_targets), 0, 0), dtype=np.float32)
    if timestamps is None:
        timestamps = pd.Series(pd.date_range("2000-01-01", periods=len(y_targets), freq="D"))
    split_values = build_causal_fixed_origin_horizon_splits(
        X_rows,
        y_targets,
        dummy_text,
        timestamps,
        context_ratio=float(cfg["context_ratio"]),
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
        horizon=horizon,
        calendar_frequency=cfg.get("calendar_frequency"),
        seasonality_k=int(cfg.get("seasonality_k", 3)),
        seasonality_L=cfg.get("seasonality_L"),
    )
    return BaselineDataSplits(
        X_context=split_values[0],
        y_context=split_values[1],
        X_train=split_values[3],
        y_train=split_values[4],
        X_val=split_values[6],
        y_val=split_values[7],
        X_test=split_values[9],
        y_test=split_values[10],
    )


def split_data(X: np.ndarray, y: np.ndarray, cfg: dict) -> BaselineDataSplits:
    dummy_text = np.zeros((len(y), 0, 0), dtype=np.float32)
    split_values = time_split(
        X,
        y,
        dummy_text,
        context_ratio=float(cfg["context_ratio"]),
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
    )
    return BaselineDataSplits(
        X_context=split_values[0],
        y_context=split_values[1],
        X_train=split_values[3],
        y_train=split_values[4],
        X_val=split_values[6],
        y_val=split_values[7],
        X_test=split_values[9],
        y_test=split_values[10],
    )


def preprocess_features(reg: TabDPTRegressor, X: np.ndarray) -> np.ndarray:
    """Apply the preprocessing learned by TabDPT on the context split."""
    X_proc = X
    if reg.missing_indicators:
        inds = np.isnan(X_proc)[:, reg.has_missing_indicator].astype(float)
        X_proc = np.hstack((X_proc, inds))
    X_proc = reg.imputer.transform(X_proc)
    if reg.scaler:
        X_proc = reg.scaler.transform(X_proc)
        if reg.normalizer == "quantile-uniform":
            X_proc = 2 * X_proc - 1
    return X_proc.astype(np.float32, copy=False)


def build_history_and_eval_split(
    X_context: np.ndarray,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_context: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    split_name: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the rolling history prefix and the eval segment for train, val, or test."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    def _trim_history_tail(X_hist: np.ndarray, y_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        trim = max(0, horizon - 1)
        if trim == 0:
            return X_hist, y_hist
        if trim >= len(y_hist):
            return X_hist[:0], y_hist[:0]
        return X_hist[:-trim], y_hist[:-trim]

    if split_name == "train":
        return X_context, y_context, X_train, y_train
    if split_name == "val":
        X_hist, y_hist = _trim_history_tail(
            np.concatenate((X_context, X_train), axis=0),
            np.concatenate((y_context, y_train), axis=0),
        )
        return X_hist, y_hist, X_val, y_val
    X_hist, y_hist = _trim_history_tail(
        np.concatenate((X_context, X_train, X_val), axis=0),
        np.concatenate((y_context, y_train, y_val), axis=0),
    )
    return X_hist, y_hist, X_test, y_test


def evaluate_rolling_estimator(
    estimator,
    *,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    max_context: int | None,
    horizon: int,
    label: str,
) -> tuple[float, float, float]:
    """Refit the estimator at every rolling step and predict one row ahead."""
    preds = np.zeros(len(y_eval), dtype=np.float32)
    for idx in range(len(y_eval)):
        X_train_step, y_train_step, _ = _build_rolling_train_step(
            X_context_proc=X_context,
            y_context=y_context,
            text_context=None,
            X_eval_proc=X_eval,
            y_eval=y_eval,
            text_eval=None,
            idx=idx,
            horizon=horizon,
            max_context=max_context,
        )
        estimator.fit(X_train_step, y_train_step)
        preds[idx] = np.asarray(estimator.predict(X_eval[idx:idx + 1]), dtype=np.float32).reshape(-1)[0]

    metrics = _compute_metrics(y_eval, preds)
    _format_metrics(label, *metrics)
    return metrics


def make_tabdpt(cfg: dict) -> TabDPTRegressor:
    return TabDPTRegressor(
        device=resolve_device(cfg["tabdpt"]["device"]),
        model_weight_path=cfg["tabdpt"]["model_weight_path"],
        use_flash=cfg["tabdpt"]["use_flash"],
        compile=cfg["tabdpt"]["compile_model"],
    )


def make_tabpfn(cfg: dict):
    if cfg["tabpfn"]["cache_dir"]:
        os.environ["TABPFN_MODEL_CACHE_DIR"] = cfg["tabpfn"]["cache_dir"]
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        model_path=cfg["tabpfn"]["model_path"] or "auto",
        device=resolve_device(cfg["tabpfn"]["device"]),
        random_state=cfg["seed"],
        fit_mode="fit_preprocessors",
        ignore_pretraining_limits=True,
        n_preprocessing_jobs=1,
    )


def run_horizon(
    cfg: dict,
    X_rows: np.ndarray,
    y_targets: np.ndarray,
    timestamps: pd.Series,
    horizon: int,
    use_tabpfn: bool,
) -> dict[str, tuple[float, float, float]]:
    """Run one direct horizon using the same aligned dataset construction as fine-tuning."""
    print(f"\n{'=' * 16} Horizon {horizon}/{cfg['prediction_window']} {'=' * 16}")
    splits = build_horizon_splits(X_rows, y_targets, cfg, horizon, timestamps)

    tabdpt = make_tabdpt(cfg)
    tabdpt.fit(splits.X_context, splits.y_context)

    X_context_proc = preprocess_features(tabdpt, splits.X_context)
    X_train_proc = preprocess_features(tabdpt, splits.X_train)
    X_val_proc = preprocess_features(tabdpt, splits.X_val)
    X_test_proc = preprocess_features(tabdpt, splits.X_test)

    metrics: dict[str, tuple[float, float, float]] = {}
    for split_name in ("val", "test"):
        X_history, y_history, X_eval, y_eval = build_history_and_eval_split(
            splits.X_context,
            splits.X_train,
            splits.X_val,
            splits.X_test,
            splits.y_context,
            splits.y_train,
            splits.y_val,
            splits.y_test,
            split_name,
            horizon,
        )
        X_history_proc, y_history_proc, X_eval_proc, y_eval_proc = build_history_and_eval_split(
            X_context_proc,
            X_train_proc,
            X_val_proc,
            X_test_proc,
            splits.y_context,
            splits.y_train,
            splits.y_val,
            splits.y_test,
            split_name,
            horizon,
        )

        metrics[f"{split_name}_tabdpt"] = evaluate_rolling(
            tabdpt,
            X_context_proc=X_history_proc,
            y_context=y_history_proc,
            text_context=None,
            X_eval_proc=X_eval_proc,
            y_eval=y_eval_proc,
            text_eval=None,
            use_text=False,
            label=f"{split_name.title()} TabDPT",
            max_context=cfg["max_context"],
            horizon=horizon,
        )

        if use_tabpfn:
            metrics[f"{split_name}_tabpfn"] = evaluate_rolling_estimator(
                make_tabpfn(cfg),
                X_context=X_history,
                y_context=y_history,
                X_eval=X_eval,
                y_eval=y_eval,
                max_context=cfg["max_context"],
                horizon=horizon,
                label=f"{split_name.title()} TabPFN",
            )

    return metrics


def mean_metrics(metric_list: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    values = np.asarray(metric_list, dtype=np.float64)
    return tuple(values.mean(axis=0).tolist())


def parse_lag_values(lag_args: Iterable[int] | None) -> list[int]:
    if not lag_args:
        return []
    values = sorted({int(v) for v in lag_args})
    if any(v < 0 for v in values):
        raise ValueError("Lag values must be non-negative.")
    return values


def evaluate_tabdpt_lag_combo(
    cfg: dict,
    X_rows: np.ndarray,
    y_targets: np.ndarray,
    timestamps: pd.Series,
) -> tuple[float, dict[int, tuple[float, float, float]]]:
    per_horizon_metrics: dict[int, tuple[float, float, float]] = {}
    maes: list[float] = []
    for horizon in range(1, cfg["prediction_window"] + 1):
        metrics = run_horizon(cfg, X_rows, y_targets, timestamps, horizon, use_tabpfn=False)
        val_metrics = metrics["val_tabdpt"]
        per_horizon_metrics[horizon] = val_metrics
        maes.append(val_metrics[0])
    return float(np.mean(maes)), per_horizon_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tabular.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--tabpfn-model-path")
    parser.add_argument("--tabpfn-cache-dir")
    parser.add_argument("--skip-tabpfn", action="store_true")
    parser.add_argument("--target-lag-days", type=int, default=None)
    parser.add_argument("--covariate-lag-days", type=int, default=None)
    parser.add_argument("--search-target-lags", type=int, nargs="+", default=None)
    parser.add_argument("--search-covariate-lags", type=int, nargs="+", default=None)
    args = parser.parse_args()

    config_file, raw_cfgs = read_cfg(args.config, args.dataset)
    summaries: list[tuple[str, dict[str, tuple[float, float, float]]]] = []

    for dataset_name, raw_cfg in raw_cfgs.items():
        cfg = parse_cfg(raw_cfg, args)
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        df = load_dataframe(cfg)
        timestamps = df[cfg["date_column"]].reset_index(drop=True) if cfg.get("date_column") else pd.Series(range(len(df)))

        search_target_lags = parse_lag_values(args.search_target_lags)
        search_covariate_lags = parse_lag_values(args.search_covariate_lags)

        if bool(search_target_lags) ^ bool(search_covariate_lags):
            raise ValueError(
                "Both --search-target-lags and --search-covariate-lags are required for lag search."
            )

        selected_target_lag = args.target_lag_days
        selected_covariate_lag = args.covariate_lag_days

        if search_target_lags and search_covariate_lags:
            print("\nLag search (TabDPT validation MAE)")
            best_combo: tuple[int, int] | None = None
            best_mae = float("inf")
            for target_lag in search_target_lags:
                for covariate_lag in search_covariate_lags:
                    print(f"trying target_lag_days={target_lag}, covariate_lag_days={covariate_lag}")
                    try:
                        X_candidate, y_candidate = build_lagged_arrays(
                            df, cfg, target_lag_days=target_lag, covariate_lag_days=covariate_lag
                        )
                        valid_mask = pd.Series(True, index=df.index)
                        max_lag = max(target_lag, covariate_lag)
                        if max_lag > 0:
                            valid_mask.iloc[:max_lag] = False
                        mean_mae, _ = evaluate_tabdpt_lag_combo(
                            cfg,
                            X_candidate,
                            y_candidate,
                            df.loc[valid_mask, cfg["date_column"]].reset_index(drop=True),
                        )
                        print(f"mean_val_tabdpt_mae={mean_mae:.6f}")
                        if mean_mae < best_mae:
                            best_mae = mean_mae
                            best_combo = (target_lag, covariate_lag)
                    except Exception as exc:
                        print(f"skipping invalid lag combo: {exc}")
            if best_combo is None:
                raise RuntimeError("Lag search did not produce any valid candidate.")
            selected_target_lag, selected_covariate_lag = best_combo
            print(
                f"best_lag_combo: target_lag_days={selected_target_lag}, "
                f"covariate_lag_days={selected_covariate_lag}, mean_val_tabdpt_mae={best_mae:.6f}"
            )

        if selected_target_lag is not None or selected_covariate_lag is not None:
            target_lag = int(selected_target_lag or 0)
            covariate_lag = int(selected_covariate_lag or 0)
            X, y = build_lagged_arrays(
                df, cfg, target_lag_days=target_lag, covariate_lag_days=covariate_lag
            )
            max_lag = max(target_lag, covariate_lag)
            timestamps = df.loc[max_lag:, cfg["date_column"]].reset_index(drop=True)
        else:
            X, y = load_numeric_arrays(cfg)

        print(dataset_name)
        print(f"config={config_file}")
        print(f"prediction_window={cfg['prediction_window']}")
        print(f"max_context={cfg['max_context']}")
        if selected_target_lag is not None or selected_covariate_lag is not None:
            print(f"target_lag_days={int(selected_target_lag or 0)}")
            print(f"covariate_lag_days={int(selected_covariate_lag or 0)}")
        if not args.skip_tabpfn:
            if cfg["tabpfn"]["model_path"]:
                print(f"tabpfn_model_path={cfg['tabpfn']['model_path']}")
            else:
                print(f"tabpfn_cache_dir={cfg['tabpfn']['cache_dir']}")

        per_horizon = [
            run_horizon(cfg, X, y, timestamps, horizon, not args.skip_tabpfn)
            for horizon in range(1, cfg["prediction_window"] + 1)
        ]

        print("\nSummary")
        summary: dict[str, tuple[float, float, float]] = {}
        for name in sorted({name for result in per_horizon for name in result}):
            summary[name] = mean_metrics([result[name] for result in per_horizon if name in result])
            _format_metrics(name, *summary[name])
        summaries.append((dataset_name, summary))
        print()

    print("summary")
    for dataset_name, summary in summaries:
        print(dataset_name)
        for name in sorted(summary):
            _format_metrics(name, *summary[name])


if __name__ == "__main__":
    main()
