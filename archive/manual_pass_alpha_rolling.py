"""
Manual alpha sweep with rolling window evaluation.

This mirrors manual_pass_alpha.py's gate sweep idea but evaluates using the
rolling-window evaluation in eval_fine_tune.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # noqa: BLE001
    plt = None

from fine_tuning.eval_fine_tune import _format_metrics
from fine_tuning.fine_tune_configs import load_dataset_config
from fine_tuning.fine_tune_dpt import (
    gate_stats,
    get_last_layer_gate_param,
    load_tabdpt_regressor,
    preprocess_features,
)
from fine_tuning.load_dataset import load_tabular_text_dataset
from fine_tuning.split_ts import time_split
from tabdpt.utils import pad_x


torch.manual_seed(0)
np.random.seed(0)

DATASET_CONFIG_PATH = "configs/bitcoin.yaml"
DEFAULT_DATASET = None

MAX_ROWS = 1000
CONTEXT_RATIO = 0.1
TUNE_RATIO = 0.1
EVAL_RATIO = 1 - CONTEXT_RATIO - TUNE_RATIO

# Sweep sigmoid gate values from 0.0 to 1.0.
GATING_VALUES = [round(x, 1) for x in np.arange(0.0, 1.0 + 0.1, 0.1)]
# If True, treat GATING_VALUES as raw logits instead of sigmoid values.
VALUES_ARE_LOGITS = False
# Clamp logit conversion for 0/1.
GATE_LOGIT_CLAMP = 10.0

# Eval range within the eval split.
EVAL_START = 0
EVAL_END = None
MAX_CONTEXT_FOR_EVAL = 1000

MODEL_WEIGHT_PATH = None
DEVICE = None

PLOT_OUTPUT_DIR = "figures/manual_pass_alpha_rolling"
PLOT_BASELINE = True
PLOT_PERCENT_ERRORS = True
PLOT_PERCENT_ERROR_DIFF = True
PRINT_CONTEXT_INDICES = True


def _prob_to_logit(prob: float, clamp: float) -> float:
    if prob <= 0.0:
        return -clamp
    if prob >= 1.0:
        return clamp
    return float(np.log(prob / (1.0 - prob)))


def _set_gate_weights(reg, weight) -> torch.nn.Parameter:
    gate = get_last_layer_gate_param(reg)
    if isinstance(weight, (list, tuple, np.ndarray)):
        weights = np.asarray(weight, dtype=np.float32).reshape(-1)
        if weights.size != gate.numel():
            raise ValueError(
                f"Per-head weights must match num_heads={gate.numel()}, got {weights.size}."
            )
        if VALUES_ARE_LOGITS:
            logits = weights
        else:
            logits = np.array([_prob_to_logit(w, GATE_LOGIT_CLAMP) for w in weights], dtype=np.float32)
    else:
        if VALUES_ARE_LOGITS:
            logits = np.full((gate.numel(),), float(weight), dtype=np.float32)
        else:
            logit = _prob_to_logit(float(weight), GATE_LOGIT_CLAMP)
            logits = np.full((gate.numel(),), logit, dtype=np.float32)
    with torch.no_grad():
        gate.copy_(torch.tensor(logits, dtype=gate.dtype, device=gate.device))
    return gate


def _slice_eval_range(
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    text_eval: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    start = 0 if EVAL_START is None else int(EVAL_START)
    end = len(y_eval) if EVAL_END is None else int(EVAL_END)
    if start < 0 or start >= len(y_eval):
        raise ValueError(f"EVAL_START must be in [0, {len(y_eval) - 1}], got {start}.")
    if end <= start or end > len(y_eval):
        raise ValueError(f"EVAL_END must be in [{start + 1}, {len(y_eval)}], got {end}.")
    return start, end, X_eval_proc[start:end], y_eval[start:end], text_eval[start:end]


def _parse_gating_vector(value: str) -> np.ndarray:
    try:
        return np.array([float(v.strip()) for v in value.split(",") if v.strip()], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Invalid gating vector: {value}") from exc


def _compute_metrics_from_preds(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return mae, rmse, mape


def _compute_percent_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return np.abs(y_true - y_pred) / denom


def _rolling_eval_with_errors(
    reg,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray | None,
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    text_eval: np.ndarray | None,
    use_text: bool,
    label: str,
    max_context: int | None,
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    if use_text and (text_context is None or text_eval is None):
        raise ValueError("Rolling eval with text requires context and eval text arrays.")

    reg.model.eval()
    preds = np.zeros(len(y_eval), dtype=np.float32)
    with torch.no_grad():
        for idx in range(len(y_eval)):
            X_train_full = np.concatenate((X_context_proc, X_eval_proc[:idx]))
            y_train_full = np.concatenate((y_context, y_eval[:idx]))
            if use_text:
                text_train_full = np.concatenate((text_context, text_eval[:idx]), axis=0)

            if max_context is not None:
                X_train_step = X_train_full[-max_context:]
                y_train_step = y_train_full[-max_context:]
                if use_text:
                    text_train_step = text_train_full[-max_context:]
            else:
                X_train_step = X_train_full
                y_train_step = y_train_full
                if use_text:
                    text_train_step = text_train_full

            X_test_step = X_eval_proc[idx:idx + 1]

            if use_text:
                train_text_batch = text_train_step[None, ...]
                test_text_batch = text_eval[idx:idx + 1][None, ...]
                text_enhanced_attn_weight = reg._compute_attn_weight_pairwise_avg(
                    train_text_batch,
                    test_text_batch,
                )
            else:
                text_enhanced_attn_weight = None

            X_train_tensor = torch.tensor(X_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_train_tensor = pad_x(X_train_tensor, reg.max_features)
            X_test_tensor = torch.tensor(X_test_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_test_tensor = pad_x(X_test_tensor, reg.max_features)
            y_context_tensor = torch.tensor(y_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

            pred = reg.model(
                x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                y_src=y_context_tensor.unsqueeze(-1),
                task="reg",
                text_enhanced_attn_weight=text_enhanced_attn_weight,
            )
            preds[idx] = pred.squeeze(-1).reshape(-1).detach().cpu().numpy()[0]

    mae, rmse, mape = _compute_metrics_from_preds(y_eval, preds)
    _format_metrics(label, mae, rmse, mape)
    abs_errors = np.abs(y_eval - preds)
    pct_errors = _compute_percent_errors(y_eval, preds)
    return (mae, rmse, mape), abs_errors, pct_errors


def _plot_abs_errors(
    *,
    abs_errors: np.ndarray,
    label: str,
    baseline_errors: np.ndarray | None,
    output_path: Path,
) -> None:
    if plt is None:
        print("Matplotlib not available; skipping plot.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(abs_errors, label=label)
    if baseline_errors is not None:
        plt.plot(baseline_errors, label="baseline")
    plt.title("Absolute Error per Prediction")
    plt.xlabel("Prediction Index")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_percent_errors(
    *,
    percent_errors: np.ndarray,
    label: str,
    baseline_errors: np.ndarray | None,
    output_path: Path,
) -> None:
    if plt is None:
        print("Matplotlib not available; skipping plot.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(percent_errors, label=label)
    if baseline_errors is not None:
        plt.plot(baseline_errors, label="baseline")
    plt.title("Absolute Percentage Error per Prediction")
    plt.xlabel("Prediction Index")
    plt.ylabel("Absolute Percentage Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_percent_error_diff(
    *,
    percent_errors: np.ndarray,
    baseline_errors: np.ndarray,
    label: str,
    output_path: Path,
) -> None:
    if plt is None:
        print("Matplotlib not available; skipping plot.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diff = percent_errors - baseline_errors
    plt.figure(figsize=(10, 4))
    plt.plot(diff, label=label)
    plt.axhline(0.0, color="black", linewidth=1.0)
    plt.title("Percentage Error Difference (gating - baseline)")
    plt.xlabel("Prediction Index")
    plt.ylabel("Percentage Error Difference")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _summarize_diff_area(
    *,
    percent_errors: np.ndarray,
    baseline_errors: np.ndarray,
    label: str,
) -> None:
    diff = percent_errors - baseline_errors
    above = float(diff[diff > 0].sum())
    below = float(diff[diff < 0].sum())
    print(
        f"Diff area ({label}) | above0={above:.6f} below0={below:.6f} "
        f"abs_below0={abs(below):.6f}"
    )


def _print_context_indices(
    *,
    context_len: int,
    eval_len: int,
    max_context: int | None,
) -> None:
    print("\n== Context indices per prediction ==")
    for idx in range(eval_len):
        context_end = context_len + idx
        if max_context is None:
            start_idx = 0
        else:
            start_idx = max(0, context_end - max_context)
        print(f"pred_idx={idx} context_idx=[{start_idx}:{context_end})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual gate sweep with rolling evaluation."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset key in the config file.")
    parser.add_argument("--config", default=DATASET_CONFIG_PATH, help="Path to dataset config YAML.")
    parser.add_argument(
        "--gating-vector",
        default=None,
        help="Comma-separated per-head gating values (sigmoid space). Overrides sweep if set.",
    )
    args = parser.parse_args()

    dataset_cfg = load_dataset_config(args.config, args.dataset)
    data_path = dataset_cfg.get("data_path")
    date_column = dataset_cfg.get("date_column")
    numeric_features = dataset_cfg.get("numeric_features")
    target_column = dataset_cfg.get("target_column")
    embedding_columns = dataset_cfg.get("embedding_columns")
    embedding_lags = dataset_cfg.get("embedding_lags")
    embedding_column_template = dataset_cfg.get("embedding_column_template")

    if args.dataset:
        print(f"Using dataset '{args.dataset}' from {args.config}")
    else:
        print(f"Using dataset config {args.config}")

    X, y, text = load_tabular_text_dataset(
        path=data_path,
        date_column=date_column,
        numeric_features=numeric_features,
        target_column=target_column,
        embedding_lags=embedding_lags,
        embedding_columns=embedding_columns,
        embedding_column_template=embedding_column_template,
        max_rows=MAX_ROWS,
    )
    if X.shape[1] == 0:
        X = np.zeros((X.shape[0], 1), dtype=np.float32)

    (
        X_context,
        y_context,
        text_context,
        X_tune,
        y_tune,
        text_tune,
        X_eval,
        y_eval,
        text_eval,
    ) = time_split(
        X,
        y,
        text,
        context_ratio=CONTEXT_RATIO,
        tune_ratio=TUNE_RATIO,
        eval_ratio=EVAL_RATIO,
    )
    print(f"Split sizes: context={len(y_context)} tune={len(y_tune)} eval={len(y_eval)}")

    reg = load_tabdpt_regressor(
        device=DEVICE,
        model_weight_path=MODEL_WEIGHT_PATH,
        text_enhanced=True,
        use_flash=True,
        compile_model=True,
    )
    X_fit = np.concatenate((X_context, X_tune))
    y_fit = np.concatenate((y_context, y_tune))
    text_fit = np.concatenate((text_context, text_tune), axis=0)
    reg.fit(X_fit, y_fit, text_fit)

    reduction_mode = None
    reduction_payload = None
    X_context_proc = preprocess_features(
        reg,
        X_context,
        reduction_mode=reduction_mode,
        reduction_payload=reduction_payload,
    )
    X_tune_proc = preprocess_features(
        reg,
        X_tune,
        reduction_mode=reduction_mode,
        reduction_payload=reduction_payload,
    )
    X_eval_proc = preprocess_features(
        reg,
        X_eval,
        reduction_mode=reduction_mode,
        reduction_payload=reduction_payload,
    )

    eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
    eval_y_context = np.concatenate((y_context, y_tune))
    eval_text_context = np.concatenate((text_context, text_tune), axis=0)

    start, end, X_eval_range, y_eval_range, text_eval_range = _slice_eval_range(
        X_eval_proc,
        y_eval,
        text_eval,
    )
    if start > 0:
        eval_context_proc = np.concatenate((eval_context_proc, X_eval_proc[:start]))
        eval_y_context = np.concatenate((eval_y_context, y_eval[:start]))
        eval_text_context = np.concatenate((eval_text_context, text_eval[:start]), axis=0)
    print(f"Eval range: [{start}:{end}) out of eval split ({len(y_eval)} rows).")
    if PRINT_CONTEXT_INDICES:
        _print_context_indices(
            context_len=len(eval_context_proc),
            eval_len=len(y_eval_range),
            max_context=MAX_CONTEXT_FOR_EVAL,
        )

    print("\n== Baseline (no text attn) ==")
    baseline, baseline_errors, baseline_pct_errors = _rolling_eval_with_errors(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=eval_text_context,
        X_eval_proc=X_eval_range,
        y_eval=y_eval_range,
        text_eval=text_eval_range,
        use_text=False,
        label="Eval (no text attn)",
        max_context=MAX_CONTEXT_FOR_EVAL,
    )
    if PLOT_BASELINE:
        _plot_abs_errors(
            abs_errors=baseline_errors,
            label="baseline",
            baseline_errors=None,
            output_path=Path(PLOT_OUTPUT_DIR) / "abs_error_baseline.png",
        )
        if PLOT_PERCENT_ERRORS:
            _plot_percent_errors(
                percent_errors=baseline_pct_errors,
                label="baseline",
                baseline_errors=None,
                output_path=Path(PLOT_OUTPUT_DIR) / "percent_error_baseline.png",
            )

    print("\n== Manual gate sweep ==")
    if args.gating_vector:
        gating_items = [( _parse_gating_vector(args.gating_vector), "custom" )]
    else:
        gating_items = [(gating, f"{gating:.1f}") for gating in GATING_VALUES]

    for gating, label_suffix in gating_items:
        gate = _set_gate_weights(reg, gating)
        print(f"\n-- Gating={label_suffix} | {gate_stats(gate)} --")
        _, abs_errors, pct_errors = _rolling_eval_with_errors(
            reg,
            X_context_proc=eval_context_proc,
            y_context=eval_y_context,
            text_context=eval_text_context,
            X_eval_proc=X_eval_range,
            y_eval=y_eval_range,
            text_eval=text_eval_range,
            use_text=True,
            label=f"Eval (gating={label_suffix})",
            max_context=MAX_CONTEXT_FOR_EVAL,
        )
        _plot_abs_errors(
            abs_errors=abs_errors,
            label=f"gating={label_suffix}",
            baseline_errors=baseline_errors if PLOT_BASELINE else None,
            output_path=Path(PLOT_OUTPUT_DIR) / f"abs_error_gating_{label_suffix}.png",
        )
        if PLOT_PERCENT_ERRORS:
            _plot_percent_errors(
                percent_errors=pct_errors,
                label=f"gating={label_suffix}",
                baseline_errors=baseline_pct_errors if PLOT_BASELINE else None,
                output_path=Path(PLOT_OUTPUT_DIR) / f"percent_error_gating_{label_suffix}.png",
            )
        if PLOT_PERCENT_ERROR_DIFF and PLOT_BASELINE:
            _summarize_diff_area(
                percent_errors=pct_errors,
                baseline_errors=baseline_pct_errors,
                label=f"gating={label_suffix}",
            )
            _plot_percent_error_diff(
                percent_errors=pct_errors,
                baseline_errors=baseline_pct_errors,
                label=f"gating={label_suffix}",
                output_path=Path(PLOT_OUTPUT_DIR) / f"percent_error_diff_{label_suffix}.png",
            )

    print("\n== Baseline recap ==")
    _format_metrics("Eval (no text attn)", *baseline)


if __name__ == "__main__":
    main()
