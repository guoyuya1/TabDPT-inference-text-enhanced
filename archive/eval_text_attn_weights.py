"""
Evaluate a list of text-attention gate weights with rolling evaluation.

This script reuses helpers from fine_tune_dpt.py and eval_fine_tune.py.
"""

from __future__ import annotations

import numpy as np
import torch

from fine_tuning.eval_fine_tune import _format_metrics, evaluate_rolling
from fine_tuning.fine_tune_dpt import (
    gate_stats,
    get_last_layer_gate_param,
    load_tabdpt_regressor,
    preprocess_features,
)
from fine_tuning.load_dataset import load_tabular_text_dataset
from fine_tuning.split_ts import time_split


# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/bitcoin/bitcoin_final_with_embeddings_lag_3.csv"
DATE_COLUMN = "Date"
NUMERIC_FEATURES = [
    "Open_lag1", "Open_lag2", "Open_lag3",
    "High_lag1", "High_lag2", "High_lag3",
    "Low_lag1", "Low_lag2", "Low_lag3",
    "Close_lag1", "Close_lag2", "Close_lag3",
    "Adj_Close_lag1", "Adj_Close_lag2", "Adj_Close_lag3",
]
TARGET_COLUMN = "Adj_Close"
TEXT_FEATURE_NAME = "summary_gpt-5-mini"

TEXT_EMBEDDING_LAGS = [0, 1, 2, 3]
EMBEDDING_COLUMNS = None
EMBEDDING_COLUMN_TEMPLATE = f"embedding_{TEXT_FEATURE_NAME}_lag{{lag}}"

MAX_ROWS = 1000
CONTEXT_RATIO = 0.2
TUNE_RATIO = 0.6
EVAL_RATIO = 0.2

# List of desired text-attention weights (sigmoid values in [0, 1]).
TEXT_ATTENTION_WEIGHTS = [round(x, 1) for x in np.arange(0.0, 1.0 + 0.1, 0.1)]
# If True, interpret TEXT_ATTENTION_WEIGHTS as raw logits instead of sigmoid values.
WEIGHTS_ARE_LOGITS = False
# Clamp for logit conversion when weights hit 0 or 1.
GATE_LOGIT_CLAMP = 10.0

# Eval range within the eval split (0-based, end-exclusive).
EVAL_START = 0
EVAL_END = None
MAX_CONTEXT_FOR_EVAL = 1000

# Model loading.
MODEL_WEIGHT_PATH = None
DEVICE = None


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
        if WEIGHTS_ARE_LOGITS:
            logits = weights
        else:
            logits = np.array([_prob_to_logit(w, GATE_LOGIT_CLAMP) for w in weights], dtype=np.float32)
    else:
        if WEIGHTS_ARE_LOGITS:
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


def main() -> None:
    X, y, text = load_tabular_text_dataset(
        path=DATA_PATH,
        date_column=DATE_COLUMN,
        numeric_features=NUMERIC_FEATURES,
        target_column=TARGET_COLUMN,
        embedding_lags=TEXT_EMBEDDING_LAGS,
        embedding_columns=EMBEDDING_COLUMNS,
        embedding_column_template=EMBEDDING_COLUMN_TEMPLATE,
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

    reg = load_tabdpt_regressor(
        device=DEVICE,
        model_weight_path=MODEL_WEIGHT_PATH,
        text_enhanced=True,
        use_flash=True,
        compile_model=True,
    )
    reg.fit(X_context, y_context, text_context)

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

    print("\n== Baseline (no text attn) ==")
    baseline = evaluate_rolling(
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

    print("\n== Text-attention weights ==")
    for weight in TEXT_ATTENTION_WEIGHTS:
        gate = _set_gate_weights(reg, weight)
        print(f"\n-- Weight={weight} | {gate_stats(gate)} --")
        evaluate_rolling(
            reg,
            X_context_proc=eval_context_proc,
            y_context=eval_y_context,
            text_context=eval_text_context,
            X_eval_proc=X_eval_range,
            y_eval=y_eval_range,
            text_eval=text_eval_range,
            use_text=True,
            label=f"Eval (text weight={weight})",
            max_context=MAX_CONTEXT_FOR_EVAL,
        )

    print("\n== Baseline recap ==")
    _format_metrics("Eval (no text attn)", *baseline)


if __name__ == "__main__":
    main()
