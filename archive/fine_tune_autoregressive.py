"""
Autoregressive gate fine-tuning using the same dataset config + rolling eval pipeline.

This keeps the autoregressive training style but aligns data loading, splits,
and evaluation with fine_tune_dpt.py/eval_fine_tune.py.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

try:
    import schedulefree  # type: ignore
except Exception:  # noqa: BLE001
    schedulefree = None

from eval_fine_tune import _format_metrics, evaluate_rolling
from fine_tune_dpt import (
    freeze_all_but_last_gate,
    gate_stats,
    load_dataset_config,
    load_tabdpt_regressor,
    preprocess_features,
)
from load_dataset import load_tabular_text_dataset
from split_ts import time_split


torch.manual_seed(0)
np.random.seed(0)

DATASET_CONFIG_PATH = "configs/bitcoin.yaml"
DEFAULT_DATASET = None

MAX_ROWS = 1000
CONTEXT_RATIO = 0.2
TUNE_RATIO = 0.6
EVAL_RATIO = 0.2

EPOCHS = 50
GATE_LR = 1e-1
GATE_LOGIT_CLAMP = 10.0
LOG_EVERY = 5

MODEL_WEIGHT_PATH = None
DEVICE = None

MAX_CONTEXT_FOR_EVAL = 1000


def fine_tune_gate_autoregressive(
    reg,
    *,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    text_tune: np.ndarray,
) -> None:
    gate = freeze_all_but_last_gate(reg)
    if schedulefree is None:
        print("WARNING: `schedulefree` not installed; falling back to torch.optim.AdamW.")
        optimizer = torch.optim.AdamW([gate], lr=GATE_LR, weight_decay=0.0)
    else:
        optimizer = schedulefree.AdamWScheduleFree([gate], lr=GATE_LR, weight_decay=0.0)
    if hasattr(optimizer, "train"):
        optimizer.train()

    for epoch in range(1, EPOCHS + 1):
        reg.model.train()
        X_train_tensor, X_test_tensor, y_train_tensor, text_attn_weight = reg._predict_autoregressive_fine_tune(
            X_tune, text=text_tune
        )
        y_target = torch.tensor(y_tune, dtype=torch.float32, device=reg.device)
        preds = reg.model(
            x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
            y_src=y_train_tensor.unsqueeze(-1),
            task="reg",
            text_enhanced_attn_weight=text_attn_weight,
        )
        preds = preds.squeeze(-1)
        loss = torch.nn.functional.mse_loss(preds, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if GATE_LOGIT_CLAMP is not None:
            with torch.no_grad():
                gate.clamp_(-GATE_LOGIT_CLAMP, GATE_LOGIT_CLAMP)

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | {gate_stats(gate)}")

    for p in reg.model.parameters():
        p.requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoregressive fine-tuning aligned with rolling eval."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset key in the config file.")
    parser.add_argument("--config", default=DATASET_CONFIG_PATH, help="Path to dataset config YAML.")
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

    print("\n== Baseline (before tuning) ==")
    eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
    eval_y_context = np.concatenate((y_context, y_tune))
    eval_text_context = np.concatenate((text_context, text_tune), axis=0)
    baseline_no_text = evaluate_rolling(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=eval_text_context,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=text_eval,
        use_text=False,
        label="Eval (no text attn)",
        max_context=MAX_CONTEXT_FOR_EVAL,
    )
    baseline_text = evaluate_rolling(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=eval_text_context,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=text_eval,
        use_text=True,
        label="Eval (with text attn)",
        max_context=MAX_CONTEXT_FOR_EVAL,
    )

    print("\n== Fine-tuning alpha gate (autoregressive) ==")
    fine_tune_gate_autoregressive(reg, X_tune=X_tune, y_tune=y_tune, text_tune=text_tune)

    print("\n== After tuning ==")
    tuned_no_text = evaluate_rolling(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=eval_text_context,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=text_eval,
        use_text=False,
        label="Eval (no text attn)",
        max_context=MAX_CONTEXT_FOR_EVAL,
    )
    tuned_text = evaluate_rolling(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=eval_text_context,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=text_eval,
        use_text=True,
        label="Eval (with text attn)",
        max_context=MAX_CONTEXT_FOR_EVAL,
    )

    print("\n== Summary ==")
    _format_metrics("Baseline (no text attn)", *baseline_no_text)
    _format_metrics("Baseline (with text attn)", *baseline_text)
    _format_metrics("Tuned (no text attn)", *tuned_no_text)
    _format_metrics("Tuned (with text attn)", *tuned_text)


if __name__ == "__main__":
    main()
