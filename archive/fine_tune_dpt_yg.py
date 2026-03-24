"""
Fine-tune the model's trainable `alpha` (text-attention blending gate).

The goal: only update the gate parameters that control how much the model trusts
external text attention when predicting *test rows from train context*.

This script is written in a step-by-step style. Each step is explained in code
and comments so you can port it to a notebook cell-by-cell.

What the gate is in THIS repo
-----------------------------
In `tabdpt/model.py`, the transformer stack is set up as:
- layers 0..(L-2): normal attention
- last layer: `text_enhanced=True` and has `alpha` (per-head logits)

During the last layer's datapoint attention:
- train rows attend to train rows (no external attention)
- test rows attend to train rows, optionally mixing in text similarity attention
  computed from text embeddings

The gate parameter stored on the last block is a vector of logits with shape (H,),
one per attention head. The forward pass applies `sigmoid()` to obtain values in
(0, 1), which are then used as a blending coefficient.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    import schedulefree  # type: ignore
except Exception:  # noqa: BLE001
    schedulefree = None

from tabdpt import TabDPTRegressor
from tabdpt.utils import pad_x
try:
    from omegaconf import OmegaConf
except ImportError:
    import yaml

    class _OmegaConfCompat:
        @staticmethod
        def load(path):
            with open(Path(path), "r", encoding="utf-8") as f:
                return yaml.safe_load(f)

        @staticmethod
        def to_container(value, resolve: bool = True):  # noqa: ARG001
            return value

    OmegaConf = _OmegaConfCompat

from fine_tuning.eval_fine_tune import _format_metrics, evaluate_rolling
from fine_tuning.load_dataset import load_tabular_text_dataset
from fine_tuning.split_ts import time_split


# -----------------------------
# Step 0) Reproducibility
# -----------------------------
torch.manual_seed(0)
np.random.seed(0)


# -----------------------------
# Step 1) Configuration
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

# Text lags to use (your regressor averages similarity across the L dimension).
TEXT_EMBEDDING_LAGS = [0, 1, 2, 3]
# Either provide an explicit list of embedding columns, or a template for lags.
# If EMBEDDING_COLUMNS is not None, it takes precedence.
EMBEDDING_COLUMNS = None
EMBEDDING_COLUMN_TEMPLATE = f"embedding_{TEXT_FEATURE_NAME}_lag{{lag}}"

# Dataset config file (YAML) for a single dataset.
DATASET_CONFIG_PATH = "configs/bitcoin.yaml"
DEFAULT_DATASET = None

# Use chronological splits to avoid leakage in time series.
# Optional cap to limit the dataset size.
MAX_ROWS = 1000
# Fraction-based split: three ratios that must sum to 1.0.
CONTEXT_RATIO = 0.2
TUNE_RATIO = 0.65
EVAL_RATIO = 0.15
# How many new tune rows to add per batch step.
TUNE_BATCH_SIZE = 650
# Optional: cap the training context length during tuning to avoid OOM.
# If None, use all available context; if an int, keep only the last K rows of
# [global_context + past_tune] when building each training window.
MAX_CONTEXT_FOR_TUNE = 1000

# Model loading / inference options.
# Set MODEL_WEIGHT_PATH to a local .safetensors file to avoid HF downloads.
MODEL_WEIGHT_PATH = None
DEVICE = None  # e.g., "cuda:0" or "cpu"
USE_FLASH = True
COMPILE_MODEL = True
# Fine-tuning hyperparameters (we tune only a few numbers: per-head gate logits).
EPOCHS = 20
LEARNING_RATE = 1e-4  # kept for backward compatibility; see GATE_LR below
LOG_EVERY = 5  # epoch-level logging cadence
STEP_LOG_EVERY = 50  # step-level logging inside each epoch

# Diagnostics: compare eval metrics with/without text attention.
DEBUG_TEXT_EFFECT = True

# Use a higher LR for the gate to encourage it to move away from ~0.5 if helpful.
GATE_LR = 5e-1

# Optional: clamp gate *logits* to avoid extreme saturation of sigmoid.
GATE_LOGIT_CLAMP = 10.0

# Optional: encourage gate probabilities away from 0.5 (toward extremes).
# Minimizing gate*(1-gate) pushes sigmoids toward 0 or 1. Set to 0.0 to disable.
GATE_REG_STRENGTH = 0

# Reuse the no-text baseline metrics after tuning to avoid nondeterministic drift.
FREEZE_NO_TEXT_BASELINE = True

# Run a full eval (no-text + text) after each epoch during fine-tuning.
EVAL_EACH_EPOCH = True

# Optional context cap during rolling eval (None uses full context).
MAX_CONTEXT_FOR_EVAL = MAX_CONTEXT_FOR_TUNE
# Optional context cap during rolling tuning-set eval (None uses full context).
MAX_CONTEXT_FOR_TUNE_EVAL = MAX_CONTEXT_FOR_TUNE


# -----------------------------
# Step 2) Data loading utilities
# -----------------------------

def load_dataset_config(config_path: str, dataset_name: str | None) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config not found: {config_path}")
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Dataset config must be a mapping: {config_path}")

    if dataset_name:
        if dataset_name not in config:
            if "data_path" in config:
                raise ValueError(
                    f"Config {config_path} looks like a single dataset; omit --dataset."
                )
            available = ", ".join(sorted(config.keys()))
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
        dataset_cfg = config[dataset_name]
    else:
        dataset_cfg = config
        if "data_path" not in dataset_cfg:
            raise ValueError(
                f"Config {config_path} contains multiple datasets; pass --dataset."
            )

    required = ["data_path", "numeric_features", "target_column", "embedding_columns"]
    missing = [key for key in required if key not in dataset_cfg]
    if missing:
        label = dataset_name or os.path.basename(config_path)
        raise ValueError(f"Dataset '{label}' missing keys: {', '.join(missing)}")
    if not isinstance(dataset_cfg["numeric_features"], list):
        raise ValueError(f"Dataset '{dataset_name}' numeric_features must be a list.")
    if not isinstance(dataset_cfg["embedding_columns"], list):
        raise ValueError(f"Dataset '{dataset_name}' embedding_columns must be a list.")
    return dataset_cfg

def load_tabdpt_regressor(
    *,
    device: str | None,
    model_weight_path: str | None,
    text_enhanced: bool = True,
) -> TabDPTRegressor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return TabDPTRegressor(
        device=device,
        text_enhanced=text_enhanced,
        model_weight_path=model_weight_path,
        use_flash=USE_FLASH,
        compile=COMPILE_MODEL,
    )


def preprocess_features(
    reg: TabDPTRegressor,
    X: np.ndarray,
    *,
    reduction_mode: str | None,
    reduction_payload: np.ndarray | None,
) -> np.ndarray:
    X_proc = X
    if reg.missing_indicators:
        inds = np.isnan(X_proc)[:, reg.has_missing_indicator].astype(float)
        X_proc = np.hstack((X_proc, inds))
    X_proc = reg.imputer.transform(X_proc)
    if reg.scaler:
        X_proc = reg.scaler.transform(X_proc)
        if reg.normalizer == "quantile-uniform":
            X_proc = 2 * X_proc - 1

    if reduction_mode == "pca":
        if reduction_payload is None:
            raise ValueError("PCA reduction requested without a payload.")
        X_proc = X_proc @ reduction_payload
    elif reduction_mode == "subsample":
        if reduction_payload is None:
            raise ValueError("Subsample reduction requested without a payload.")
        X_proc = X_proc[:, reduction_payload][:, :reg.max_features]

    return X_proc.astype(np.float32)


# Evaluation helpers live in eval_fine_tune.py.
# -----------------------------
# Step 5) Gate-only parameter selection
# -----------------------------
def get_last_layer_gate_param(reg: TabDPTRegressor) -> torch.nn.Parameter:
    """
    Return the trainable gate parameter (per-head logits) from the last transformer block.

    Only the last transformer block is `text_enhanced=True` in this repo.
    """
    last_block = reg.model.transformer_encoder[-1]
    if getattr(last_block, "alpha", None) is None:
        raise RuntimeError("Last block has no alpha gate. Is text_enhanced enabled in the model?")
    return last_block.alpha


def freeze_all_but_last_gate(reg: TabDPTRegressor) -> tuple[torch.nn.Parameter, list[torch.nn.Parameter]]:
    """Freeze all parameters except the last block's alpha gate + per-head text linear layers."""
    for p in reg.model.parameters():
        p.requires_grad_(False)
    last_block = reg.model.transformer_encoder[-1]
    gate = get_last_layer_gate_param(reg)
    gate.requires_grad_(True)
    tunable_params: list[torch.nn.Parameter] = [gate]
    for linear in last_block.text_attn_linears:
        for p in linear.parameters():
            p.requires_grad_(True)
            tunable_params.append(p)
    return gate, tunable_params


def gate_stats(gate_logits: torch.Tensor) -> str:
    """Human-readable summary for per-head gate logits and their sigmoid values."""
    logits = gate_logits.detach().float().cpu().reshape(-1)
    gate = torch.sigmoid(logits)
    sample_count = min(6, gate.numel())
    sample_vals = gate[:sample_count].tolist()
    sample_str = ", ".join(f"{v:.3f}" for v in sample_vals)
    if gate.numel() > sample_count:
        sample_str = f"{sample_str}, ..."
    return (
        f"logits(mean/min/max)={logits.mean().item():.4f}/{logits.min().item():.4f}/{logits.max().item():.4f} | "
        f"sigmoid(mean/min/max)={gate.mean().item():.4f}/{gate.min().item():.4f}/{gate.max().item():.4f} | "
        f"sigmoid(sample)=[{sample_str}]"
    )


def linear_stats(reg: TabDPTRegressor) -> str:
    """Compact one-line summary for last block text linear layer parameters."""
    last_block = reg.model.transformer_encoder[-1]
    summaries: list[str] = []
    for idx, proj in enumerate(last_block.text_attn_linears):
        # Support both plain nn.Linear and nn.Sequential(Linear, GELU, ...).
        linear = proj
        if not hasattr(linear, "weight"):
            linear = next((m for m in proj.modules() if isinstance(m, torch.nn.Linear)), None)
            if linear is None:
                summaries.append(f"L{idx}(no-linear)")
                continue

        weight = linear.weight.detach().float().cpu().reshape(-1)
        w_val = weight.item() if weight.numel() == 1 else weight.mean().item()
        line = f"L{idx}(w={w_val:.3f}"
        if linear.bias is not None:
            bias = linear.bias.detach().float().cpu().reshape(-1)
            b_val = bias.item() if bias.numel() == 1 else bias.mean().item()
            line += f", b={b_val:.3f}"
        line += ")"
        summaries.append(line)
    return " ".join(summaries)


# -----------------------------
# Step 6) Fine-tuning loop (calls reg.model directly)
# Rolling window: at step i, train on context + tune[0:i], predict tune[i].
# -----------------------------
def fine_tune_external_gate(
    reg: TabDPTRegressor,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray,
    X_tune: np.ndarray,
    X_tune_proc: np.ndarray,
    y_tune: np.ndarray,
    text_tune: np.ndarray,
    X_eval: np.ndarray | None = None,
    X_eval_proc: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    text_eval: np.ndarray | None = None,
) -> None:
    """
    Fine-tune only the last layer's text mixing parameters (alpha gate + per-head text affine).

    Why call the model directly?
    - `reg.predict()` disables gradients, so we can't optimize with it.
    - We still re-use everything from `reg.fit()`:
        - fitted imputers/scalers
        - text similarity attention via `_compute_attn_weight_pairwise_avg(...)`

    Rolling window (per step inside each epoch):
    - Train set = context + all prior tune rows
    - Test row = current tune row
    - Loss = MSE on that current row (raw space)
    """
    # (1) Select tunable params (gate only) and build optimizer.
    gate, tunable_params = freeze_all_but_last_gate(reg)
    # Use Schedule-Free AdamW when available; fall back to AdamW otherwise.
    if schedulefree is None:
        print("WARNING: `schedulefree` not installed; falling back to torch.optim.AdamW.")
        optimizer = torch.optim.AdamW(tunable_params, lr=GATE_LR, weight_decay=0.0)
    else:
        # optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=0.0)
        optimizer = schedulefree.AdamWScheduleFree(tunable_params, lr=GATE_LR, weight_decay=0.0)
    print("Tuning last-layer text-mixing params")

    # (3) Rolling-window gradient loop (with base context then growing window)
    reg.model.eval()
    if hasattr(optimizer, "train"):
        optimizer.train()

    num_steps = int(np.ceil(len(y_tune) / TUNE_BATCH_SIZE))
    for epoch in range(1, EPOCHS + 1):
        for step_idx in range(num_steps):
            optimizer.zero_grad()

            start = step_idx * TUNE_BATCH_SIZE
            end = min(len(y_tune), start + TUNE_BATCH_SIZE)

            # Base context portion inside the tune set + accumulated history.
            base_cap = MAX_CONTEXT_FOR_TUNE if MAX_CONTEXT_FOR_TUNE is not None else len(y_tune)
            base_limit = min(base_cap, len(y_tune))
            past_limit = max(base_limit, start)

            # Train set for this step: global context + tune[0:past_limit]
            X_train_full = np.concatenate((X_context_proc, X_tune_proc[:past_limit]))
            y_train_full = np.concatenate((y_context, y_tune[:past_limit]))
            text_train_full = np.concatenate((text_context, text_tune[:past_limit]), axis=0)

            # Apply optional context cap to avoid long sequences (OOM protection).
            if MAX_CONTEXT_FOR_TUNE is not None:
                X_train_step = X_train_full[-MAX_CONTEXT_FOR_TUNE:]
                y_train_step = y_train_full[-MAX_CONTEXT_FOR_TUNE:]
                text_train_step = text_train_full[-MAX_CONTEXT_FOR_TUNE:]
            else:
                X_train_step = X_train_full
                y_train_step = y_train_full
                text_train_step = text_train_full

            # Test batch for this step: tune[start:end]
            X_test_step = X_tune_proc[start:end]
            y_target = torch.tensor(y_tune[start:end], dtype=torch.float32, device=reg.device)

            train_text_batch = text_train_step[None, ...]
            test_text_batch = text_tune[start:end][None, ...]
            attn_weight_external = reg._compute_attn_weight_pairwise_avg(train_text_batch, test_text_batch)

            X_train_tensor = torch.tensor(X_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_train_tensor = pad_x(X_train_tensor, reg.max_features)
            X_test_tensor = torch.tensor(X_test_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_test_tensor = pad_x(X_test_tensor, reg.max_features)
            y_context_tensor = torch.tensor(y_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

            preds = reg.model(
                x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                y_src=y_context_tensor.unsqueeze(-1),
                task="reg",
                text_enhanced_attn_weight=attn_weight_external,
            )
            preds = preds.squeeze(-1).reshape(-1)

            loss = torch.nn.functional.mse_loss(preds, y_target)
            if GATE_REG_STRENGTH and GATE_REG_STRENGTH > 0:
                gate_prob = torch.sigmoid(gate)
                gate_reg = (gate_prob * (1 - gate_prob)).mean()
                loss = loss + GATE_REG_STRENGTH * gate_reg

            loss.backward()
            optimizer.step()

            # Optional stability clamp in logit space (NOT in [0,1] space).
            if GATE_LOGIT_CLAMP is not None:
                with torch.no_grad():
                    gate.clamp_(-GATE_LOGIT_CLAMP, GATE_LOGIT_CLAMP)

            if (step_idx + 1) % STEP_LOG_EVERY == 0 or (step_idx == num_steps - 1):
                preds_np = preds.detach().cpu().numpy()
                mse = mean_squared_error(y_target.detach().cpu().numpy(), preds_np)
                rmse = float(np.sqrt(mse))
                mae = mean_absolute_error(y_target.detach().cpu().numpy(), preds_np)
                print(
                    f"Epoch {epoch:02d} Step {step_idx+1:03d}/{num_steps} "
                    f"| MAE: {mae:.4f} | RMSE: {rmse:.4f}"
                )
        print(
            f"Epoch {epoch:02d} text-mixing params | "
            f"gate: {gate_stats(gate)} | {linear_stats(reg)}"
        )
        if EVAL_EACH_EPOCH:
            if X_eval is None or X_eval_proc is None or y_eval is None or text_eval is None:
                raise ValueError("EVAL_EACH_EPOCH requires X_eval, X_eval_proc, y_eval, and text_eval.")
            print(f"\n== Eval after epoch {epoch:02d} ==")
            evaluate_rolling(
                reg,
                X_context_proc=X_context_proc,
                y_context=y_context,
                text_context=text_context,
                X_eval_proc=X_tune_proc,
                y_eval=y_tune,
                text_eval=text_tune,
                use_text=False,
                label="Tune (no text attn)",
                max_context=MAX_CONTEXT_FOR_TUNE_EVAL,
            )
            evaluate_rolling(
                reg,
                X_context_proc=X_context_proc,
                y_context=y_context,
                text_context=text_context,
                X_eval_proc=X_tune_proc,
                y_eval=y_tune,
                text_eval=text_tune,
                use_text=True,
                label="Tune (with text attn)",
                max_context=MAX_CONTEXT_FOR_TUNE_EVAL,
            )
            eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
            eval_y_context = np.concatenate((y_context, y_tune))
            eval_text_context = np.concatenate((text_context, text_tune), axis=0)
            evaluate_rolling(
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
            evaluate_rolling(
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

    # (4) Freeze again to avoid surprises if you re-use `reg` later.
    for p in reg.model.parameters():
        p.requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune alpha gate on a configured dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset key in the config file.")
    parser.add_argument("--config", default=DATASET_CONFIG_PATH, help="Path to dataset config YAML.")
    args = parser.parse_args()

    dataset_cfg = load_dataset_config(args.config, args.dataset)
    data_path = dataset_cfg.get("data_path", DATA_PATH)
    date_column = dataset_cfg.get("date_column", DATE_COLUMN)
    numeric_features = dataset_cfg.get("numeric_features", NUMERIC_FEATURES)
    target_column = dataset_cfg.get("target_column", TARGET_COLUMN)
    embedding_columns = dataset_cfg.get("embedding_columns", EMBEDDING_COLUMNS)
    embedding_lags = dataset_cfg.get("embedding_lags", TEXT_EMBEDDING_LAGS)
    embedding_column_template = dataset_cfg.get("embedding_column_template", EMBEDDING_COLUMN_TEMPLATE)

    if args.dataset:
        print(f"Using dataset '{args.dataset}' from {args.config}")
    else:
        print(f"Using dataset config {args.config}")

    # Step 1: load the dataset
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
        # TabDPT expects at least one numeric feature; use a constant placeholder.
        X = np.zeros((X.shape[0], 1), dtype=np.float32)

    # Step 2: split chronologically
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

    # Step 3: initialize regressor + store context set
    reg = load_tabdpt_regressor(device=DEVICE, model_weight_path=MODEL_WEIGHT_PATH, text_enhanced=True)
    reg.fit(X_context, y_context, text_context)

    # Feature reduction disabled; features are assumed to fit reg.max_features.
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

    # Step 4: baseline eval
    print("\n== Baseline (before tuning) ==")
    baseline_no_text_tune = evaluate_rolling(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_tune_proc,
        y_eval=y_tune,
        text_eval=text_tune,
        use_text=False,
        label="Tune (no text attn)",
        max_context=MAX_CONTEXT_FOR_TUNE_EVAL,
    )
    baseline_text_tune = evaluate_rolling(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_tune_proc,
        y_eval=y_tune,
        text_eval=text_tune,
        use_text=True,
        label="Tune (with text attn)",
        max_context=MAX_CONTEXT_FOR_TUNE_EVAL,
    )
    eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
    eval_y_context = np.concatenate((y_context, y_tune))
    eval_text_context = np.concatenate((text_context, text_tune), axis=0)
    baseline_no_text_eval = evaluate_rolling(
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
    baseline_text_eval = evaluate_rolling(
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
    if DEBUG_TEXT_EFFECT:
        delta_mae = baseline_text_eval[0] - baseline_no_text_eval[0]
        delta_rmse = baseline_text_eval[1] - baseline_no_text_eval[1]
        delta_mape = baseline_text_eval[2] - baseline_no_text_eval[2]
        print(
            f"Eval text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
    print("Initial text-mixing gate:", gate_stats(get_last_layer_gate_param(reg)))

    # Step 5: fine-tune gate on tune split
    print("\n== Fine-tuning text-mixing parameters ==")
    fine_tune_external_gate(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_tune=X_tune,
        X_tune_proc=X_tune_proc,
        y_tune=y_tune,
        text_tune=text_tune,
        X_eval=X_eval,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=text_eval,
    )

    # Step 6: eval after tuning
    print("\n== After tuning ==")
    if FREEZE_NO_TEXT_BASELINE:
        _format_metrics("Tune (no text attn)", *baseline_no_text_tune)
        _format_metrics("Eval (no text attn)", *baseline_no_text_eval)
        tuned_no_text_eval = baseline_no_text_eval
    else:
        evaluate_rolling(
            reg,
            X_context_proc=X_context_proc,
            y_context=y_context,
            text_context=text_context,
            X_eval_proc=X_tune_proc,
            y_eval=y_tune,
            text_eval=text_tune,
            use_text=False,
            label="Tune (no text attn)",
            max_context=MAX_CONTEXT_FOR_TUNE_EVAL,
        )
        tuned_no_text_eval = evaluate_rolling(
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
    evaluate_rolling(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_tune_proc,
        y_eval=y_tune,
        text_eval=text_tune,
        use_text=True,
        label="Tune (with text attn)",
        max_context=MAX_CONTEXT_FOR_TUNE_EVAL,
    )
    tuned_text_eval = evaluate_rolling(
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
    if DEBUG_TEXT_EFFECT:
        delta_mae = tuned_text_eval[0] - tuned_no_text_eval[0]
        delta_rmse = tuned_text_eval[1] - tuned_no_text_eval[1]
        delta_mape = tuned_text_eval[2] - tuned_no_text_eval[2]
        print(
            f"Eval text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
    print("Tuned text-mixing gate:", gate_stats(get_last_layer_gate_param(reg)))

    print("\n== Baseline (before tuning) ==")
    _format_metrics("Tune (no text attn)", *baseline_no_text_tune)
    _format_metrics("Eval (no text attn)", *baseline_no_text_eval)
    _format_metrics("Tune (with text attn)", *baseline_text_tune)
    _format_metrics("Eval (with text attn)", *baseline_text_eval)


if __name__ == "__main__":
    main()
