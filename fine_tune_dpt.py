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

import numpy as np
import schedulefree  # type: ignore
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fine_tune_configs import TuningConfig, load_fine_tune_config
from tabdpt import TabDPTRegressor
from tabdpt.utils import pad_x

from eval_fine_tune import _format_metrics, evaluate_rolling
from load_dataset import load_tabular_text_dataset
from split_ts import time_split


def load_tabdpt_regressor(
    *,
    device: str | None,
    model_weight_path: str | None,
    text_enhanced: bool,
    use_flash: bool,
    compile_model: bool,
) -> TabDPTRegressor:
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabDPTRegressor(
        device=device,
        text_enhanced=text_enhanced,
        model_weight_path=model_weight_path,
        use_flash=use_flash,
        compile=compile_model,
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


def get_trainining_info(reg: TabDPTRegressor, gate_logits: torch.Tensor) -> str:
    """Compact one-line summary for the gate and last block text linear params."""
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
    return f"gate: {gate_stats(gate_logits)} | {' '.join(summaries)}"


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
    X_tune_proc: np.ndarray,
    y_tune: np.ndarray,
    text_tune: np.ndarray,
    tuning_cfg: TuningConfig,
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
    - Fixed context = global context + tune rows before this batch
    - Within the batch, each row is predicted from base context plus earlier batch rows
    - Gradients are accumulated point-by-point and averaged over the batch
    """
    gate, tunable_params = freeze_all_but_last_gate(reg)

    optimizer = schedulefree.AdamWScheduleFree(tunable_params, lr=tuning_cfg.gate_lr, weight_decay=0.0)
    print("Tuning last-layer text-mixing params")

    reg.model.eval()
    if hasattr(optimizer, "train"):
        optimizer.train()

    num_steps = int(np.ceil(len(y_tune) / tuning_cfg.tune_batch_size))
    for epoch in range(1, tuning_cfg.epochs + 1):
        for step_idx in range(num_steps):
            optimizer.zero_grad()

            start = step_idx * tuning_cfg.tune_batch_size
            end = min(len(y_tune), start + tuning_cfg.tune_batch_size)

            current_batch_size = end - start
            preds_for_log: list[torch.Tensor] = []

            for point_idx in range(start, end):
                X_point_context = np.concatenate((X_context_proc, X_tune_proc[:point_idx]))
                y_point_context = np.concatenate((y_context, y_tune[:point_idx]))
                text_point_context = np.concatenate((text_context, text_tune[:point_idx]), axis=0)
                # When max_context_for_tune is set, each tuning step only sees the most recent
                # rows from the accumulated context so training matches a fixed rolling window.
                if tuning_cfg.max_context_for_tune is not None:
                    X_context_step = X_point_context[-tuning_cfg.max_context_for_tune:]
                    y_context_step = y_point_context[-tuning_cfg.max_context_for_tune:]
                    text_context_step = text_point_context[-tuning_cfg.max_context_for_tune:]
                # Otherwise, each step uses the full available history: the original context
                # plus every earlier tune row before the current prediction target.
                else:
                    X_context_step = X_point_context
                    y_context_step = y_point_context
                    text_context_step = text_point_context

                train_text_batch = text_context_step[None, ...]
                test_text_batch = text_tune[point_idx:point_idx + 1][None, ...]
                attn_weight_external = reg._compute_attn_weight_pairwise_avg(train_text_batch, test_text_batch)

                X_train_tensor = torch.tensor(X_context_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
                X_train_tensor = pad_x(X_train_tensor, reg.max_features)
                X_test_tensor = torch.tensor(
                    X_tune_proc[point_idx:point_idx + 1],
                    dtype=torch.float32,
                    device=reg.device,
                ).unsqueeze(0)
                X_test_tensor = pad_x(X_test_tensor, reg.max_features)
                y_context_tensor = torch.tensor(y_context_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

                pred_point = reg.model(
                    x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                    y_src=y_context_tensor.unsqueeze(-1),
                    task="reg",
                    text_enhanced_attn_weight=attn_weight_external,
                ).squeeze(-1).reshape(-1)
                preds_for_log.append(pred_point.detach())

                y_point_target = torch.tensor(
                    y_tune[point_idx:point_idx + 1],
                    dtype=torch.float32,
                    device=reg.device,
                )
                point_loss = torch.nn.functional.mse_loss(pred_point, y_point_target)
                (point_loss / current_batch_size).backward()

            preds = torch.cat(preds_for_log, dim=0)
            y_target = torch.tensor(y_tune[start:end], dtype=torch.float32, device=reg.device)
            optimizer.step()

            if tuning_cfg.gate_logit_clamp is not None:
                with torch.no_grad():
                    gate.clamp_(-tuning_cfg.gate_logit_clamp, tuning_cfg.gate_logit_clamp)

            # print if meeting log frequency or last step in epoch
            if (step_idx + 1) % tuning_cfg.step_log_every == 0 or step_idx == num_steps - 1:
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
            f"{get_trainining_info(reg, gate)}"
        )
        if tuning_cfg.eval_each_epoch:
            if X_eval_proc is None or y_eval is None or text_eval is None:
                raise ValueError("EVAL_EACH_EPOCH requires X_eval_proc, y_eval, and text_eval.")
            print(f"\n== Eval after epoch {epoch:02d} ==")
            # Rolling tune-split evaluation: predict tune points with text attention using only pre-tune context.
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
                max_context=tuning_cfg.max_context_for_tune_eval,
            )
            # Build eval-time context by appending the full tune segment, so eval predictions only use past rows.
            eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
            eval_y_context = np.concatenate((y_context, y_tune))
            eval_text_context = np.concatenate((text_context, text_tune), axis=0)
            # Rolling eval-split evaluation: predict held-out eval points with text attention from context+tune history.
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
                max_context=tuning_cfg.max_context_for_eval,
            )

    for p in reg.model.parameters():
        p.requires_grad_(False)



def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune alpha gate on a configured dataset.")
    parser.add_argument("--dataset", help="Dataset key in the config file.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    args = parser.parse_args()

    run_cfg = load_fine_tune_config(args.config, args.dataset)
    torch.manual_seed(run_cfg.seed)
    np.random.seed(run_cfg.seed)

    if args.dataset:
        print(f"Using dataset '{args.dataset}' from {args.config}")
    else:
        print(f"Using dataset config {args.config}")

    X, y, text = load_tabular_text_dataset(
        path=run_cfg.data_path,
        date_column=run_cfg.date_column,
        numeric_features=run_cfg.numeric_features,
        target_column=run_cfg.target_column,
        embedding_lags=run_cfg.embedding_lags,
        embedding_columns=run_cfg.embedding_columns,
        embedding_column_template=run_cfg.embedding_column_template,
        max_rows=run_cfg.max_rows,
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
        context_ratio=run_cfg.context_ratio,
        tune_ratio=run_cfg.tune_ratio,
        eval_ratio=run_cfg.eval_ratio,
    )
    print(f"Split sizes: context={len(y_context)} tune={len(y_tune)} eval={len(y_eval)}")

    reg = load_tabdpt_regressor(
        device=run_cfg.model.device,
        model_weight_path=run_cfg.model.model_weight_path,
        text_enhanced=run_cfg.model.text_enhanced,
        use_flash=run_cfg.model.use_flash,
        compile_model=run_cfg.model.compile_model,
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
        max_context=run_cfg.tuning.max_context_for_tune_eval,
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
        max_context=run_cfg.tuning.max_context_for_eval,
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
        max_context=run_cfg.tuning.max_context_for_eval,
    )
    if run_cfg.tuning.debug_text_effect:
        delta_mae = baseline_text_eval[0] - baseline_no_text_eval[0]
        delta_rmse = baseline_text_eval[1] - baseline_no_text_eval[1]
        delta_mape = baseline_text_eval[2] - baseline_no_text_eval[2]
        print(
            f"Eval text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
    print("Initial text-mixing gate:", gate_stats(get_last_layer_gate_param(reg)))

    print("\n== Fine-tuning text-mixing parameters ==")
    fine_tune_external_gate(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_tune_proc=X_tune_proc,
        y_tune=y_tune,
        text_tune=text_tune,
        tuning_cfg=run_cfg.tuning,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=text_eval,
    )

    print("\n== After tuning ==")
    _format_metrics("Tune (no text attn)", *baseline_no_text_tune)
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
        max_context=run_cfg.tuning.max_context_for_tune_eval,
    )
    _format_metrics("Eval (no text attn)", *baseline_no_text_eval)
    tuned_no_text_eval = baseline_no_text_eval
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
        max_context=run_cfg.tuning.max_context_for_eval,
    )
    if run_cfg.tuning.debug_text_effect:
        delta_mae = tuned_text_eval[0] - tuned_no_text_eval[0]
        delta_rmse = tuned_text_eval[1] - tuned_no_text_eval[1]
        delta_mape = tuned_text_eval[2] - tuned_no_text_eval[2]
        print(
            f"Eval text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
    print("Tuned text-mixing gate:", gate_stats(get_last_layer_gate_param(reg)))


if __name__ == "__main__":
    main()
