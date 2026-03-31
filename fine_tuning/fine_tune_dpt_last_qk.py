"""
Config-driven fine-tuning of TabDPT's last-layer `W_Q` and `W_K` weights.

This mirrors the rolling one-step tuning loop from `fine_tune_dpt.py`, but it
always runs without text attention. Only the final transformer's attention
projections are updated:

- `q_proj.weight` (`W_Q`)
- the key half of `kv_proj.weight` (`W_K`)

The value half of `kv_proj.weight` remains frozen via a gradient mask.

The script reuses the existing YAML schema from `fine_tune_dpt.py`. In
particular, `tuning.gate_lr` is used as the optimizer learning rate for Q/K
fine-tuning to avoid introducing a second config format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabdpt import TabDPTRegressor
from tabdpt.utils import pad_x

try:
    import schedulefree  # type: ignore
except ImportError:
    schedulefree = None

try:
    from .eval_fine_tune import _format_metrics, evaluate_rolling
    from .fine_tune_configs import TuningConfig, load_fine_tune_config
    from .load_dataset import load_tabular_dataset
    from .split_ts import time_split
except ImportError:
    from eval_fine_tune import _format_metrics, evaluate_rolling
    from fine_tune_configs import TuningConfig, load_fine_tune_config
    from load_dataset import load_tabular_dataset
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


def _normalized_regression_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mean_y: torch.Tensor,
    std_y: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    mean_y = mean_y.reshape(-1)
    std_y = std_y.reshape(-1)
    y_pred_norm = (y_pred - mean_y) / std_y
    y_true_norm = (y_true - mean_y) / std_y
    if loss_type == "l1":
        return torch.nn.functional.l1_loss(y_pred_norm, y_true_norm)
    if loss_type == "l2":
        return torch.nn.functional.mse_loss(y_pred_norm, y_true_norm)
    raise ValueError(f"Unsupported tuning loss_type: {loss_type!r}. Expected 'l1' or 'l2'.")


def compute_gradient_norm(params: list[torch.nn.Parameter]) -> float:
    total_norm = 0.0
    for p in params:
        if p.grad is not None:
            total_norm += float(torch.norm(p.grad).cpu().item()) ** 2
    return float(np.sqrt(total_norm))


def _build_optimizer(
    params: list[torch.nn.Parameter],
    *,
    lr: float,
) -> tuple[torch.optim.Optimizer, str]:
    if schedulefree is not None:
        return schedulefree.AdamWScheduleFree(params, lr=lr, weight_decay=0.0), "AdamWScheduleFree"
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0), "AdamW"


def _get_last_block(reg: TabDPTRegressor) -> torch.nn.Module:
    if not hasattr(reg.model, "transformer_encoder"):
        raise RuntimeError("Loaded model has no transformer_encoder.")
    return reg.model.transformer_encoder[-1]


def _split_last_layer_kv_weight(reg: TabDPTRegressor) -> tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]:
    last_block = _get_last_block(reg)
    if not hasattr(last_block, "kv_proj"):
        raise RuntimeError("Last transformer block has no kv_proj layer.")
    kv_weight = last_block.kv_proj.weight
    split_idx = int(last_block.embed_dim)
    return kv_weight, kv_weight[:split_idx], kv_weight[split_idx:]


def get_last_layer_qk_params(reg: TabDPTRegressor) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    last_block = _get_last_block(reg)
    if not hasattr(last_block, "q_proj"):
        raise RuntimeError("Last transformer block has no q_proj layer.")
    kv_weight, _, _ = _split_last_layer_kv_weight(reg)
    return last_block.q_proj.weight, kv_weight


def freeze_all_but_last_qk(
    reg: TabDPTRegressor,
) -> tuple[torch.nn.Parameter, torch.nn.Parameter, list[torch.nn.Parameter], torch.utils.hooks.RemovableHandle]:
    """Freeze all parameters except the last block's `W_Q` and `W_K`."""
    for p in reg.model.parameters():
        p.requires_grad_(False)

    last_block = _get_last_block(reg)
    q_weight = last_block.q_proj.weight
    kv_weight, _, _ = _split_last_layer_kv_weight(reg)
    q_weight.requires_grad_(True)
    kv_weight.requires_grad_(True)

    key_mask = torch.zeros_like(kv_weight)
    key_mask[: last_block.embed_dim] = 1.0
    hook = kv_weight.register_hook(lambda grad, mask=key_mask: grad * mask)
    return q_weight, kv_weight, [q_weight, kv_weight], hook


def _weight_stats(label: str, tensor: torch.Tensor) -> str:
    flat = tensor.detach().float().cpu().reshape(-1)
    sample_count = min(4, flat.numel())
    sample_vals = flat[:sample_count].tolist()
    sample_str = ", ".join(f"{v:.4f}" for v in sample_vals)
    if flat.numel() > sample_count:
        sample_str = f"{sample_str}, ..."
    return (
        f"{label}: mean={flat.mean().item():.4f} std={flat.std(unbiased=False).item():.4f} "
        f"min={flat.min().item():.4f} max={flat.max().item():.4f} sample=[{sample_str}]"
    )


def get_training_info(reg: TabDPTRegressor) -> str:
    """Compact one-line summary for the trainable last-layer Q/K weights."""
    last_block = _get_last_block(reg)
    q_weight = last_block.q_proj.weight
    _, k_weight, v_weight = _split_last_layer_kv_weight(reg)
    return " | ".join(
        [
            _weight_stats("W_Q", q_weight),
            _weight_stats("W_K", k_weight),
            _weight_stats("W_V(frozen)", v_weight),
        ]
    )


def fine_tune_last_qk(
    reg: TabDPTRegressor,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    X_tune_proc: np.ndarray,
    y_tune: np.ndarray,
    tuning_cfg: TuningConfig,
    X_eval_proc: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
) -> None:
    """
    Fine-tune only the last layer's regular-attention `W_Q` and `W_K`.

    This uses the same rolling one-step logic as `fine_tune_dpt.py`, except the
    model always runs with `text_enhanced_attn_weight=None`.
    """
    _, _, tunable_params, kv_hook = freeze_all_but_last_qk(reg)
    optimizer, optimizer_name = _build_optimizer(tunable_params, lr=tuning_cfg.gate_lr)
    print(
        f"Tuning last-layer W_Q/W_K (loss_type={tuning_cfg.loss_type}, "
        f"lr={tuning_cfg.gate_lr}, optimizer={optimizer_name})"
    )

    reg.model.eval()
    if hasattr(optimizer, "train"):
        optimizer.train()

    num_steps = int(np.ceil(len(y_tune) / tuning_cfg.tune_batch_size))
    best_loss = float("inf")
    best_mae = float("inf")
    try:
        for epoch in range(1, tuning_cfg.epochs + 1):
            epoch_separator = "=" * 24
            epoch_best_mae = float("inf")
            print(f"\n{epoch_separator} Epoch {epoch:02d}/{tuning_cfg.epochs:02d} {epoch_separator}")
            for step_idx in range(num_steps):
                optimizer.zero_grad()

                start = step_idx * tuning_cfg.tune_batch_size
                end = min(len(y_tune), start + tuning_cfg.tune_batch_size)
                current_batch_size = end - start

                preds_for_log: list[torch.Tensor] = []
                loss_sum = 0.0

                for point_idx in range(start, end):
                    X_point_context = np.concatenate((X_context_proc, X_tune_proc[:point_idx]))
                    y_point_context = np.concatenate((y_context, y_tune[:point_idx]))

                    if tuning_cfg.max_context_for_tune is not None:
                        X_context_step = X_point_context[-tuning_cfg.max_context_for_tune:]
                        y_context_step = y_point_context[-tuning_cfg.max_context_for_tune:]
                    else:
                        X_context_step = X_point_context
                        y_context_step = y_point_context

                    X_train_tensor = torch.tensor(
                        X_context_step,
                        dtype=torch.float32,
                        device=reg.device,
                    ).unsqueeze(0)
                    X_train_tensor = pad_x(X_train_tensor, reg.max_features)
                    X_test_tensor = torch.tensor(
                        X_tune_proc[point_idx:point_idx + 1],
                        dtype=torch.float32,
                        device=reg.device,
                    ).unsqueeze(0)
                    X_test_tensor = pad_x(X_test_tensor, reg.max_features)
                    y_context_tensor = torch.tensor(
                        y_context_step,
                        dtype=torch.float32,
                        device=reg.device,
                    ).unsqueeze(0)

                    pred_point, std_y, mean_y = reg.model(
                        x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                        y_src=y_context_tensor.unsqueeze(-1),
                        task="reg",
                        text_enhanced_attn_weight=None,
                    )
                    pred_point = pred_point.squeeze(-1).reshape(-1)
                    preds_for_log.append(pred_point.detach())

                    y_point_target = torch.tensor(
                        y_tune[point_idx:point_idx + 1],
                        dtype=torch.float32,
                        device=reg.device,
                    )
                    point_loss = _normalized_regression_loss(
                        pred_point,
                        y_point_target,
                        mean_y,
                        std_y,
                        tuning_cfg.loss_type,
                    )
                    loss_sum += float(point_loss.detach().cpu())
                    (point_loss / current_batch_size).backward()

                preds = torch.cat(preds_for_log, dim=0)
                y_target = torch.tensor(y_tune[start:end], dtype=torch.float32, device=reg.device)
                optimizer.step()
                preds_np = preds.detach().cpu().numpy()
                avg_loss = loss_sum / current_batch_size
                best_loss = min(best_loss, avg_loss)
                mse = mean_squared_error(y_target.detach().cpu().numpy(), preds_np)
                rmse = float(np.sqrt(mse))
                mae = mean_absolute_error(y_target.detach().cpu().numpy(), preds_np)
                best_mae = min(best_mae, mae)
                epoch_best_mae = min(epoch_best_mae, mae)

                if (step_idx + 1) % tuning_cfg.step_log_every == 0 or step_idx == num_steps - 1:
                    grad_norm = compute_gradient_norm(tunable_params)
                    print(
                        f"Epoch {epoch:02d} Step {step_idx+1:03d}/{num_steps} "
                        f"| Loss: {avg_loss:.4f} | BestLoss: {best_loss:.4f} "
                        f"| MAE: {mae:.4f} | BestMAE: {best_mae:.4f} "
                        f"| RMSE: {rmse:.4f} | GradNorm: {grad_norm:.6f}"
                    )

            if tuning_cfg.log_text_mixing_params:
                print(f"Epoch {epoch:02d} attention params | {get_training_info(reg)}")

            if tuning_cfg.eval_each_epoch:
                if X_eval_proc is None or y_eval is None:
                    raise ValueError("EVAL_EACH_EPOCH requires X_eval_proc and y_eval.")
                print(f"\n{'=' * 18} Eval after epoch {epoch:02d} {'=' * 18}")
                evaluate_rolling(
                    reg,
                    X_context_proc=X_context_proc,
                    y_context=y_context,
                    text_context=None,
                    X_eval_proc=X_tune_proc,
                    y_eval=y_tune,
                    text_eval=None,
                    use_text=False,
                    label="Tune (no text attn)",
                    max_context=tuning_cfg.max_context_for_tune_eval,
                )
                eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
                eval_y_context = np.concatenate((y_context, y_tune))
                evaluate_rolling(
                    reg,
                    X_context_proc=eval_context_proc,
                    y_context=eval_y_context,
                    text_context=None,
                    X_eval_proc=X_eval_proc,
                    y_eval=y_eval,
                    text_eval=None,
                    use_text=False,
                    label="Eval (no text attn)",
                    max_context=tuning_cfg.max_context_for_eval,
                )
            print(f"{'=' * 64}\n")
    finally:
        kv_hook.remove()
        for p in reg.model.parameters():
            p.requires_grad_(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune last-layer W_Q/W_K on a configured dataset without text attention."
    )
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
    if run_cfg.model.text_enhanced:
        print("Ignoring config model.text_enhanced=True: this script always tunes standard attention only.")

    X, y = load_tabular_dataset(
        path=run_cfg.data_path,
        date_column=run_cfg.date_column,
        numeric_features=run_cfg.numeric_features,
        target_column=run_cfg.target_column,
        max_rows=run_cfg.max_rows,
    )
    if X.shape[1] == 0:
        X = np.zeros((X.shape[0], 1), dtype=np.float32)

    (
        X_context,
        y_context,
        _,
        X_tune,
        y_tune,
        _,
        X_eval,
        y_eval,
        _,
    ) = time_split(
        X,
        y,
        None,
        context_ratio=run_cfg.context_ratio,
        tune_ratio=run_cfg.tune_ratio,
        eval_ratio=run_cfg.eval_ratio,
    )
    print(f"Split sizes: context={len(y_context)} tune={len(y_tune)} eval={len(y_eval)}")

    reg = load_tabdpt_regressor(
        device=run_cfg.model.device,
        model_weight_path=run_cfg.model.model_weight_path,
        text_enhanced=False,
        use_flash=run_cfg.model.use_flash,
        compile_model=run_cfg.model.compile_model,
    )
    reg.fit(X_context, y_context)

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

    print("Initial last-layer attention params:", get_training_info(reg))

    print("\n== Baseline (before tuning) ==")
    baseline_no_text_tune = evaluate_rolling(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=None,
        X_eval_proc=X_tune_proc,
        y_eval=y_tune,
        text_eval=None,
        use_text=False,
        label="Tune (no text attn)",
        max_context=run_cfg.tuning.max_context_for_tune_eval,
    )
    eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
    eval_y_context = np.concatenate((y_context, y_tune))
    baseline_no_text_eval = evaluate_rolling(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=None,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=None,
        use_text=False,
        label="Eval (no text attn)",
        max_context=run_cfg.tuning.max_context_for_eval,
    )

    print("\n== Fine-tuning W_Q/W_K ==")
    fine_tune_last_qk(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        X_tune_proc=X_tune_proc,
        y_tune=y_tune,
        tuning_cfg=run_cfg.tuning,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
    )

    print("\n== After tuning ==")
    _format_metrics("Tune (before tuning)", *baseline_no_text_tune)
    evaluate_rolling(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=None,
        X_eval_proc=X_tune_proc,
        y_eval=y_tune,
        text_eval=None,
        use_text=False,
        label="Tune (after tuning)",
        max_context=run_cfg.tuning.max_context_for_tune_eval,
    )
    _format_metrics("Eval (before tuning)", *baseline_no_text_eval)
    evaluate_rolling(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=None,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=None,
        use_text=False,
        label="Eval (after tuning)",
        max_context=run_cfg.tuning.max_context_for_eval,
    )
    print("Tuned last-layer attention params:", get_training_info(reg))


if __name__ == "__main__":
    main()
