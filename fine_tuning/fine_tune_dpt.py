"""
Config-driven fine-tuning of TabDPT's text-mixing parameters in the final transformer layers.

This script expects a processed CSV with numeric features, a numeric target, and
text embedding columns. It loads one YAML config, splits the data
chronologically into context / train / val / test, fits the base TabDPT
regressor on the context split, and then fine-tunes only the text-mixing
parameters in the text-enhanced transformer layers:

- the per-head `alpha` gate logits
- the text attention projection used to score train/test text pairs

Fine-tuning uses rolling one-step prediction on the train split, selects the
best epoch by validation MAE with early stopping, keeps the top 3 validation-MAE
checkpoints, restores the best text-mixing weights, and prints final held-out
test metrics.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass

import numpy as np
import schedulefree  # type: ignore
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tabdpt import TabDPTRegressor
from tabdpt.utils import pad_x

try:
    from .eval_fine_tune import (
        _format_metrics,
        evaluate_rolling,
        evaluate_rolling_pca,
        evaluate_rolling_truncate_text,
    )
    from .fine_tune_configs import TuningConfig, load_fine_tune_config
    from .load_dataset import load_tabular_text_dataset
    from .split_ts import time_split
except ImportError:
    from eval_fine_tune import (
        _format_metrics,
        evaluate_rolling,
        evaluate_rolling_pca,
        evaluate_rolling_truncate_text,
    )
    from fine_tune_configs import TuningConfig, load_fine_tune_config
    from load_dataset import load_tabular_text_dataset
    from split_ts import time_split


TOP_VALIDATION_MAE_MODELS = 3


@dataclass(frozen=True)
class ValidationMaeCheckpoint:
    epoch: int
    val_mae: float
    state_dict: dict[str, torch.Tensor]


def _clone_state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[name] = value.detach().cpu().clone()
        else:
            cloned[name] = copy.deepcopy(value)
    return cloned


def _maybe_record_top_validation_mae_checkpoint(
    checkpoints: list[ValidationMaeCheckpoint],
    *,
    epoch: int,
    val_mae: float,
    model: torch.nn.Module,
    limit: int = TOP_VALIDATION_MAE_MODELS,
) -> None:
    candidate_key = (val_mae, epoch)
    if len(checkpoints) >= limit:
        worst_checkpoint = max(checkpoints, key=lambda checkpoint: (checkpoint.val_mae, checkpoint.epoch))
        if candidate_key >= (worst_checkpoint.val_mae, worst_checkpoint.epoch):
            return

    checkpoints.append(
        ValidationMaeCheckpoint(
            epoch=epoch,
            val_mae=val_mae,
            state_dict=_clone_state_dict_to_cpu(model.state_dict()),
        )
    )
    checkpoints.sort(key=lambda checkpoint: (checkpoint.val_mae, checkpoint.epoch))
    del checkpoints[limit:]


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


def _layer_label(layer_idx: int) -> str:
    return f"L{layer_idx + 1}"


def _get_text_enhanced_blocks(
    reg: TabDPTRegressor,
) -> list[tuple[int, torch.nn.Module]]:
    text_blocks: list[tuple[int, torch.nn.Module]] = []
    for layer_idx, block in enumerate(reg.model.transformer_encoder):
        has_gate = getattr(block, "alpha", None) is not None
        if not getattr(block, "text_enhanced", False) and not has_gate:
            continue
        if not has_gate:
            raise RuntimeError(
                f"{_layer_label(layer_idx)} is marked as text-enhanced but has no alpha gate."
            )
        text_blocks.append((layer_idx, block))
    if not text_blocks:
        raise RuntimeError("Model has no text-enhanced transformer blocks. Is text_enhanced enabled?")
    return text_blocks


def get_text_enhanced_gate_params(
    reg: TabDPTRegressor,
) -> list[tuple[str, torch.nn.Parameter]]:
    """Return the trainable gate tensors for every text-enhanced transformer block."""
    return [(_layer_label(layer_idx), block.alpha) for layer_idx, block in _get_text_enhanced_blocks(reg)]


def get_last_layer_gate_param(reg: TabDPTRegressor) -> torch.nn.Parameter:
    """Backward-compatible wrapper that returns the final text-enhanced block's gate tensor."""
    return get_text_enhanced_gate_params(reg)[-1][1]


def _get_text_mixing_modules_for_block(
    block: torch.nn.Module,
    *,
    layer_label: str,
) -> list[tuple[str, torch.nn.Module]]:
    """
    Return a block's text-mixing modules.

    The current branch uses per-head text projections/norms stored in
    `ModuleList`s, but we keep fallbacks for older experimental layouts to make
    local iteration less brittle.
    """
    if getattr(block, "text_head_projs", None) is not None:
        text_modules: list[tuple[str, torch.nn.Module]] = [("head_proj", block.text_head_projs)]
        if getattr(block, "text_head_q_norms", None) is not None:
            text_modules.append(("text_q_norm", block.text_head_q_norms))
        if getattr(block, "text_head_k_norms", None) is not None:
            text_modules.append(("text_k_norm", block.text_head_k_norms))
        return text_modules
    if getattr(block, "text_shared_proj", None) is not None:
        text_modules = [("shared_proj", block.text_shared_proj)]
        if getattr(block, "text_q_norm", None) is not None:
            text_modules.append(("text_q_norm", block.text_q_norm))
        if getattr(block, "text_k_norm", None) is not None:
            text_modules.append(("text_k_norm", block.text_k_norm))
        return text_modules
    if getattr(block, "text_attn_linears", None) is not None:
        return [(f"text_attn_{idx}", module) for idx, module in enumerate(block.text_attn_linears)]
    raise RuntimeError(
        f"{layer_label} has no recognized text-mixing module. Expected `text_head_projs` "
        "for the current architecture, `text_shared_proj` for the prior one, or "
        "`text_attn_linears` for the legacy one."
    )


def _get_text_enhanced_block_specs(
    reg: TabDPTRegressor,
) -> list[tuple[str, torch.nn.Module, list[tuple[str, torch.nn.Module]]]]:
    return [
        (_layer_label(layer_idx), block, _get_text_mixing_modules_for_block(block, layer_label=_layer_label(layer_idx)))
        for layer_idx, block in _get_text_enhanced_blocks(reg)
    ]


def _freeze_all_but_text_mixing(
    reg: TabDPTRegressor,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Freeze the model except the text-enhanced blocks' gate and text-attention parameters."""
    for p in reg.model.parameters():
        p.requires_grad_(False)

    gate_params: list[torch.nn.Parameter] = []
    text_attn_params: list[torch.nn.Parameter] = []
    seen_param_ids: set[int] = set()
    for _, gate in get_text_enhanced_gate_params(reg):
        if id(gate) in seen_param_ids:
            continue
        gate.requires_grad_(True)
        gate_params.append(gate)
        seen_param_ids.add(id(gate))

    for _, _, text_modules in _get_text_enhanced_block_specs(reg):
        for _, module in text_modules:
            for param in module.parameters():
                if id(param) in seen_param_ids:
                    continue
                param.requires_grad_(True)
                text_attn_params.append(param)
                seen_param_ids.add(id(param))
    return gate_params, text_attn_params


def freeze_all_but_last_text_mixing(
    reg: TabDPTRegressor,
) -> tuple[torch.nn.Parameter, list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Backward-compatible wrapper that now unfreezes every text-enhanced block."""
    gate_params, text_attn_params = _freeze_all_but_text_mixing(reg)
    return gate_params[-1], gate_params, text_attn_params


def freeze_all_but_last_gate(reg: TabDPTRegressor) -> tuple[torch.nn.Parameter, list[torch.nn.Parameter]]:
    """Backward-compatible wrapper that returns the combined gate + text-mixing params."""
    gate, gate_params, text_attn_params = freeze_all_but_last_text_mixing(reg)
    return gate, gate_params + text_attn_params


def build_text_mixing_optimizer(
    *,
    gate_params: list[torch.nn.Parameter],
    text_attn_params: list[torch.nn.Parameter],
    tuning_cfg: TuningConfig,
) -> schedulefree.AdamWScheduleFree:
    """Build one optimizer with separate LR groups for gate and text-attention params."""
    if not gate_params:
        raise ValueError("Expected at least one gate parameter for text-mixing fine-tuning.")
    if not text_attn_params:
        raise ValueError("Expected at least one text-attention parameter for text-mixing fine-tuning.")

    gate_param_ids = {id(param) for param in gate_params}
    text_attn_param_ids = {id(param) for param in text_attn_params}
    if gate_param_ids & text_attn_param_ids:
        raise ValueError("Gate and text-attention parameter groups must be disjoint.")

    param_groups = [
        {
            "params": gate_params,
            "lr": tuning_cfg.gate_lr,
            "weight_decay": 0.0,
        },
        {
            "params": text_attn_params,
            "lr": tuning_cfg.text_attn_lr,
            "weight_decay": 0.0,
        },
    ]
    return schedulefree.AdamWScheduleFree(
        param_groups,
        lr=tuning_cfg.gate_lr,
        weight_decay=0.0,
    )


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


def text_mixing_gate_stats(reg: TabDPTRegressor) -> str:
    """Human-readable summary for the gates in every text-enhanced transformer block."""
    return " || ".join(
        f"{layer_label} gate: {gate_stats(gate_logits)}"
        for layer_label, gate_logits in get_text_enhanced_gate_params(reg)
    )


def _text_module_param_summary(label: str, module: torch.nn.Module) -> str:
    submodules = list(module) if isinstance(module, torch.nn.ModuleList) else [module]
    resolved_modules: list[torch.nn.Module] = []
    for submodule in submodules:
        weighted_module = submodule
        if not hasattr(weighted_module, "weight"):
            weighted_module = next((m for m in submodule.modules() if isinstance(m, torch.nn.Linear)), None)
            if weighted_module is None:
                continue
        resolved_modules.append(weighted_module)
    if not resolved_modules:
        return f"{label}(no-linear)"

    if len(resolved_modules) == 1:
        weighted_module = resolved_modules[0]
        weight = weighted_module.weight.detach().float().cpu().reshape(-1)
        w_val = weight.item() if weight.numel() == 1 else weight.mean().item()
        w_norm = weight.norm().item()
        line = f"{label}(||W||={w_norm:.3f}, mean={w_val:.6f}"
        if weighted_module.bias is not None:
            bias = weighted_module.bias.detach().float().cpu().reshape(-1)
            b_val = bias.item() if bias.numel() == 1 else bias.mean().item()
            line += f", b={b_val:.3f}"
        line += ")"
        return line

    weight_norms = torch.tensor(
        [weighted_module.weight.detach().float().cpu().reshape(-1).norm().item() for weighted_module in resolved_modules]
    )
    weight_means = torch.tensor(
        [weighted_module.weight.detach().float().cpu().reshape(-1).mean().item() for weighted_module in resolved_modules]
    )
    line = (
        f"{label}(heads={len(resolved_modules)}, "
        f"||W|| mean/min/max={weight_norms.mean().item():.3f}/"
        f"{weight_norms.min().item():.3f}/{weight_norms.max().item():.3f}, "
        f"mean(mean)={weight_means.mean().item():.6f}"
    )
    bias_means = [
        weighted_module.bias.detach().float().cpu().reshape(-1).mean().item()
        for weighted_module in resolved_modules
        if weighted_module.bias is not None
    ]
    if bias_means:
        line += f", b_mean={float(np.mean(bias_means)):.3f}"
    line += ")"
    return line


def get_trainining_info(reg: TabDPTRegressor, gate_logits: torch.Tensor | None = None) -> str:
    """Compact one-line summary for each text-enhanced block's gate and text-mixing params."""
    del gate_logits
    layer_summaries: list[str] = []
    for layer_label, block, text_modules in _get_text_enhanced_block_specs(reg):
        module_summaries = [_text_module_param_summary(module_label, module) for module_label, module in text_modules]
        layer_summaries.append(
            f"{layer_label} gate: {gate_stats(block.alpha)} | {' '.join(module_summaries)}"
        )
    return " || ".join(layer_summaries)


def format_text_score_info(
    reg: TabDPTRegressor,
    *,
    text_train: torch.Tensor,
    text_test: torch.Tensor,
    sample_size: int = 8,
) -> str:
    """
    Summarize the raw text score logits produced by the learned projection path.

    The current text-enhanced model computes scores with each text-enhanced
    block's `_text_attention_logits(...)`, which returns raw pre-softmax logits
    with shape (B, H, N_test, N_train).
    """
    try:
        text_blocks = _get_text_enhanced_blocks(reg)
    except RuntimeError:
        return "text_score(unavailable)"

    summaries: list[str] = []
    for layer_idx, block in text_blocks:
        if getattr(block, "_text_attention_logits", None) is None:
            summaries.append(f"{_layer_label(layer_idx)} text_score(unavailable)")
            continue

        with torch.no_grad():
            logits = block._text_attention_logits(text_train, text_test).detach().float().cpu()

        flat = logits.reshape(-1)
        head0_query = logits[0, 0, 0].reshape(-1)
        sample_count = min(sample_size, head0_query.numel())
        sample_vals = ", ".join(f"{v:.4f}" for v in head0_query[:sample_count].tolist())
        if head0_query.numel() > sample_count:
            sample_vals = f"{sample_vals}, ..."

        summaries.append(
            f"{_layer_label(layer_idx)} text_score(shape={tuple(logits.shape)}, "
            f"mean/min/max={flat.mean().item():.4f}/{flat.min().item():.4f}/{flat.max().item():.4f}, "
            f"head0_sample=[{sample_vals}])"
        )

    return " || ".join(summaries)


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


def _early_stopping_score(
    *,
    metric_name: str,
    mae: float,
    rmse: float,
    mape: float,
) -> float:
    metric_name = metric_name.lower()
    if metric_name == "mae":
        return mae
    if metric_name == "rmse":
        return rmse
    if metric_name == "mape":
        return mape
    raise ValueError(
        f"Unsupported early_stopping_metric: {metric_name!r}. Expected one of 'mae', 'rmse', or 'mape'."
    )


def _evaluate_rolling_loss_and_mae(
    reg: TabDPTRegressor,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray | None,
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    text_eval: np.ndarray | None,
    use_text: bool,
    max_context: int | None,
    loss_type: str,
) -> tuple[float, float, float, float]:
    """Run rolling evaluation and return average normalized loss plus metrics."""
    if use_text and (text_context is None or text_eval is None):
        raise ValueError("Rolling eval with text requires context and eval text arrays.")

    reg.model.eval()
    preds = np.zeros(len(y_eval), dtype=np.float32)
    loss_sum = 0.0
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
                text_train_b, text_test_b = reg.text_embeddings_batched(
                    train_text_batch,
                    test_text_batch,
                )
            else:
                text_train_b = text_test_b = None

            X_train_tensor = torch.tensor(X_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_train_tensor = pad_x(X_train_tensor, reg.max_features)
            X_test_tensor = torch.tensor(X_test_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_test_tensor = pad_x(X_test_tensor, reg.max_features)
            y_context_tensor = torch.tensor(y_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

            pred, std_y, mean_y = reg.model(
                x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                y_src=y_context_tensor.unsqueeze(-1),
                task="reg",
                text_train=text_train_b,
                text_test=text_test_b,
            )
            pred = pred.squeeze(-1).reshape(-1)
            preds[idx] = pred.detach().cpu().numpy()[0]

            y_target = torch.tensor(
                y_eval[idx:idx + 1],
                dtype=torch.float32,
                device=reg.device,
            )
            point_loss = _normalized_regression_loss(
                pred,
                y_target,
                mean_y,
                std_y,
                loss_type,
            )
            loss_sum += float(point_loss.detach().cpu())

    mse = mean_squared_error(y_eval, preds)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_eval, preds)
    denom = np.clip(np.abs(y_eval), 1e-8, None)
    mape = float(np.mean(np.abs((y_eval - preds) / denom)) * 100.0)
    avg_loss = loss_sum / len(y_eval)
    return avg_loss, mae, rmse, mape


def _print_epoch_section(
    *,
    epoch: int,
    train_loss: float,
    train_mae: float,
    val_loss: float,
    val_mae: float,
    param_stats: str,
) -> None:
    print(f"\n== Epoch {epoch:02d} ==")
    print(f"Train  | Loss: {train_loss:.4f} | MAE: {train_mae:.4f}")
    print(f"Val    | Loss: {val_loss:.4f} | MAE: {val_mae:.4f}")
    print(f"Params | {param_stats}")


def fine_tune_external_gate(
    reg: TabDPTRegressor,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray,
    X_train_proc: np.ndarray,
    y_train: np.ndarray,
    text_train: np.ndarray,
    tuning_cfg: TuningConfig,
    X_val_proc: np.ndarray,
    y_val: np.ndarray,
    text_val: np.ndarray,
) -> list[ValidationMaeCheckpoint]:
    """
    Fine-tune only the text-enhanced blocks' text-mixing parameters.

    Why call the model directly?
    - `reg.predict()` disables gradients, so we can't optimize with it.
    - We still re-use everything from `reg.fit()`:
        - fitted imputers/scalers
        - batching helpers that shape text inputs for the text-enhanced blocks

    Rolling window (per step inside each epoch):
    - Fixed context = global context + train rows before this batch
    - Within the batch, each row is predicted from base context plus earlier batch rows
    - Gradients are accumulated point-by-point and averaged over the batch
    """
    if tuning_cfg.early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive.")

    _, gate_params, text_attn_params = freeze_all_but_last_text_mixing(reg)
    text_block_labels = ", ".join(layer_label for layer_label, _ in get_text_enhanced_gate_params(reg))
    optimizer = build_text_mixing_optimizer(
        gate_params=gate_params,
        text_attn_params=text_attn_params,
        tuning_cfg=tuning_cfg,
    )
    print(
        f"Tuning text-mixing params on blocks [{text_block_labels}] "
        f"(loss_type={tuning_cfg.loss_type}, gate_lr={tuning_cfg.gate_lr}, "
        f"text_attn_lr={tuning_cfg.text_attn_lr}, optimizer={optimizer.__class__.__name__}, "
        f"early_stopping_metric={tuning_cfg.early_stopping_metric}, "
        f"patience={tuning_cfg.early_stopping_patience})"
    )

    reg.model.eval()
    if hasattr(optimizer, "train"):
        optimizer.train()

    num_steps = int(np.ceil(len(y_train) / tuning_cfg.tune_batch_size))
    best_score = float("inf")
    best_epoch = 0
    best_state_dict: dict[str, torch.Tensor] | None = None
    top_validation_mae_checkpoints: list[ValidationMaeCheckpoint] = []
    epochs_without_improvement = 0
    val_context_proc = np.concatenate((X_context_proc, X_train_proc))
    val_y_context = np.concatenate((y_context, y_train))
    val_text_context = np.concatenate((text_context, text_train), axis=0)

    for epoch in range(1, tuning_cfg.epochs + 1):
        for step_idx in range(num_steps):
            optimizer.zero_grad()

            start = step_idx * tuning_cfg.tune_batch_size
            end = min(len(y_train), start + tuning_cfg.tune_batch_size)

            current_batch_size = end - start

            for point_idx in range(start, end):
                X_point_context = np.concatenate((X_context_proc, X_train_proc[:point_idx]))
                y_point_context = np.concatenate((y_context, y_train[:point_idx]))
                text_point_context = np.concatenate((text_context, text_train[:point_idx]), axis=0)
                # When max_context is set, each training step only sees the most recent
                # rows from the accumulated context so training matches a fixed rolling window.
                if tuning_cfg.max_context is not None:
                    X_context_step = X_point_context[-tuning_cfg.max_context:]
                    y_context_step = y_point_context[-tuning_cfg.max_context:]
                    text_context_step = text_point_context[-tuning_cfg.max_context:]
                # Otherwise, each step uses the full available history: the original context
                # plus every earlier train row before the current prediction target.
                else:
                    X_context_step = X_point_context
                    y_context_step = y_point_context
                    text_context_step = text_point_context

                train_text_batch = text_context_step[None, ...]
                test_text_batch = text_train[point_idx:point_idx + 1][None, ...]
                text_train_b, text_test_b = reg.text_embeddings_batched(
                    train_text_batch,
                    test_text_batch,
                )

                X_train_tensor = torch.tensor(X_context_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
                X_train_tensor = pad_x(X_train_tensor, reg.max_features)
                X_test_tensor = torch.tensor(
                    X_train_proc[point_idx:point_idx + 1],
                    dtype=torch.float32,
                    device=reg.device,
                ).unsqueeze(0)
                X_test_tensor = pad_x(X_test_tensor, reg.max_features)
                y_context_tensor = torch.tensor(y_context_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

                pred_point, std_y, mean_y = reg.model(
                    x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                    y_src=y_context_tensor.unsqueeze(-1),
                    task="reg",
                    text_train=text_train_b,
                    text_test=text_test_b,
                )
                pred_point = pred_point.squeeze(-1).reshape(-1)

                y_point_target = torch.tensor(
                    y_train[point_idx:point_idx + 1],
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
                (point_loss / current_batch_size).backward()

            optimizer.step()

            if tuning_cfg.gate_logit_clamp is not None:
                with torch.no_grad():
                    for gate in gate_params:
                        gate.clamp_(-tuning_cfg.gate_logit_clamp, tuning_cfg.gate_logit_clamp)

        train_loss, train_mae, _, _ = _evaluate_rolling_loss_and_mae(
            reg,
            X_context_proc=X_context_proc,
            y_context=y_context,
            text_context=text_context,
            X_eval_proc=X_train_proc,
            y_eval=y_train,
            text_eval=text_train,
            use_text=True,
            max_context=tuning_cfg.max_context,
            loss_type=tuning_cfg.loss_type,
        )
        val_loss, val_mae, val_rmse, val_mape = _evaluate_rolling_loss_and_mae(
            reg,
            X_context_proc=val_context_proc,
            y_context=val_y_context,
            text_context=val_text_context,
            X_eval_proc=X_val_proc,
            y_eval=y_val,
            text_eval=text_val,
            use_text=True,
            max_context=tuning_cfg.max_context,
            loss_type=tuning_cfg.loss_type,
        )
        param_stats = get_trainining_info(reg)
        _print_epoch_section(
            epoch=epoch,
            train_loss=train_loss,
            train_mae=train_mae,
            val_loss=val_loss,
            val_mae=val_mae,
            param_stats=param_stats,
        )
        _maybe_record_top_validation_mae_checkpoint(
            top_validation_mae_checkpoints,
            epoch=epoch,
            val_mae=val_mae,
            model=reg.model,
        )
        current_score = _early_stopping_score(
            metric_name=tuning_cfg.early_stopping_metric,
            mae=val_mae,
            rmse=val_rmse,
            mape=val_mape,
        )
        if current_score < best_score:
            best_score = current_score
            best_epoch = epoch
            best_state_dict = _clone_state_dict_to_cpu(reg.model.state_dict())
            epochs_without_improvement = 0
            print(
                f"New best validation {tuning_cfg.early_stopping_metric.upper()}: "
                f"{current_score:.4f} at epoch {epoch:02d}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement for {epochs_without_improvement} epoch(s) "
                f"(best epoch {best_epoch:02d}, best {tuning_cfg.early_stopping_metric.upper()}={best_score:.4f})"
            )
            if epochs_without_improvement >= tuning_cfg.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch:02d}")
                break

    if best_state_dict is None:
        raise RuntimeError("Early stopping did not record a best model state.")

    reg.model.load_state_dict(best_state_dict)
    print(
        f"Restored best text-mixing params from epoch {best_epoch:02d} "
        f"(validation {tuning_cfg.early_stopping_metric.upper()}={best_score:.4f})"
    )
    print(
        "Top validation-MAE checkpoints: "
        + ", ".join(
            f"#{rank} epoch {checkpoint.epoch:02d} (MAE={checkpoint.val_mae:.4f})"
            for rank, checkpoint in enumerate(top_validation_mae_checkpoints, start=1)
        )
    )

    for p in reg.model.parameters():
        p.requires_grad_(False)

    return top_validation_mae_checkpoints



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune text-mixing parameters on the final text-enhanced transformer layers."
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
        X_train,
        y_train,
        text_train,
        X_val,
        y_val,
        text_val,
        X_test,
        y_test,
        text_test,
    ) = time_split(
        X,
        y,
        text,
        context_ratio=run_cfg.context_ratio,
        train_ratio=run_cfg.train_ratio,
        val_ratio=run_cfg.val_ratio,
        test_ratio=run_cfg.test_ratio,
    )
    print(
        "Split sizes: "
        f"context={len(y_context)} train={len(y_train)} val={len(y_val)} test={len(y_test)}"
    )

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
    X_train_proc = preprocess_features(
        reg,
        X_train,
        reduction_mode=reduction_mode,
        reduction_payload=reduction_payload,
    )
    X_val_proc = preprocess_features(
        reg,
        X_val,
        reduction_mode=reduction_mode,
        reduction_payload=reduction_payload,
    )
    X_test_proc = preprocess_features(
        reg,
        X_test,
        reduction_mode=reduction_mode,
        reduction_payload=reduction_payload,
    )

    print("\n== Baseline (before tuning) ==")
    baseline_no_text_train = evaluate_rolling(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_train_proc,
        y_eval=y_train,
        text_eval=text_train,
        use_text=False,
        label="Train (no text attn)",
        max_context=run_cfg.tuning.max_context,
    )
    baseline_pca_train = evaluate_rolling_pca(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_train_proc,
        y_eval=y_train,
        text_eval=text_train,
        label="Train (PCA)",
        max_context=run_cfg.tuning.max_context,
    )
    baseline_truncate_train = evaluate_rolling_truncate_text(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_train_proc,
        y_eval=y_train,
        text_eval=text_train,
        label="Train (text truncate)",
        max_context=run_cfg.tuning.max_context,
    )
    val_context_proc = np.concatenate((X_context_proc, X_train_proc))
    val_y_context = np.concatenate((y_context, y_train))
    val_text_context = np.concatenate((text_context, text_train), axis=0)
    baseline_no_text_val = evaluate_rolling(
        reg,
        X_context_proc=val_context_proc,
        y_context=val_y_context,
        text_context=val_text_context,
        X_eval_proc=X_val_proc,
        y_eval=y_val,
        text_eval=text_val,
        use_text=False,
        label="Val (no text attn)",
        max_context=run_cfg.tuning.max_context,
    )
    baseline_text_val = evaluate_rolling(
        reg,
        X_context_proc=val_context_proc,
        y_context=val_y_context,
        text_context=val_text_context,
        X_eval_proc=X_val_proc,
        y_eval=y_val,
        text_eval=text_val,
        use_text=True,
        label="Val (with text attn)",
        max_context=run_cfg.tuning.max_context,
    )
    baseline_pca_val = evaluate_rolling_pca(
        reg,
        X_context_proc=val_context_proc,
        y_context=val_y_context,
        text_context=val_text_context,
        X_eval_proc=X_val_proc,
        y_eval=y_val,
        text_eval=text_val,
        label="Val (PCA)",
        max_context=run_cfg.tuning.max_context,
    )
    baseline_truncate_val = evaluate_rolling_truncate_text(
        reg,
        X_context_proc=val_context_proc,
        y_context=val_y_context,
        text_context=val_text_context,
        X_eval_proc=X_val_proc,
        y_eval=y_val,
        text_eval=text_val,
        label="Val (text truncate)",
        max_context=run_cfg.tuning.max_context,
    )
    if run_cfg.tuning.debug_text_effect:
        delta_mae = baseline_text_val[0] - baseline_no_text_val[0]
        delta_rmse = baseline_text_val[1] - baseline_no_text_val[1]
        delta_mape = baseline_text_val[2] - baseline_no_text_val[2]
        print(
            f"Val text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
        print(
            "Initial text-mixing params:",
            get_trainining_info(reg),
        )
    else:
        print("Initial text-mixing gates:", text_mixing_gate_stats(reg))

    print("\n== Fine-tuning text-mixing parameters ==")
    top_validation_mae_checkpoints = fine_tune_external_gate(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_train_proc=X_train_proc,
        y_train=y_train,
        text_train=text_train,
        tuning_cfg=run_cfg.tuning,
        X_val_proc=X_val_proc,
        y_val=y_val,
        text_val=text_val,
    )

    print("\n== After tuning ==")
    _format_metrics("Train (no text attn)", *baseline_no_text_train)
    _format_metrics("Train (PCA)", *baseline_pca_train)
    if baseline_truncate_train is not None:
        _format_metrics(baseline_truncate_train[1], *baseline_truncate_train[0])
    evaluate_rolling(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_train_proc,
        y_eval=y_train,
        text_eval=text_train,
        use_text=True,
        label="Train (with text attn)",
        max_context=run_cfg.tuning.max_context,
    )

    test_context_proc = np.concatenate((X_context_proc, X_train_proc, X_val_proc))
    test_y_context = np.concatenate((y_context, y_train, y_val))
    test_text_context = np.concatenate((text_context, text_train, text_val), axis=0)
    tuned_no_text_test = evaluate_rolling(
        reg,
        X_context_proc=test_context_proc,
        y_context=test_y_context,
        text_context=test_text_context,
        X_eval_proc=X_test_proc,
        y_eval=y_test,
        text_eval=text_test,
        use_text=False,
        label="Test (no text attn)",
        max_context=run_cfg.tuning.max_context,
    )
    evaluate_rolling_pca(
        reg,
        X_context_proc=test_context_proc,
        y_context=test_y_context,
        text_context=test_text_context,
        X_eval_proc=X_test_proc,
        y_eval=y_test,
        text_eval=text_test,
        label="Test (PCA)",
        max_context=run_cfg.tuning.max_context,
    )
    evaluate_rolling_truncate_text(
        reg,
        X_context_proc=test_context_proc,
        y_context=test_y_context,
        text_context=test_text_context,
        X_eval_proc=X_test_proc,
        y_eval=y_test,
        text_eval=text_test,
        label="Test (text truncate)",
        max_context=run_cfg.tuning.max_context,
    )
    tuned_text_test = evaluate_rolling(
        reg,
        X_context_proc=test_context_proc,
        y_context=test_y_context,
        text_context=test_text_context,
        X_eval_proc=X_test_proc,
        y_eval=y_test,
        text_eval=text_test,
        use_text=True,
        label="Test (with text attn)",
        max_context=run_cfg.tuning.max_context,
    )
    if run_cfg.tuning.debug_text_effect:
        delta_mae = tuned_text_test[0] - tuned_no_text_test[0]
        delta_rmse = tuned_text_test[1] - tuned_no_text_test[1]
        delta_mape = tuned_text_test[2] - tuned_no_text_test[2]
        print(
            f"Test text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
        print(
            "Tuned text-mixing params:",
            get_trainining_info(reg),
        )
    else:
        print("Tuned text-mixing gates:", text_mixing_gate_stats(reg))

    restored_best_state_dict = _clone_state_dict_to_cpu(reg.model.state_dict())
    print(f"\n== Top {len(top_validation_mae_checkpoints)} Validation-MAE Models On Test ==")
    for rank, checkpoint in enumerate(top_validation_mae_checkpoints, start=1):
        reg.model.load_state_dict(checkpoint.state_dict)
        print(f"Rank {rank} | Epoch {checkpoint.epoch:02d} | Val MAE: {checkpoint.val_mae:.4f}")
        evaluate_rolling(
            reg,
            X_context_proc=test_context_proc,
            y_context=test_y_context,
            text_context=test_text_context,
            X_eval_proc=X_test_proc,
            y_eval=y_test,
            text_eval=text_test,
            use_text=True,
            label=f"Test top {rank} (with text attn)",
            max_context=run_cfg.tuning.max_context,
        )
    reg.model.load_state_dict(restored_best_state_dict)


if __name__ == "__main__":
    main()
