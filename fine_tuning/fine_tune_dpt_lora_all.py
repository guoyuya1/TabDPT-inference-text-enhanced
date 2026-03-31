"""
Config-driven LoRA fine-tuning for all TabDPT linear layers.

This script mirrors the rolling one-step tuning loop used by the repo's other
fine-tuning entry points, but instead of updating base weights directly it
injects LoRA adapters into every `nn.Linear` module in the TabDPT model and
optimizes only the LoRA parameters.

The script always runs without text attention, matching the recent no-text
fine-tuning scripts in this repo.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tabdpt import TabDPTRegressor
from tabdpt.utils import pad_x

try:
    from .eval_fine_tune import _format_metrics, evaluate_rolling
    from .fine_tune_configs import TuningConfig, load_fine_tune_config
    from .fine_tune_dpt_last_qk import (
        _build_optimizer,
        _normalized_regression_loss,
        compute_gradient_norm,
        load_tabdpt_regressor,
        preprocess_features,
    )
    from .load_dataset import load_tabular_dataset
    from .split_ts import time_split
except ImportError:
    from eval_fine_tune import _format_metrics, evaluate_rolling
    from fine_tune_configs import TuningConfig, load_fine_tune_config
    from fine_tune_dpt_last_qk import (
        _build_optimizer,
        _normalized_regression_loss,
        compute_gradient_norm,
        load_tabdpt_regressor,
        preprocess_features,
    )
    from load_dataset import load_tabular_dataset
    from split_ts import time_split


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base_layer)!r}.")

        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        param_device = base_layer.weight.device
        param_dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
                device=param_device,
                dtype=param_dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                device=param_device,
                dtype=param_dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.base_layer.weight.requires_grad_(False)
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_hidden = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_hidden, self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora_linear_layers(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    prefix: str = "",
) -> list[str]:
    replaced: list[str] = []
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                LoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )
            replaced.append(full_name)
            continue
        replaced.extend(
            inject_lora_linear_layers(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                prefix=full_name,
            )
        )
    return replaced


def iter_lora_layers(model: nn.Module) -> list[tuple[str, LoRALinear]]:
    base_model = _unwrap_model(model)
    return [
        (name, module)
        for name, module in base_model.named_modules()
        if isinstance(module, LoRALinear)
    ]


def set_lora_modules_mode(model: nn.Module, *, training: bool) -> None:
    for _, module in iter_lora_layers(model):
        module.train(training)


def freeze_all_but_lora(reg: TabDPTRegressor) -> list[torch.nn.Parameter]:
    for p in _unwrap_model(reg.model).parameters():
        p.requires_grad_(False)

    tunable_params: list[torch.nn.Parameter] = []
    for _, module in iter_lora_layers(reg.model):
        module.lora_A.requires_grad_(True)
        module.lora_B.requires_grad_(True)
        tunable_params.extend([module.lora_A, module.lora_B])
    return tunable_params


def get_training_info(reg: TabDPTRegressor) -> str:
    lora_layers = iter_lora_layers(reg.model)
    total_layers = len(lora_layers)
    total_trainable = sum(
        module.lora_A.numel() + module.lora_B.numel()
        for _, module in lora_layers
    )

    sample_summaries: list[str] = []
    for name, module in lora_layers[:6]:
        a_norm = float(module.lora_A.detach().float().norm().cpu().item())
        b_norm = float(module.lora_B.detach().float().norm().cpu().item())
        sample_summaries.append(
            f"{name}(r={module.rank}, scale={module.scaling:.3f}, A_norm={a_norm:.4f}, B_norm={b_norm:.4f})"
        )
    if total_layers > len(sample_summaries):
        sample_summaries.append(f"... +{total_layers - len(sample_summaries)} more")

    return (
        f"LoRA layers={total_layers} | trainable_params={total_trainable} | "
        + " ".join(sample_summaries)
    )


def fine_tune_lora_all(
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
    Fine-tune LoRA adapters on all TabDPT linear layers.

    This uses the same rolling one-step logic as the repo's existing tuners and
    always runs with `text_enhanced_attn_weight=None`.
    """
    tunable_params = freeze_all_but_lora(reg)
    optimizer, optimizer_name = _build_optimizer(tunable_params, lr=tuning_cfg.gate_lr)
    print(
        f"Tuning LoRA adapters on all linear layers (loss_type={tuning_cfg.loss_type}, "
        f"lr={tuning_cfg.gate_lr}, optimizer={optimizer_name})"
    )

    reg.model.eval()
    if hasattr(optimizer, "train"):
        optimizer.train()

    num_steps = int(np.ceil(len(y_tune) / tuning_cfg.tune_batch_size))
    best_loss = float("inf")
    best_mae = float("inf")
    for epoch in range(1, tuning_cfg.epochs + 1):
        epoch_separator = "=" * 24
        set_lora_modules_mode(reg.model, training=True)
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

            if (step_idx + 1) % tuning_cfg.step_log_every == 0 or step_idx == num_steps - 1:
                grad_norm = compute_gradient_norm(tunable_params)
                print(
                    f"Epoch {epoch:02d} Step {step_idx+1:03d}/{num_steps} "
                    f"| Loss: {avg_loss:.4f} | BestLoss: {best_loss:.4f} "
                    f"| MAE: {mae:.4f} | BestMAE: {best_mae:.4f} "
                    f"| RMSE: {rmse:.4f} | GradNorm: {grad_norm:.6f}"
                )

        if tuning_cfg.log_text_mixing_params:
            print(f"Epoch {epoch:02d} LoRA params | {get_training_info(reg)}")

        if tuning_cfg.eval_each_epoch:
            if X_eval_proc is None or y_eval is None:
                raise ValueError("EVAL_EACH_EPOCH requires X_eval_proc and y_eval.")
            reg.model.eval()
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

    for p in _unwrap_model(reg.model).parameters():
        p.requires_grad_(False)
    reg.model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune LoRA adapters on all TabDPT linear layers without text attention."
    )
    parser.add_argument("--dataset", help="Dataset key in the config file.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank for every wrapped linear layer.")
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA scaling alpha for every wrapped linear layer.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout probability applied on adapter inputs.",
    )
    args = parser.parse_args()

    run_cfg = load_fine_tune_config(args.config, args.dataset)
    torch.manual_seed(run_cfg.seed)
    np.random.seed(run_cfg.seed)

    if args.dataset:
        print(f"Using dataset '{args.dataset}' from {args.config}")
    else:
        print(f"Using dataset config {args.config}")
    if run_cfg.model.text_enhanced:
        print("Ignoring config model.text_enhanced=True: this script always tunes without text attention.")

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
    wrapped_modules = inject_lora_linear_layers(
        _unwrap_model(reg.model),
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    if not wrapped_modules:
        raise RuntimeError("No nn.Linear modules were found for LoRA injection.")
    print(
        f"Injected LoRA into {len(wrapped_modules)} linear layers "
        f"(rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout})."
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

    print("Initial LoRA params:", get_training_info(reg))

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

    print("\n== Fine-tuning LoRA adapters ==")
    fine_tune_lora_all(
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
    print("Tuned LoRA params:", get_training_info(reg))


if __name__ == "__main__":
    main()
