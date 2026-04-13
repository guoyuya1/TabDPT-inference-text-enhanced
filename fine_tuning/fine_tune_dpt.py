"""
Config-driven fine-tuning of TabDPT's text-mixing parameters in configured transformer layers.

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
        _build_rolling_train_step,
        _format_metrics,
        evaluate_rolling,
        evaluate_rolling_pca,
        evaluate_rolling_truncate_text,
    )
    from .fine_tune_configs import DataConfig, TuningConfig, load_fine_tune_config
    from .load_dataset import build_direct_multi_horizon_dataset, load_tabular_text_dataset
    from .split_ts import time_split
except ImportError:
    from eval_fine_tune import (
        _build_rolling_train_step,
        _format_metrics,
        evaluate_rolling,
        evaluate_rolling_pca,
        evaluate_rolling_truncate_text,
    )
    from fine_tune_configs import DataConfig, TuningConfig, load_fine_tune_config
    from load_dataset import build_direct_multi_horizon_dataset, load_tabular_text_dataset
    from split_ts import time_split


TOP_VALIDATION_MAE_MODELS = 3


@dataclass(frozen=True)
class ValidationMaeCheckpoint:
    epoch: int
    val_mae: float
    state_dict: dict[str, torch.Tensor]


@dataclass(frozen=True)
class FineTuneDataSplits:
    X_context: np.ndarray
    y_context: np.ndarray
    text_context: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    text_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    text_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    text_test: np.ndarray


@dataclass(frozen=True)
class PreparedFineTuneTrial:
    reg: TabDPTRegressor
    splits: FineTuneDataSplits
    X_context_proc: np.ndarray
    X_train_proc: np.ndarray
    X_val_proc: np.ndarray
    X_test_proc: np.ndarray
    prediction_horizon: int


@dataclass(frozen=True)
class RollingMetrics:
    loss: float
    mae: float
    rmse: float
    mape: float


@dataclass(frozen=True)
class FineTuneOutcome:
    top_validation_mae_checkpoints: list[ValidationMaeCheckpoint]
    best_epoch: int
    best_score: float


@dataclass(frozen=True)
class HorizonRunResult:
    horizon: int
    best_epoch: int
    tuned_no_text_test: tuple[float, float, float]
    tuned_text_test: tuple[float, float, float]


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
    text_attn_layers: list[int],
    use_flash: bool,
    compile_model: bool,
) -> TabDPTRegressor:
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabDPTRegressor(
        device=device,
        text_attn_layers=text_attn_layers,
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


def set_random_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_fine_tune_arrays(run_cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return X, y, text


def prediction_horizons(run_cfg: DataConfig) -> range:
    return range(1, run_cfg.prediction_window + 1)


def split_fine_tune_data(
    run_cfg: DataConfig,
    *,
    X: np.ndarray,
    y: np.ndarray,
    text: np.ndarray,
) -> FineTuneDataSplits:
    return FineTuneDataSplits(
        *time_split(
            X,
            y,
            text,
            context_ratio=run_cfg.context_ratio,
            train_ratio=run_cfg.train_ratio,
            val_ratio=run_cfg.val_ratio,
            test_ratio=run_cfg.test_ratio,
        )
    )


def load_and_split_fine_tune_data(run_cfg: DataConfig) -> FineTuneDataSplits:
    X, y, text = load_fine_tune_arrays(run_cfg)
    if run_cfg.prediction_window > 1:
        X, y, text = build_direct_multi_horizon_dataset(
            X,
            y,
            text,
            prediction_window=run_cfg.prediction_window,
        )
    return split_fine_tune_data(run_cfg, X=X, y=y, text=text)


def select_fine_tune_splits_for_horizon(
    data_splits: FineTuneDataSplits,
    *,
    horizon: int,
) -> FineTuneDataSplits:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    if data_splits.y_context.ndim == 1:
        if horizon != 1:
            raise ValueError(
                f"Requested horizon {horizon}, but the loaded splits are single-horizon."
            )
        return data_splits

    prediction_window = int(data_splits.y_context.shape[1])
    if horizon > prediction_window:
        raise ValueError(
            f"Requested horizon {horizon}, but only {prediction_window} horizon(s) are available."
        )

    horizon_idx = horizon - 1
    return FineTuneDataSplits(
        X_context=data_splits.X_context,
        y_context=data_splits.y_context[:, horizon_idx].astype(np.float32, copy=False),
        text_context=data_splits.text_context,
        X_train=data_splits.X_train,
        y_train=data_splits.y_train[:, horizon_idx].astype(np.float32, copy=False),
        text_train=data_splits.text_train,
        X_val=data_splits.X_val,
        y_val=data_splits.y_val[:, horizon_idx].astype(np.float32, copy=False),
        text_val=data_splits.text_val,
        X_test=data_splits.X_test,
        y_test=data_splits.y_test[:, horizon_idx].astype(np.float32, copy=False),
        text_test=data_splits.text_test,
    )


def prepare_fine_tune_trial(
    run_cfg: DataConfig,
    *,
    data_splits: FineTuneDataSplits | None = None,
    horizon: int = 1,
) -> PreparedFineTuneTrial:
    if data_splits is None:
        data_splits = load_and_split_fine_tune_data(run_cfg)
    selected_splits = select_fine_tune_splits_for_horizon(data_splits, horizon=horizon)

    reg = load_tabdpt_regressor(
        device=run_cfg.model.device,
        model_weight_path=run_cfg.model.model_weight_path,
        text_attn_layers=run_cfg.model.text_attn_layers,
        use_flash=run_cfg.model.use_flash,
        compile_model=run_cfg.model.compile_model,
    )
    reg.fit(selected_splits.X_context, selected_splits.y_context, selected_splits.text_context)

    reduction_mode = None
    reduction_payload = None
    return PreparedFineTuneTrial(
        reg=reg,
        splits=selected_splits,
        X_context_proc=preprocess_features(
            reg,
            selected_splits.X_context,
            reduction_mode=reduction_mode,
            reduction_payload=reduction_payload,
        ),
        X_train_proc=preprocess_features(
            reg,
            selected_splits.X_train,
            reduction_mode=reduction_mode,
            reduction_payload=reduction_payload,
        ),
        X_val_proc=preprocess_features(
            reg,
            selected_splits.X_val,
            reduction_mode=reduction_mode,
            reduction_payload=reduction_payload,
        ),
        X_test_proc=preprocess_features(
            reg,
            selected_splits.X_test,
            reduction_mode=reduction_mode,
            reduction_payload=reduction_payload,
        ),
        prediction_horizon=horizon,
    )


def _prepared_eval_inputs(
    prepared_trial: PreparedFineTuneTrial,
    *,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    splits = prepared_trial.splits
    if split_name == "train":
        return (
            prepared_trial.X_context_proc,
            splits.y_context,
            splits.text_context,
            prepared_trial.X_train_proc,
            splits.y_train,
            splits.text_train,
        )
    if split_name == "val":
        return (
            np.concatenate((prepared_trial.X_context_proc, prepared_trial.X_train_proc)),
            np.concatenate((splits.y_context, splits.y_train)),
            np.concatenate((splits.text_context, splits.text_train), axis=0),
            prepared_trial.X_val_proc,
            splits.y_val,
            splits.text_val,
        )
    if split_name == "test":
        return (
            np.concatenate((prepared_trial.X_context_proc, prepared_trial.X_train_proc, prepared_trial.X_val_proc)),
            np.concatenate((splits.y_context, splits.y_train, splits.y_val)),
            np.concatenate((splits.text_context, splits.text_train, splits.text_val), axis=0),
            prepared_trial.X_test_proc,
            splits.y_test,
            splits.text_test,
        )
    raise ValueError(f"Unsupported split_name: {split_name!r}. Expected 'train', 'val', or 'test'.")


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
    horizon: int = 1,
) -> tuple[float, float, float, float]:
    """Run rolling evaluation and return average normalized loss plus metrics."""
    if use_text and (text_context is None or text_eval is None):
        raise ValueError("Rolling eval with text requires context and eval text arrays.")

    reg.model.eval()
    preds = np.zeros(len(y_eval), dtype=np.float32)
    loss_sum = 0.0
    with torch.no_grad():
        for idx in range(len(y_eval)):
            X_train_step, y_train_step, text_train_step = _build_rolling_train_step(
                X_context_proc=X_context_proc,
                y_context=y_context,
                text_context=text_context if use_text else None,
                X_eval_proc=X_eval_proc,
                y_eval=y_eval,
                text_eval=text_eval if use_text else None,
                idx=idx,
                horizon=horizon,
                max_context=max_context,
            )
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


def evaluate_prepared_split(
    prepared_trial: PreparedFineTuneTrial,
    *,
    split_name: str,
    use_text: bool,
    tuning_cfg: TuningConfig,
) -> RollingMetrics:
    X_context_proc, y_context, text_context, X_eval_proc, y_eval, text_eval = _prepared_eval_inputs(
        prepared_trial,
        split_name=split_name,
    )
    loss, mae, rmse, mape = _evaluate_rolling_loss_and_mae(
        prepared_trial.reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=text_eval,
        use_text=use_text,
        max_context=tuning_cfg.max_context,
        loss_type=tuning_cfg.loss_type,
        horizon=prepared_trial.prediction_horizon,
    )
    return RollingMetrics(loss=loss, mae=mae, rmse=rmse, mape=mape)


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


def fine_tune_prepared_trial(
    prepared_trial: PreparedFineTuneTrial,
    *,
    tuning_cfg: TuningConfig,
) -> FineTuneOutcome:
    splits = prepared_trial.splits
    return fine_tune_external_gate(
        prepared_trial.reg,
        X_context_proc=prepared_trial.X_context_proc,
        y_context=splits.y_context,
        text_context=splits.text_context,
        X_train_proc=prepared_trial.X_train_proc,
        y_train=splits.y_train,
        text_train=splits.text_train,
        tuning_cfg=tuning_cfg,
        X_val_proc=prepared_trial.X_val_proc,
        y_val=splits.y_val,
        text_val=splits.text_val,
        prediction_horizon=prepared_trial.prediction_horizon,
    )


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
    prediction_horizon: int = 1,
) -> FineTuneOutcome:
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
                X_context_step, y_context_step, text_context_step = _build_rolling_train_step(
                    X_context_proc=X_context_proc,
                    y_context=y_context,
                    text_context=text_context,
                    X_eval_proc=X_train_proc,
                    y_eval=y_train,
                    text_eval=text_train,
                    idx=point_idx,
                    horizon=prediction_horizon,
                    max_context=tuning_cfg.max_context,
                )

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
            horizon=prediction_horizon,
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
            horizon=prediction_horizon,
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

    return FineTuneOutcome(
        top_validation_mae_checkpoints=top_validation_mae_checkpoints,
        best_epoch=best_epoch,
        best_score=best_score,
    )
def _mean_metric_triplets(metric_triplets: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    return tuple(
        float(np.mean([triplet[idx] for triplet in metric_triplets]))
        for idx in range(3)
    )


def _run_single_horizon(
    run_cfg: DataConfig,
    *,
    data_splits: FineTuneDataSplits,
    horizon: int,
) -> HorizonRunResult:
    print(f"\n{'=' * 16} Horizon {horizon}/{run_cfg.prediction_window} {'=' * 16}")
    prepared_trial = prepare_fine_tune_trial(run_cfg, data_splits=data_splits, horizon=horizon)
    reg = prepared_trial.reg
    splits = prepared_trial.splits
    X_context_proc = prepared_trial.X_context_proc
    X_train_proc = prepared_trial.X_train_proc
    X_val_proc = prepared_trial.X_val_proc
    X_test_proc = prepared_trial.X_test_proc
    y_context = splits.y_context
    text_context = splits.text_context
    y_train = splits.y_train
    text_train = splits.text_train
    y_val = splits.y_val
    text_val = splits.text_val
    y_test = splits.y_test
    text_test = splits.text_test

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
        horizon=horizon,
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
        horizon=horizon,
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
        horizon=horizon,
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
        horizon=horizon,
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
        horizon=horizon,
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
        horizon=horizon,
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
        horizon=horizon,
    )
    if run_cfg.tuning.debug_text_effect:
        delta_mae = baseline_text_val[0] - baseline_no_text_val[0]
        delta_rmse = baseline_text_val[1] - baseline_no_text_val[1]
        delta_mape = baseline_text_val[2] - baseline_no_text_val[2]
        print(
            f"Val text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
        print("Initial text-mixing params:", get_trainining_info(reg))
    else:
        print("Initial text-mixing gates:", text_mixing_gate_stats(reg))

    print("\n== Fine-tuning text-mixing parameters ==")
    fine_tune_outcome = fine_tune_prepared_trial(
        prepared_trial,
        tuning_cfg=run_cfg.tuning,
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
        horizon=horizon,
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
        horizon=horizon,
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
        horizon=horizon,
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
        horizon=horizon,
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
        horizon=horizon,
    )
    if run_cfg.tuning.debug_text_effect:
        delta_mae = tuned_text_test[0] - tuned_no_text_test[0]
        delta_rmse = tuned_text_test[1] - tuned_no_text_test[1]
        delta_mape = tuned_text_test[2] - tuned_no_text_test[2]
        print(
            f"Test text effect | ΔMAE={delta_mae:.6f} | "
            f"ΔRMSE={delta_rmse:.6f} | ΔMAPE={delta_mape:.6f}%"
        )
        print("Tuned text-mixing params:", get_trainining_info(reg))
    else:
        print("Tuned text-mixing gates:", text_mixing_gate_stats(reg))

    restored_best_state_dict = _clone_state_dict_to_cpu(reg.model.state_dict())
    print(
        f"\n== Top {len(fine_tune_outcome.top_validation_mae_checkpoints)} "
        "Validation-MAE Models On Test =="
    )
    for rank, checkpoint in enumerate(fine_tune_outcome.top_validation_mae_checkpoints, start=1):
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
            horizon=horizon,
        )
    reg.model.load_state_dict(restored_best_state_dict)

    return HorizonRunResult(
        horizon=horizon,
        best_epoch=fine_tune_outcome.best_epoch,
        tuned_no_text_test=tuned_no_text_test,
        tuned_text_test=tuned_text_test,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune text-mixing parameters on the final text-enhanced transformer layers."
    )
    parser.add_argument("--dataset", help="Dataset key in the config file.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    args = parser.parse_args()

    run_cfg = load_fine_tune_config(args.config, args.dataset)
    set_random_seeds(run_cfg.seed)

    if args.dataset:
        print(f"Using dataset '{args.dataset}' from {args.config}")
    else:
        print(f"Using dataset config {args.config}")

    data_splits = load_and_split_fine_tune_data(run_cfg)
    print(
        "Split sizes: "
        f"context={len(data_splits.y_context)} train={len(data_splits.y_train)} "
        f"val={len(data_splits.y_val)} test={len(data_splits.y_test)} | "
        f"prediction_window={run_cfg.prediction_window}"
    )

    horizon_results = [
        _run_single_horizon(run_cfg, data_splits=data_splits, horizon=horizon)
        for horizon in prediction_horizons(run_cfg)
    ]

    if len(horizon_results) > 1:
        print("\n================ Mean Summary Across Horizons ================")
        _format_metrics(
            "Mean test (no text attn)",
            *_mean_metric_triplets([result.tuned_no_text_test for result in horizon_results]),
        )
        _format_metrics(
            "Mean test (with text attn)",
            *_mean_metric_triplets([result.tuned_text_test for result in horizon_results]),
        )
        print(
            "Best epochs by horizon: "
            + ", ".join(f"h{result.horizon}={result.best_epoch}" for result in horizon_results)
        )


if __name__ == "__main__":
    main()
