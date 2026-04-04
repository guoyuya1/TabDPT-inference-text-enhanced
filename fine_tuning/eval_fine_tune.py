from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tabdpt import TabDPTRegressor
from tabdpt.utils import pad_x


def _format_metrics(label: str, mae: float, rmse: float, mape: float) -> None:
    print(f"{label} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}%")


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return mae, rmse, mape


def _available_history_counts(fixed_context_len: int, step_idx: int, horizon: int) -> tuple[int, int]:
    """
    Count which labels are known at a rolling prediction step for a direct horizon model.

    For horizon h, row t predicts timestamp t+h-1, so when forecasting row `step_idx`
    in the rolling window, the last h-1 rows immediately before it still do not have
    observed targets. Those rows are therefore excluded from the training context until
    later steps.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    unavailable_fixed_tail = max(0, horizon - 1 - step_idx)
    fixed_count = max(0, fixed_context_len - unavailable_fixed_tail)
    rolling_count = max(0, step_idx - (horizon - 1))
    return fixed_count, rolling_count


def _build_rolling_train_step(
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray | None,
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    text_eval: np.ndarray | None,
    idx: int,
    horizon: int,
    max_context: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    fixed_count, rolling_count = _available_history_counts(len(y_context), idx, horizon)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    text_parts: list[np.ndarray] = []

    if fixed_count > 0:
        x_parts.append(X_context_proc[:fixed_count])
        y_parts.append(y_context[:fixed_count])
        if text_context is not None:
            text_parts.append(text_context[:fixed_count])

    if rolling_count > 0:
        x_parts.append(X_eval_proc[:rolling_count])
        y_parts.append(y_eval[:rolling_count])
        if text_eval is not None:
            text_parts.append(text_eval[:rolling_count])

    if not x_parts:
        raise ValueError(
            "No known context rows are available for this rolling step. "
            "Increase context_ratio or reduce prediction_window."
        )

    X_train_step = x_parts[0] if len(x_parts) == 1 else np.concatenate(x_parts, axis=0)
    y_train_step = y_parts[0] if len(y_parts) == 1 else np.concatenate(y_parts, axis=0)
    if text_context is None and text_eval is None:
        text_train_step = None
    else:
        text_train_step = text_parts[0] if len(text_parts) == 1 else np.concatenate(text_parts, axis=0)

    if max_context is not None:
        X_train_step = X_train_step[-max_context:]
        y_train_step = y_train_step[-max_context:]
        if text_train_step is not None:
            text_train_step = text_train_step[-max_context:]

    return X_train_step, y_train_step, text_train_step


def _append_text_pca_features(
    *,
    X_train_step: np.ndarray,
    X_test_step: np.ndarray,
    text_train_step: np.ndarray,
    text_test_step: np.ndarray,
    device: torch.device,
    max_total_features: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit PCA on text embeddings (train context only), per lag, and append PCA components to X.

    PCA rule (per user spec):
    - Let numeric_dim = X_train_step.shape[1]
    - Let embed_dim = text embedding feature dim (last axis)
    - q = floor((max_total_features - numeric_dim) / embed_dim), clamped to >=1
    - For each lag, fit PCA separately and transform both train and test.
    """
    if text_train_step.ndim != 3:
        raise ValueError(f"Expected text_train_step shape (N, L, D), got {text_train_step.shape}.")
    if text_test_step.ndim != 3:
        raise ValueError(f"Expected text_test_step shape (M, L, D), got {text_test_step.shape}.")
    if text_train_step.shape[1] != text_test_step.shape[1] or text_train_step.shape[2] != text_test_step.shape[2]:
        raise ValueError(
            "Train/test text embeddings must have same (lags, embed_dim). "
            f"Got train={text_train_step.shape}, test={text_test_step.shape}."
        )

    numeric_dim = int(X_train_step.shape[1])
    lags = int(text_train_step.shape[1])
    embed_dim = int(text_train_step.shape[2])

    remaining = max_total_features - numeric_dim
    q_target = remaining // max(1, embed_dim)
    q_target = max(1, int(q_target))

    pca_train_parts: list[np.ndarray] = []
    pca_test_parts: list[np.ndarray] = []

    with torch.no_grad():
        for lag_idx in range(lags):
            train_lag = torch.tensor(text_train_step[:, lag_idx, :], dtype=torch.float32, device=device)
            test_lag = torch.tensor(text_test_step[:, lag_idx, :], dtype=torch.float32, device=device)

            # Center using training mean.
            mean = train_lag.mean(dim=0, keepdim=True)
            train_centered = train_lag - mean
            test_centered = test_lag - mean

            # Low-rank PCA: choose q <= min(N-1, D).
            n = int(train_centered.shape[0])
            d = int(train_centered.shape[1])
            q = min(q_target, max(1, n - 1), d)

            # torch.pca_lowrank returns V with shape (D, q). We project X @ V.
            _, _, V = torch.pca_lowrank(train_centered, q=q, center=False)
            train_proj = train_centered @ V
            test_proj = test_centered @ V

            pca_train_parts.append(train_proj.detach().cpu().numpy().astype(np.float32, copy=False))
            pca_test_parts.append(test_proj.detach().cpu().numpy().astype(np.float32, copy=False))

    X_train_aug = np.concatenate([X_train_step] + pca_train_parts, axis=1).astype(np.float32, copy=False)
    X_test_aug = np.concatenate([X_test_step] + pca_test_parts, axis=1).astype(np.float32, copy=False)
    return X_train_aug, X_test_aug


def _choose_text_truncate_dim(
    *,
    numeric_dim: int,
    num_lags: int,
    embed_dim: int,
    max_total_features: int = 100,
) -> int | None:
    """
    Pick fixed per-lag truncation size: prefer 32, else 16, else unavailable.

    Requires numeric_dim + k * num_lags <= max_total_features and embed_dim >= k.
    """
    for k in (32, 16):
        if embed_dim < k:
            continue
        if numeric_dim + k * num_lags <= max_total_features:
            return k
    return None


def _append_text_truncate_features(
    *,
    X_train_step: np.ndarray,
    X_test_step: np.ndarray,
    text_train_step: np.ndarray,
    text_test_step: np.ndarray,
    truncate_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Take the first `truncate_dim` dims of each lag's embedding and concat to X (no PCA)."""
    if text_train_step.ndim != 3:
        raise ValueError(f"Expected text_train_step shape (N, L, D), got {text_train_step.shape}.")
    if text_test_step.ndim != 3:
        raise ValueError(f"Expected text_test_step shape (M, L, D), got {text_test_step.shape}.")
    if text_train_step.shape[1] != text_test_step.shape[1] or text_train_step.shape[2] != text_test_step.shape[2]:
        raise ValueError(
            "Train/test text embeddings must have same (lags, embed_dim). "
            f"Got train={text_train_step.shape}, test={text_test_step.shape}."
        )
    embed_dim = int(text_train_step.shape[2])
    if truncate_dim > embed_dim:
        raise ValueError(f"truncate_dim={truncate_dim} exceeds embed_dim={embed_dim}.")

    lags = int(text_train_step.shape[1])
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for lag_idx in range(lags):
        train_parts.append(
            text_train_step[:, lag_idx, :truncate_dim].astype(np.float32, copy=False)
        )
        test_parts.append(
            text_test_step[:, lag_idx, :truncate_dim].astype(np.float32, copy=False)
        )

    X_train_aug = np.concatenate([X_train_step] + train_parts, axis=1).astype(np.float32, copy=False)
    X_test_aug = np.concatenate([X_test_step] + test_parts, axis=1).astype(np.float32, copy=False)
    return X_train_aug, X_test_aug


def evaluate_rolling(
    reg: TabDPTRegressor,
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
    horizon: int = 1,
) -> tuple[float, float, float]:
    """
    Rolling evaluation with a fixed context and a sequential eval window.

    X_context_proc must contain all rows before the first eval row. For horizon
    h>1, the last h-1 rows immediately before each prediction are withheld until
    their targets become observable. If max_context is set, the model trains on
    only the last max_context known rows of the rolling train window.
    """
    if use_text and (text_context is None or text_eval is None):
        raise ValueError("Rolling eval with text requires context and eval text arrays.")

    reg.model.eval()
    preds = np.zeros(len(y_eval), dtype=np.float32)
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
            pred, _, _ = pred
            preds[idx] = pred.squeeze(-1).reshape(-1).detach().cpu().numpy()[0]

    mae, rmse, mape = _compute_metrics(y_eval, preds)
    _format_metrics(label, mae, rmse, mape)
    return mae, rmse, mape


def evaluate_rolling_pca(
    reg: TabDPTRegressor,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray,
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    text_eval: np.ndarray,
    label: str,
    max_context: int | None,
    horizon: int = 1,
    max_total_features: int = 100,
) -> tuple[float, float, float]:
    """
    Rolling evaluation like `evaluate_rolling`, but appends PCA-compressed text features to X.

    Notes:
    - Context/query selection is identical to `evaluate_rolling`.
    - PCA is fitted per rolling step on the *current training text context only*.
    - Each lag's embedding block gets its own PCA fit+transform.
    - No text attention is used; text is only used to derive PCA features appended to X.
    """
    reg.model.eval()
    preds = np.zeros(len(y_eval), dtype=np.float32)

    with torch.no_grad():
        for idx in range(len(y_eval)):
            X_train_step, y_train_step, text_train_step = _build_rolling_train_step(
                X_context_proc=X_context_proc,
                y_context=y_context,
                text_context=text_context,
                X_eval_proc=X_eval_proc,
                y_eval=y_eval,
                text_eval=text_eval,
                idx=idx,
                horizon=horizon,
                max_context=max_context,
            )

            X_test_step = X_eval_proc[idx:idx + 1]
            text_test_step = text_eval[idx:idx + 1]

            # Fit PCA on text_train_step, per lag, and append to X for this step.
            X_train_aug, X_test_aug = _append_text_pca_features(
                X_train_step=X_train_step,
                X_test_step=X_test_step,
                text_train_step=text_train_step,
                text_test_step=text_test_step,
                device=torch.device(reg.device),
                max_total_features=max_total_features,
            )

            X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_train_tensor = pad_x(X_train_tensor, reg.max_features)
            X_test_tensor = torch.tensor(X_test_aug, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_test_tensor = pad_x(X_test_tensor, reg.max_features)
            y_context_tensor = torch.tensor(y_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

            pred = reg.model(
                x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                y_src=y_context_tensor.unsqueeze(-1),
                task="reg",
                text_enhanced_attn_weight=None,
            )
            pred, _, _ = pred
            preds[idx] = pred.squeeze(-1).reshape(-1).detach().cpu().numpy()[0]

    mae, rmse, mape = _compute_metrics(y_eval, preds)
    _format_metrics(label, mae, rmse, mape)
    return mae, rmse, mape


def evaluate_rolling_truncate_text(
    reg: TabDPTRegressor,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray,
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    text_eval: np.ndarray,
    label: str,
    max_context: int | None,
    horizon: int = 1,
    max_total_features: int = 100,
) -> tuple[tuple[float, float, float], str] | None:
    """
    Rolling evaluation like `evaluate_rolling_pca`, but uses the first k dims of each lag's
    text embedding (k is 32 or 16 chosen from the 100-feature budget). No PCA, no text attention.

    Context/query row selection matches `evaluate_rolling` / `evaluate_rolling_pca`.
    If neither k=32 nor k=16 fits (budget or embed size), prints unavailable and returns None.

    Returns ((mae, rmse, mape), metric_label) so callers can re-print after tuning.
    """
    numeric_dim = int(X_context_proc.shape[1])
    num_lags = int(text_context.shape[1])
    embed_dim = int(text_context.shape[2])
    k = _choose_text_truncate_dim(
        numeric_dim=numeric_dim,
        num_lags=num_lags,
        embed_dim=embed_dim,
        max_total_features=max_total_features,
    )
    if k is None:
        print(f"{label} | text truncate: unavailable")
        return None

    metric_label = f"{label} d={k}"
    reg.model.eval()
    preds = np.zeros(len(y_eval), dtype=np.float32)

    with torch.no_grad():
        for idx in range(len(y_eval)):
            X_train_step, y_train_step, text_train_step = _build_rolling_train_step(
                X_context_proc=X_context_proc,
                y_context=y_context,
                text_context=text_context,
                X_eval_proc=X_eval_proc,
                y_eval=y_eval,
                text_eval=text_eval,
                idx=idx,
                horizon=horizon,
                max_context=max_context,
            )

            X_test_step = X_eval_proc[idx:idx + 1]
            text_test_step = text_eval[idx:idx + 1]

            X_train_aug, X_test_aug = _append_text_truncate_features(
                X_train_step=X_train_step,
                X_test_step=X_test_step,
                text_train_step=text_train_step,
                text_test_step=text_test_step,
                truncate_dim=k,
            )

            X_train_tensor = torch.tensor(X_train_aug, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_train_tensor = pad_x(X_train_tensor, reg.max_features)
            X_test_tensor = torch.tensor(X_test_aug, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_test_tensor = pad_x(X_test_tensor, reg.max_features)
            y_context_tensor = torch.tensor(y_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

            pred = reg.model(
                x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                y_src=y_context_tensor.unsqueeze(-1),
                task="reg",
                text_enhanced_attn_weight=None,
            )
            pred, _, _ = pred
            preds[idx] = pred.squeeze(-1).reshape(-1).detach().cpu().numpy()[0]

    mae, rmse, mape = _compute_metrics(y_eval, preds)
    _format_metrics(metric_label, mae, rmse, mape)
    return (mae, rmse, mape), metric_label
