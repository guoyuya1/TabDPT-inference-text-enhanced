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
) -> tuple[float, float, float]:
    """
    Rolling evaluation with a fixed context and a sequential eval window.

    X_context_proc must contain all rows before the first eval row. If max_context
    is set, the model trains on only the last max_context rows of the rolling
    train window.
    """
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

    mae, rmse, mape = _compute_metrics(y_eval, preds)
    _format_metrics(label, mae, rmse, mape)
    return mae, rmse, mape
