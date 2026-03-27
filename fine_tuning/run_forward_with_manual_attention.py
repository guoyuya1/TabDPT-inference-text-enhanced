from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tabdpt import TabDPTRegressor
from tabdpt.utils import pad_x

try:
    from .fine_tune_configs import load_fine_tune_config
    from .load_dataset import load_tabular_text_dataset
    from .split_ts import time_split
except ImportError:
    from fine_tune_configs import load_fine_tune_config
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


def preprocess_features(reg: TabDPTRegressor, X: np.ndarray) -> np.ndarray:
    X_proc = X
    if reg.missing_indicators:
        inds = np.isnan(X_proc)[:, reg.has_missing_indicator].astype(float)
        X_proc = np.hstack((X_proc, inds))
    X_proc = reg.imputer.transform(X_proc)
    if reg.scaler:
        X_proc = reg.scaler.transform(X_proc)
        if reg.normalizer == "quantile-uniform":
            X_proc = 2 * X_proc - 1
    return X_proc.astype(np.float32)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae


def rolling_forward_with_manual_attention(
    reg: TabDPTRegressor,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    manual_attn_records: list[dict],
    max_context: int | None,
) -> np.ndarray:
    if len(manual_attn_records) != len(y_eval):
        raise ValueError(
            f"Manual attention record count ({len(manual_attn_records)}) "
            f"does not match eval length ({len(y_eval)})."
        )

    reg.model.eval()
    preds = np.zeros(len(y_eval), dtype=np.float32)

    with torch.no_grad():
        for idx in range(len(y_eval)):
            X_train_full = np.concatenate((X_context_proc, X_eval_proc[:idx]))
            y_train_full = np.concatenate((y_context, y_eval[:idx]))

            if max_context is not None:
                X_train_step = X_train_full[-max_context:]
                y_train_step = y_train_full[-max_context:]
            else:
                X_train_step = X_train_full
                y_train_step = y_train_full

            X_test_step = X_eval_proc[idx : idx + 1]
            X_train_tensor = torch.tensor(X_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_train_tensor = pad_x(X_train_tensor, reg.max_features)
            X_test_tensor = torch.tensor(X_test_step, dtype=torch.float32, device=reg.device).unsqueeze(0)
            X_test_tensor = pad_x(X_test_tensor, reg.max_features)
            y_context_tensor = torch.tensor(y_train_step, dtype=torch.float32, device=reg.device).unsqueeze(0)

            manual_attn = manual_attn_records[idx]["final_attention_per_head"]
            if not isinstance(manual_attn, torch.Tensor):
                manual_attn = torch.as_tensor(manual_attn)
            # Expected shape from dump script: [num_heads, n_total_rows, n_context_rows]
            manual_attn = manual_attn.to(device=reg.device, dtype=torch.float32).unsqueeze(0)

            pred, _, _ = reg.model(
                x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                y_src=y_context_tensor.unsqueeze(-1),
                task="reg",
                text_enhanced_attn_weight=None,
                manual_last_layer_attn_weight=manual_attn,
            )
            preds[idx] = pred.squeeze(-1).reshape(-1).detach().cpu().numpy()[0]

    return preds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run forward-only inference by injecting saved last-layer attention weights."
    )
    parser.add_argument("--config", required=True, help="Target config YAML path.")
    parser.add_argument("--dataset", default=None, help="Optional dataset key in a multi-dataset config.")
    parser.add_argument(
        "--attention_pt",
        required=True,
        help="Path to .pt file saved by run_forward_with_attention_dump.py",
    )
    parser.add_argument(
        "--output_dir",
        default="fine_tuning/outputs/manual_attention_runs",
        help="Directory to save outputs.",
    )
    args = parser.parse_args()

    run_cfg = load_fine_tune_config(args.config, args.dataset)
    torch.manual_seed(run_cfg.seed)
    np.random.seed(run_cfg.seed)

    try:
        # PyTorch >=2.6 defaults to weights_only=True, which cannot load mixed
        # dict payloads containing numpy arrays from our attention dump.
        attn_payload = torch.load(args.attention_pt, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility with older PyTorch versions that do not expose
        # the weights_only kwarg.
        attn_payload = torch.load(args.attention_pt, map_location="cpu")
    tune_attn = attn_payload["attention"]["tune"]
    eval_attn = attn_payload["attention"]["eval"]

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
        _text_context,
        X_tune,
        y_tune,
        _text_tune,
        X_eval,
        y_eval,
        _text_eval,
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
    # Fit keeps preprocessing + context y, but manual attention will replace last-layer attention.
    reg.fit(X_context, y_context, None)

    X_context_proc = preprocess_features(reg, X_context)
    X_tune_proc = preprocess_features(reg, X_tune)
    X_eval_proc = preprocess_features(reg, X_eval)

    tune_preds = rolling_forward_with_manual_attention(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        X_eval_proc=X_tune_proc,
        y_eval=y_tune,
        manual_attn_records=tune_attn,
        max_context=run_cfg.tuning.max_context_for_tune_eval,
    )
    tune_rmse, tune_mae = _compute_metrics(y_tune, tune_preds)

    eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
    eval_y_context = np.concatenate((y_context, y_tune))
    eval_preds = rolling_forward_with_manual_attention(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        manual_attn_records=eval_attn,
        max_context=run_cfg.tuning.max_context_for_eval,
    )
    eval_rmse, eval_mae = _compute_metrics(y_eval, eval_preds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.dataset or Path(args.config).stem
    save_path = out_dir / f"{stem}_manual_attention_predictions.pt"
    torch.save(
        {
            "config_path": args.config,
            "dataset_key": args.dataset,
            "attention_source": args.attention_pt,
            "metrics": {
                "tune_rmse": tune_rmse,
                "tune_mae": tune_mae,
                "eval_rmse": eval_rmse,
                "eval_mae": eval_mae,
            },
            "predictions": {
                "tune_pred": tune_preds,
                "eval_pred": eval_preds,
                "tune_true": y_tune,
                "eval_true": y_eval,
            },
        },
        save_path,
    )

    print(f"Tune  | RMSE: {tune_rmse:.4f} | MAE: {tune_mae:.4f}")
    print(f"Eval  | RMSE: {eval_rmse:.4f} | MAE: {eval_mae:.4f}")
    print(f"Saved manual-attention predictions: {save_path}")


if __name__ == "__main__":
    main()
