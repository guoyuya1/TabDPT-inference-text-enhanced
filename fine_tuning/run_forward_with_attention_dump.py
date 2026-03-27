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


def rolling_forward_with_attention(
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
) -> tuple[np.ndarray, list[dict], list[dict]]:
    if use_text and (text_context is None or text_eval is None):
        raise ValueError("Text-enhanced run requires both context and eval text arrays.")

    reg.model.eval()
    preds = np.zeros(len(y_eval), dtype=np.float32)
    attn_records: list[dict] = []
    token_records: list[dict] = []

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

            X_test_step = X_eval_proc[idx : idx + 1]

            if use_text:
                train_text_batch = text_train_step[None, ...]
                test_text_batch = text_eval[idx : idx + 1][None, ...]
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

            pred, std_y, mean_y, final_attn, query_prehead_tokens = reg.model(
                x_src=torch.cat([X_train_tensor, X_test_tensor], dim=1),
                y_src=y_context_tensor.unsqueeze(-1),
                task="reg",
                text_enhanced_attn_weight=text_enhanced_attn_weight,
                return_last_layer_attn=True,
                return_query_prehead_tokens=True,
            )
            preds[idx] = pred.squeeze(-1).reshape(-1).detach().cpu().numpy()[0]

            # query_prehead_tokens shape: [n_query, B, ninp], here usually [1, 1, ninp]
            query_prehead_tokens = query_prehead_tokens.detach().cpu()
            query_flat = query_prehead_tokens.squeeze(1)  # [n_query, ninp]
            n_query, ninp = query_flat.shape
            head_dim = ninp // reg.model.num_heads
            query_by_head = query_flat.view(n_query, reg.model.num_heads, head_dim)

            attn_records.append(
                {
                    "step_index": idx,
                    "context_size": int(X_train_step.shape[0]),
                    "eval_pos": int(X_train_step.shape[0]),
                    # Shape: [num_heads, n_total_rows, n_context_rows] for this step.
                    "final_attention_per_head": final_attn.squeeze(0).detach().cpu(),
                    # Saved so predictions can be manually re-scaled/checked if needed.
                    "std_y": std_y.squeeze().detach().cpu(),
                    "mean_y": mean_y.squeeze().detach().cpu(),
                }
            )
            token_records.append(
                {
                    "step_index": idx,
                    "context_size": int(X_train_step.shape[0]),
                    "eval_pos": int(X_train_step.shape[0]),
                    # Final transformer output right before prediction head.
                    "query_prehead_tokens": query_flat,
                    # Same tensor split by attention heads: [n_query, num_heads, head_dim].
                    "query_prehead_tokens_by_head": query_by_head,
                }
            )

    return preds, attn_records, token_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run forward-only rolling inference and save final-layer attention weights."
    )
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--dataset", default=None, help="Optional dataset key in a multi-dataset config.")
    parser.add_argument(
        "--output_dir",
        default="fine_tuning/outputs/attention_dumps",
        help="Directory to save attention tensors and predictions.",
    )
    parser.add_argument(
        "--no_text",
        action="store_true",
        help="Disable text-enhanced attention even if the model supports it.",
    )
    args = parser.parse_args()

    run_cfg = load_fine_tune_config(args.config, args.dataset)
    torch.manual_seed(run_cfg.seed)
    np.random.seed(run_cfg.seed)

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

    X_context_proc = preprocess_features(reg, X_context)
    X_tune_proc = preprocess_features(reg, X_tune)
    X_eval_proc = preprocess_features(reg, X_eval)

    use_text = (not args.no_text) and reg.text_enhanced

    tune_preds, tune_attn, tune_tokens = rolling_forward_with_attention(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context if use_text else None,
        X_eval_proc=X_tune_proc,
        y_eval=y_tune,
        text_eval=text_tune if use_text else None,
        use_text=use_text,
        max_context=run_cfg.tuning.max_context_for_tune_eval,
    )
    tune_rmse, tune_mae = _compute_metrics(y_tune, tune_preds)

    eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
    eval_y_context = np.concatenate((y_context, y_tune))
    eval_text_context = np.concatenate((text_context, text_tune), axis=0)
    eval_preds, eval_attn, eval_tokens = rolling_forward_with_attention(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=eval_text_context if use_text else None,
        X_eval_proc=X_eval_proc,
        y_eval=y_eval,
        text_eval=text_eval if use_text else None,
        use_text=use_text,
        max_context=run_cfg.tuning.max_context_for_eval,
    )
    eval_rmse, eval_mae = _compute_metrics(y_eval, eval_preds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.dataset or Path(args.config).stem
    attn_save_path = out_dir / f"{stem}_forward_attn.pt"
    token_save_path = out_dir / f"{stem}_forward_tokens.pt"
    torch.save(
        {
            "config_path": args.config,
            "dataset_key": args.dataset,
            "use_text": use_text,
            "splits": {
                "context_size": int(len(y_context)),
                "tune_size": int(len(y_tune)),
                "eval_size": int(len(y_eval)),
            },
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
            "attention": {
                "tune": tune_attn,
                "eval": eval_attn,
            },
        },
        attn_save_path,
    )
    torch.save(
        {
            "config_path": args.config,
            "dataset_key": args.dataset,
            "use_text": use_text,
            "splits": {
                "context_size": int(len(y_context)),
                "tune_size": int(len(y_tune)),
                "eval_size": int(len(y_eval)),
            },
            "tokens": {
                "tune": tune_tokens,
                "eval": eval_tokens,
            },
        },
        token_save_path,
    )

    print(f"Tune  | RMSE: {tune_rmse:.4f} | MAE: {tune_mae:.4f}")
    print(f"Eval  | RMSE: {eval_rmse:.4f} | MAE: {eval_mae:.4f}")
    print(f"Saved attention dump: {attn_save_path}")
    print(f"Saved token dump: {token_save_path}")


if __name__ == "__main__":
    main()
