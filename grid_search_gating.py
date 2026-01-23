"""
Grid search over per-head gating values with rolling-window evaluation.

Each head's gating value is chosen from [0.0, 0.1, ..., 1.0] by default.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import torch

from fine_tune_dpt import (
    get_last_layer_gate_param,
    load_dataset_config,
    load_tabdpt_regressor,
    preprocess_features,
)
from load_dataset import load_climate_dataset
from split_ts import time_split
from tabdpt.utils import pad_x


torch.manual_seed(0)
np.random.seed(0)

DATASET_CONFIG_PATH = "configs/bitcoin.yaml"
DEFAULT_DATASET = None

MAX_ROWS = 1000
CONTEXT_RATIO = 0.2
TUNE_RATIO = 0.65
EVAL_RATIO = 0.15

GRID_STEP = 0.1
MAX_COMBINATIONS = 50000
LOG_EVERY = 100
RANDOM_SAMPLES = 1000

# Eval range within the eval split.
EVAL_START = 0
EVAL_END = None
MAX_CONTEXT_FOR_EVAL = 1000
MAX_CONTEXT_FOR_TUNE = MAX_CONTEXT_FOR_EVAL

MODEL_WEIGHT_PATH = None
DEVICE = None


def _prob_to_logit(prob: float, clamp: float) -> float:
    if prob <= 0.0:
        return -clamp
    if prob >= 1.0:
        return clamp
    return float(np.log(prob / (1.0 - prob)))


def _set_gate_weights(reg, weights: np.ndarray, clamp: float = 10.0) -> torch.nn.Parameter:
    gate = get_last_layer_gate_param(reg)
    if weights.size != gate.numel():
        raise ValueError(f"Per-head weights must match num_heads={gate.numel()}, got {weights.size}.")
    logits = np.array([_prob_to_logit(w, clamp) for w in weights], dtype=np.float32)
    with torch.no_grad():
        gate.copy_(torch.tensor(logits, dtype=gate.dtype, device=gate.device))
    return gate


def _slice_eval_range(
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    text_eval: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    start = 0 if EVAL_START is None else int(EVAL_START)
    end = len(y_eval) if EVAL_END is None else int(EVAL_END)
    if start < 0 or start >= len(y_eval):
        raise ValueError(f"EVAL_START must be in [0, {len(y_eval) - 1}], got {start}.")
    if end <= start or end > len(y_eval):
        raise ValueError(f"EVAL_END must be in [{start + 1}, {len(y_eval)}], got {end}.")
    return start, end, X_eval_proc[start:end], y_eval[start:end], text_eval[start:end]


def _compute_metrics_from_preds(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return mae, rmse, mape


def _rolling_eval_metrics(
    reg,
    *,
    X_context_proc: np.ndarray,
    y_context: np.ndarray,
    text_context: np.ndarray | None,
    X_eval_proc: np.ndarray,
    y_eval: np.ndarray,
    text_eval: np.ndarray | None,
    use_text: bool,
    max_context: int | None,
) -> tuple[float, float, float]:
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

    return _compute_metrics_from_preds(y_eval, preds)


def _build_grid(step: float) -> list[float]:
    values = np.arange(0.0, 1.0 + step / 2.0, step)
    return [round(float(v), 3) for v in values]


def _iter_random_combos(
    grid: list[float],
    num_heads: int,
    samples: int,
    seed: int,
) -> tuple[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    for idx in range(1, samples + 1):
        combo = rng.choice(grid, size=num_heads, replace=True)
        yield idx, combo


def _update_top_k(
    best: list[tuple[float, np.ndarray]],
    *,
    score: float,
    combo: np.ndarray,
    k: int,
) -> list[tuple[float, np.ndarray]]:
    updated = best + [(score, combo.copy())]
    updated.sort(key=lambda item: item[0])
    return updated[:k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid or random search over per-head gating values."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset key in the config file.")
    parser.add_argument("--config", default=DATASET_CONFIG_PATH, help="Path to dataset config YAML.")
    parser.add_argument("--grid-step", type=float, default=GRID_STEP, help="Step size for gating grid.")
    parser.add_argument(
        "--random-samples",
        type=int,
        default=RANDOM_SAMPLES,
        help="If >0, run random search with this many samples.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--max-combinations", type=int, default=MAX_COMBINATIONS, help="Safety cap.")
    parser.add_argument("--force", action="store_true", help="Run even if combinations exceed cap.")
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--top-k", type=int, default=5, help="Number of top tuning combos to report.")
    args = parser.parse_args()

    dataset_cfg = load_dataset_config(args.config, args.dataset)
    data_path = dataset_cfg.get("data_path")
    date_column = dataset_cfg.get("date_column")
    numeric_features = dataset_cfg.get("numeric_features")
    target_column = dataset_cfg.get("target_column")
    embedding_columns = dataset_cfg.get("embedding_columns")
    embedding_lags = dataset_cfg.get("embedding_lags")
    embedding_column_template = dataset_cfg.get("embedding_column_template")

    if args.dataset:
        print(f"Using dataset '{args.dataset}' from {args.config}")
    else:
        print(f"Using dataset config {args.config}")

    X, y, text = load_climate_dataset(
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
        context_ratio=CONTEXT_RATIO,
        tune_ratio=TUNE_RATIO,
        eval_ratio=EVAL_RATIO,
    )
    print(f"Split sizes: context={len(y_context)} tune={len(y_tune)} eval={len(y_eval)}")

    reg = load_tabdpt_regressor(device=DEVICE, model_weight_path=MODEL_WEIGHT_PATH, text_enhanced=True)
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

    eval_context_proc = np.concatenate((X_context_proc, X_tune_proc))
    eval_y_context = np.concatenate((y_context, y_tune))
    eval_text_context = np.concatenate((text_context, text_tune), axis=0)

    start, end, X_eval_range, y_eval_range, text_eval_range = _slice_eval_range(
        X_eval_proc,
        y_eval,
        text_eval,
    )
    if start > 0:
        eval_context_proc = np.concatenate((eval_context_proc, X_eval_proc[:start]))
        eval_y_context = np.concatenate((eval_y_context, y_eval[:start]))
        eval_text_context = np.concatenate((eval_text_context, text_eval[:start]), axis=0)
    print(f"Eval range: [{start}:{end}) out of eval split ({len(y_eval)} rows).")

    baseline_tune = _rolling_eval_metrics(
        reg,
        X_context_proc=X_context_proc,
        y_context=y_context,
        text_context=text_context,
        X_eval_proc=X_tune_proc,
        y_eval=y_tune,
        text_eval=text_tune,
        use_text=False,
        max_context=MAX_CONTEXT_FOR_TUNE,
    )
    baseline_eval = _rolling_eval_metrics(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=eval_text_context,
        X_eval_proc=X_eval_range,
        y_eval=y_eval_range,
        text_eval=text_eval_range,
        use_text=False,
        max_context=MAX_CONTEXT_FOR_EVAL,
    )
    print(
        "Baseline (no text attn) "
        f"| Tune MAE: {baseline_tune[0]:.4f} | Eval MAE: {baseline_eval[0]:.4f}"
    )

    gate = get_last_layer_gate_param(reg)
    num_heads = gate.numel()
    grid = _build_grid(args.grid_step)
    total = len(grid) ** num_heads
    print(f"Grid values: {grid}")
    print(f"num_heads={num_heads} total_combinations={total}")
    if args.random_samples <= 0:
        if total > args.max_combinations and not args.force:
            raise ValueError(
                f"Grid size {total} exceeds max {args.max_combinations}. "
                f"Use --grid-step to reduce or --force to proceed."
            )
        total_to_run = total
        iterator = enumerate(itertools.product(grid, repeat=num_heads), start=1)
        print("Search mode: full grid")
    else:
        total_to_run = min(args.random_samples, total)
        iterator = _iter_random_combos(grid, num_heads, total_to_run, seed=args.seed)
        print(f"Search mode: random ({total_to_run} samples, seed={args.seed})")

    best_score = None
    best_combo = None
    best_metrics = None
    top_k: list[tuple[float, np.ndarray]] = []

    for idx, combo in iterator:
        combo_arr = np.array(combo, dtype=np.float32)
        _set_gate_weights(reg, combo_arr)
        metrics = _rolling_eval_metrics(
            reg,
            X_context_proc=X_context_proc,
            y_context=y_context,
            text_context=text_context,
            X_eval_proc=X_tune_proc,
            y_eval=y_tune,
            text_eval=text_tune,
            use_text=True,
            max_context=MAX_CONTEXT_FOR_TUNE,
        )
        tune_mae = metrics[0]
        if best_score is None or tune_mae < best_score:
            best_score = tune_mae
            best_combo = combo_arr.copy()
            best_metrics = metrics
            print(f"New best (tune_mae)={best_score:.4f} combo={best_combo.tolist()}")
        top_k = _update_top_k(top_k, score=tune_mae, combo=combo_arr, k=args.top_k)
        if args.log_every and idx % args.log_every == 0:
            print(f"\nProgress: {idx}/{total_to_run}")
            print(f"Top {args.top_k} (Tune MAE -> Eval MAE):")
            for rank, (tune_mae, combo) in enumerate(top_k, start=1):
                _set_gate_weights(reg, combo)
                eval_metrics = _rolling_eval_metrics(
                    reg,
                    X_context_proc=eval_context_proc,
                    y_context=eval_y_context,
                    text_context=eval_text_context,
                    X_eval_proc=X_eval_range,
                    y_eval=y_eval_range,
                    text_eval=text_eval_range,
                    use_text=True,
                    max_context=MAX_CONTEXT_FOR_EVAL,
                )
                print(
                    f"  {rank}. tune_mae={tune_mae:.4f} eval_mae={eval_metrics[0]:.4f} "
                    f"combo={combo.tolist()}"
                )

    if best_combo is None or best_metrics is None:
        raise RuntimeError("No combinations evaluated.")

    print("\n== Best combination ==")
    print(f"gating={best_combo.tolist()}")
    _set_gate_weights(reg, best_combo)
    best_eval_metrics = _rolling_eval_metrics(
        reg,
        X_context_proc=eval_context_proc,
        y_context=eval_y_context,
        text_context=eval_text_context,
        X_eval_proc=X_eval_range,
        y_eval=y_eval_range,
        text_eval=text_eval_range,
        use_text=True,
        max_context=MAX_CONTEXT_FOR_EVAL,
    )
    print(
        f"Tune MAE: {best_metrics[0]:.4f} | "
        f"Eval MAE: {best_eval_metrics[0]:.4f}"
    )


if __name__ == "__main__":
    main()
