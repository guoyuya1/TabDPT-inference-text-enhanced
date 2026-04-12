from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer

warnings.filterwarnings("ignore")

PoolType = Literal["avg", "max", "min", "attention"]
PROMPT_TEMPLATE = (
    "<|start_prompt|>Make predictions about the future based on the "
    "following information: {text}<|end_prompt|>"
)


def instance_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True).detach()
    std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + eps)
    return (x - mean) / std


class MLP(nn.Module):
    def __init__(self, layer_sizes: list[int], dropout_rate: float = 0.3):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return self.dropout(x)


class DLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        nn.init.constant_(self.linear_trend.weight, 1.0 / seq_len)
        nn.init.constant_(self.linear_seasonal.weight, 1.0 / seq_len)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        x_ = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        return F.avg_pool1d(x_, kernel_size=self.kernel_size, stride=1).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self._moving_average(x)
        seasonal = x - trend
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)


class MMTSFModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        llm_dim: int,
        pool_type: PoolType = "avg",
        prompt_weight: float = 0.5,
        kernel_size: int = 25,
    ):
        super().__init__()
        self.pool_type = pool_type
        self.prompt_weight = float(prompt_weight)
        self.ts_model = DLinear(seq_len=seq_len, pred_len=pred_len, kernel_size=kernel_size)
        self.mlp = MLP([llm_dim, llm_dim // 8, pred_len], dropout_rate=0.3)

    def _pool(self, prompt_emb: torch.Tensor, ts_pred: torch.Tensor) -> torch.Tensor:
        t = prompt_emb.transpose(1, 2)
        if self.pool_type == "avg":
            return F.adaptive_avg_pool1d(t, 1).squeeze(2)
        if self.pool_type == "max":
            return F.adaptive_max_pool1d(t, 1).squeeze(2)
        if self.pool_type == "min":
            return -F.adaptive_max_pool1d(-t, 1).squeeze(2)
        if self.pool_type == "attention":
            pe_norm = F.normalize(prompt_emb, p=2, dim=2)
            ts_norm = F.normalize(ts_pred, p=2, dim=1).unsqueeze(2)
            scores = torch.bmm(pe_norm, ts_norm)
            weights = F.softmax(scores, dim=1)
            return torch.sum(prompt_emb * weights, dim=1)
        raise ValueError(f"Unknown pool_type: {self.pool_type!r}")

    def forward(self, ts_input: torch.Tensor, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        ts_pred = self.ts_model(ts_input)
        emb = self.mlp(prompt_embeddings)
        pooled = self._pool(emb, ts_pred)
        text_pred = instance_norm(pooled)
        w = self.prompt_weight
        return (1.0 - w) * ts_pred + w * text_pred


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_text_series(df: pd.DataFrame, cfg: dict) -> list[str]:
    if cfg.get("text_column"):
        return df[cfg["text_column"]].fillna("").astype(str).tolist()

    text_cols = list(cfg.get("text_columns") or [])
    if not text_cols:
        raise ValueError("Config must provide 'text_column' or non-empty 'text_columns'.")

    joined = (
        df[text_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return joined.tolist()


def _chronological_split_sizes(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val < 0 or n_test <= 0:
        raise ValueError(f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test} for n={n}")
    return n_train, n_val, n_test


def load_series_from_config(cfg: dict) -> tuple[np.ndarray, list[str]]:
    df = pd.read_csv(cfg["data_path"])
    date_col = cfg.get("date_column")
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    if cfg.get("max_rows") is not None:
        df = df.head(int(cfg["max_rows"])).reset_index(drop=True)

    target_col = cfg["target_column"]
    y = df[target_col].astype(np.float32).to_numpy()
    texts = _get_text_series(df, cfg)
    return y, texts


def build_windows(
    series: np.ndarray,
    text_emb_rows: np.ndarray,
    seq_len: int,
    pred_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(series)
    num = n - seq_len - pred_len + 1
    if num <= 0:
        return (
            np.zeros((0, seq_len), dtype=np.float32),
            np.zeros((0, text_emb_rows.shape[1], text_emb_rows.shape[2]), dtype=np.float32),
            np.zeros((0, pred_len), dtype=np.float32),
        )

    xs, es, ys = [], [], []
    for a in range(num):
        b = a + seq_len
        c = b + pred_len
        xs.append(series[a:b])
        ys.append(series[b:c])
        # Use text aligned with forecast origin (last timestep in lookback).
        es.append(text_emb_rows[b - 1])

    x = np.stack(xs).astype(np.float32)
    e = np.stack(es).astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    return x, e, y


def load_llm(device: torch.device, llm_name: str) -> tuple[GPT2Tokenizer, GPT2Model]:
    print(f"Loading LLM: {llm_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = GPT2Model.from_pretrained(llm_name)
    for p in llm.parameters():
        p.requires_grad = False
    llm = llm.to(device).eval()
    print("LLM loaded (weights frozen).")
    return tokenizer, llm


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer: GPT2Tokenizer,
    llm: GPT2Model,
    device: torch.device,
    max_token_len: int,
    batch_size: int,
) -> np.ndarray:
    prompts = [PROMPT_TEMPLATE.format(text=t) for t in texts]
    all_embs: list[np.ndarray] = []
    n_batches = math.ceil(len(prompts) / batch_size)
    for i in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="encoding texts"):
        batch = prompts[i : i + batch_size]
        input_ids = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_token_len,
        ).input_ids.to(device)
        h = llm(input_ids=input_ids).last_hidden_state
        all_embs.append(h.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))
    denom = np.clip(np.abs(y_true_flat), 1e-8, None)
    mape = float(np.mean(np.abs((y_true_flat - y_pred_flat) / denom)) * 100.0)
    return mae, rmse, mape


@torch.no_grad()
def evaluate_model(
    model: MMTSFModel,
    x: torch.Tensor,
    e: torch.Tensor,
    y: torch.Tensor,
    scaler: StandardScaler | None,
    batch_size: int,
) -> tuple[float, float, float, float]:
    model.eval()
    criterion = nn.MSELoss()
    n = x.shape[0]
    preds = []
    losses = []
    for i in range(0, n, batch_size):
        xb = x[i : i + batch_size]
        eb = e[i : i + batch_size]
        yb = y[i : i + batch_size]
        out = model(xb, eb)
        losses.append(float(criterion(out, yb).detach().cpu()))
        preds.append(out.detach().cpu().numpy())

    y_np = y.detach().cpu().numpy()
    p_np = np.concatenate(preds, axis=0).astype(np.float32)

    if scaler is not None:
        y_np = scaler.inverse_transform(y_np.reshape(-1, 1)).reshape(y_np.shape)
        p_np = scaler.inverse_transform(p_np.reshape(-1, 1)).reshape(p_np.shape)

    mae, rmse, mape = compute_metrics(y_np, p_np)
    mse = float(np.mean(losses)) if losses else float("nan")
    return mse, mae, rmse, mape


def train_with_early_stopping(
    x_train: np.ndarray,
    e_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    e_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    e_test: np.ndarray,
    y_test: np.ndarray,
    cfg: dict,
    scaler: StandardScaler | None,
    device: torch.device,
) -> None:
    seq_len = int(cfg["seq_len"])
    pred_len = int(cfg["pred_len"])
    pool_type = cfg.get("pool_type", "avg")
    prompt_weight = float(cfg.get("prompt_weight", 0.5))
    kernel_size = int(cfg.get("kernel_size", 25))
    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("train_epochs", 20))
    patience = int(cfg.get("patience", 3))

    model = MMTSFModel(
        seq_len=seq_len,
        pred_len=pred_len,
        llm_dim=int(e_train.shape[-1]),
        pool_type=pool_type,
        prompt_weight=prompt_weight,
        kernel_size=kernel_size,
    ).to(device)

    lr_ts = float(cfg.get("learning_rate", 1e-3))
    lr_mlp = float(cfg.get("learning_rate2", 1e-2))
    optim_ts = torch.optim.Adam(model.ts_model.parameters(), lr=lr_ts)
    optim_mlp = torch.optim.Adam(model.mlp.parameters(), lr=lr_mlp)
    criterion = nn.MSELoss()

    xtr = torch.from_numpy(x_train).to(device)
    etr = torch.from_numpy(e_train).to(device)
    ytr = torch.from_numpy(y_train).to(device)

    xva = torch.from_numpy(x_val).to(device)
    eva = torch.from_numpy(e_val).to(device)
    yva = torch.from_numpy(y_val).to(device)

    xte = torch.from_numpy(x_test).to(device)
    ete = torch.from_numpy(e_test).to(device)
    yte = torch.from_numpy(y_test).to(device)

    ckpt = Path(cfg.get("checkpoint_path", "./baseline/multimodal/mm_tsflib_best.pt"))
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    best_rmse = float("inf")
    bad = 0

    n = xtr.shape[0]
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        losses = []

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = xtr[idx]
            eb = etr[idx]
            yb = ytr[idx]

            optim_ts.zero_grad()
            optim_mlp.zero_grad()
            out = model(xb, eb)
            loss = criterion(out, yb)
            loss.backward()
            optim_ts.step()
            optim_mlp.step()
            losses.append(float(loss.detach().cpu()))

        _, v_mae, v_rmse, v_mape = evaluate_model(model, xva, eva, yva, scaler, batch_size)
        print(
            f"epoch {ep:03d} | train_loss={np.mean(losses):.6f} | "
            f"val MAE={v_mae:.4f} RMSE={v_rmse:.4f} MAPE={v_mape:.2f}%"
        )

        if v_rmse < best_rmse:
            best_rmse = v_rmse
            bad = 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    val_scores = evaluate_model(model, xva, eva, yva, scaler, batch_size)
    test_scores = evaluate_model(model, xte, ete, yte, scaler, batch_size)

    print(f"best checkpoint: {ckpt}")
    print(
        f"val  | MAE={val_scores[1]:.4f} RMSE={val_scores[2]:.4f} "
        f"MAPE={val_scores[3]:.2f}%"
    )
    print(
        f"test | MAE={test_scores[1]:.4f} RMSE={test_scores[2]:.4f} "
        f"MAPE={test_scores[3]:.2f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "--conf", dest="config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seq_len = int(cfg["seq_len"])
    pred_len = int(cfg["pred_len"])
    llm_name = cfg.get("llm_name", "openai-community/gpt2")
    max_token_len = int(cfg.get("max_token_len", 256))
    text_batch_size = int(cfg.get("text_batch_size", 16))

    series, texts = load_series_from_config(cfg)
    n = len(series)
    n_train, n_val, n_test = _chronological_split_sizes(
        n,
        float(cfg.get("train_ratio", 0.65)),
        float(cfg.get("val_ratio", 0.2)),
    )

    print(f"rows={n} | train={n_train} val={n_val} test={n_test} | device={device}")

    scaler: StandardScaler | None = None
    if bool(cfg.get("scale", True)):
        scaler = StandardScaler()
        scaler.fit(series[:n_train].reshape(-1, 1))
        series = scaler.transform(series.reshape(-1, 1)).reshape(-1).astype(np.float32)

    tokenizer, llm = load_llm(device, llm_name)
    row_emb = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        llm=llm,
        device=device,
        max_token_len=max_token_len,
        batch_size=text_batch_size,
    )
    del llm

    tr_slice = slice(0, n_train)
    va_slice = slice(n_train, n_train + n_val)
    te_slice = slice(n_train + n_val, n)

    x_train, e_train, y_train = build_windows(series[tr_slice], row_emb[tr_slice], seq_len, pred_len)
    x_val, e_val, y_val = build_windows(series[va_slice], row_emb[va_slice], seq_len, pred_len)
    x_test, e_test, y_test = build_windows(series[te_slice], row_emb[te_slice], seq_len, pred_len)

    if min(len(x_train), len(x_val), len(x_test)) == 0:
        raise ValueError(
            "One split produced zero windows. Increase max_rows or adjust train/val ratios "
            "and seq_len/pred_len."
        )

    print(
        f"windows | train={len(x_train)} val={len(x_val)} test={len(x_test)} | "
        f"emb_shape={tuple(e_train.shape[1:])}"
    )

    train_with_early_stopping(
        x_train=x_train,
        e_train=e_train,
        y_train=y_train,
        x_val=x_val,
        e_val=e_val,
        y_val=y_val,
        x_test=x_test,
        e_test=e_test,
        y_test=y_test,
        cfg=cfg,
        scaler=scaler,
        device=device,
    )


if __name__ == "__main__":
    main()
