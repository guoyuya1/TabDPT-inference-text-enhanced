"""
Train GPT4MTS on a single CSV. Numeric scaling matches the upstream repo pattern in
`Dataset_GDELT.__read_data__`: fit StandardScaler on the train segment only, transform
the full series, then slice train/val/test. Metrics inverse-transform preds/targets so
MAE/RMSE/MAPE are on the original scale. GPT4MTS.forward still applies its own per-window
mean/std (same as upstream).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from GPT4MTS import GPT4MTS


@dataclass
class SplitData:
    x: np.ndarray
    s: np.ndarray
    y: np.ndarray


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, s: np.ndarray, y: np.ndarray, seq_len: int, pred_len: int):
        self.x = x
        self.s = s
        self.y = y
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n = len(x) - seq_len - pred_len + 1

    def __len__(self) -> int:
        return max(self.n, 0)

    def __getitem__(self, idx: int):
        a = idx
        b = idx + self.seq_len
        c = b + self.pred_len
        seq_x = torch.tensor(self.x[a:b], dtype=torch.float32)  # [L, C]
        seq_s = torch.tensor(self.s[a:b], dtype=torch.float32)  # [L, D]
        seq_y = torch.tensor(self.y[b:c], dtype=torch.float32)  # [P]
        return seq_x, seq_s, seq_y


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def patch_params_for_seq_len(seq_len: int, patch_size: int, stride: int) -> tuple[int, int]:
    """
    GPT4MTS pads length with stride on the right, then unfold(patch_size, stride).
    Need seq_len + stride >= patch_size. Cap patch_size to seq_len and stride to patch_size.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    ps = max(1, min(int(patch_size), seq_len))
    st = max(1, min(int(stride), ps))
    if seq_len + st < ps:
        st = max(1, ps - seq_len)
        if seq_len + st < ps:
            ps = seq_len
            st = 1
    return ps, st


def embed_texts(texts: list[str], model_name: str, device: torch.device, batch_size: int) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModel.from_pretrained(model_name).to(device)
    lm.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            h = lm(**enc).last_hidden_state[:, 0, :]  # CLS
            outs.append(h.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return mae, rmse, mape


def run_eval(
    model: GPT4MTS,
    loader: DataLoader,
    device: torch.device,
    scaler: StandardScaler | None = None,
) -> tuple[float, float, float]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for seq_x, seq_s, seq_y in loader:
            seq_x = seq_x.to(device)
            seq_s = seq_s.to(device)
            pred = model(seq_x, 0, seq_s).squeeze(-1)
            ys.append(seq_y.numpy())
            ps.append(pred.detach().cpu().numpy())
    y = np.concatenate(ys, axis=0).reshape(-1)
    p = np.concatenate(ps, axis=0).reshape(-1)
    if scaler is not None:
        y = scaler.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        p = scaler.inverse_transform(p.reshape(-1, 1)).reshape(-1)
    return compute_metrics(y, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "--conf", dest="config", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    df = pd.read_csv(cfg["data_path"])
    date_col = cfg.get("date_column")
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    if cfg.get("max_rows") is not None:
        df = df.head(int(cfg["max_rows"])).reset_index(drop=True)

    text_col = cfg["text_column"]
    target_col = cfg["target_column"]
    num_cols = [target_col]

    x = df[num_cols].astype(np.float32).to_numpy()
    y = df[target_col].astype(np.float32).to_numpy()
    texts = df[text_col].fillna("").astype(str).tolist()
    s = embed_texts(texts, cfg.get("text_encoder", "bert-base-uncased"), device, cfg.get("text_batch_size", 16))

    n = len(df)
    n_train = int(n * cfg["train_ratio"])
    n_val = int(n * cfg["val_ratio"])
    n_test = n - n_train - n_val
    s_train, s_val, s_test = s[:n_train], s[n_train : n_train + n_val], s[n_train + n_val :]

    target_scaler: StandardScaler | None = None
    if bool(cfg.get("scale", True)):
        target_scaler = StandardScaler()
        target_scaler.fit(y[:n_train].reshape(-1, 1))
        y = target_scaler.transform(y.reshape(-1, 1)).reshape(-1).astype(np.float32)
        # Univariate: input history is the same series as target.
        x = target_scaler.transform(x.reshape(-1, 1)).reshape(-1, x.shape[1]).astype(np.float32)

    x_train, x_val, x_test = x[:n_train], x[n_train : n_train + n_val], x[n_train + n_val :]
    y_train, y_val, y_test = y[:n_train], y[n_train : n_train + n_val], y[n_train + n_val :]

    train = SplitData(x_train, s_train, y_train)
    val = SplitData(x_val, s_val, y_val)
    test = SplitData(x_test, s_test, y_test)

    seq_len = int(cfg["seq_len"])
    pred_len = int(cfg["pred_len"])
    label_len = int(cfg.get("label_len", seq_len // 2))
    bs = int(cfg.get("batch_size", 8))

    req_ps = int(cfg.get("patch_size", 8))
    req_st = int(cfg.get("stride", 4))
    patch_size, stride = patch_params_for_seq_len(seq_len, req_ps, req_st)
    if patch_size != req_ps or stride != req_st:
        print(
            f"seq_len={seq_len}: adjusted patch_size {req_ps}->{patch_size}, stride {req_st}->{stride} "
            f"(short lookback; defaults assume seq_len >= patch_size).",
            flush=True,
        )

    train_loader = DataLoader(WindowDataset(train.x, train.s, train.y, seq_len, pred_len), batch_size=bs, shuffle=True)
    val_loader = DataLoader(WindowDataset(val.x, val.s, val.y, seq_len, pred_len), batch_size=bs, shuffle=False)
    test_loader = DataLoader(WindowDataset(test.x, test.s, test.y, seq_len, pred_len), batch_size=bs, shuffle=False)

    model_cfg = SimpleNamespace(
        is_gpt=1,
        revin=bool(cfg.get("revin", False)),
        patch_size=patch_size,
        pretrain=int(cfg.get("pretrain", 1)),
        stride=stride,
        seq_len=seq_len,
        gpt_layers=int(cfg.get("gpt_layers", 6)),
        d_model=int(cfg.get("d_model", 768)),
        pred_len=pred_len,
        freeze=int(cfg.get("freeze", 1)),
    )
    model = GPT4MTS(model_cfg, device).to(device)

    lr = float(cfg.get("learning_rate", 1e-3))
    epochs = int(cfg.get("train_epochs", 10))
    patience = int(cfg.get("patience", 3))
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    bad = 0
    ckpt = Path(cfg.get("checkpoint_path", "./gpt4mts_best.pt"))
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for seq_x, seq_s, seq_y in train_loader:
            seq_x = seq_x.to(device)
            seq_s = seq_s.to(device)
            seq_y = seq_y.to(device)
            pred = model(seq_x, ep, seq_s).squeeze(-1)
            loss = criterion(pred, seq_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.detach().cpu()))

        v_mae, v_rmse, v_mape = run_eval(model, val_loader, device, target_scaler)
        print(f"epoch {ep:03d} | train_loss={np.mean(losses):.6f} | val MAE={v_mae:.4f} RMSE={v_rmse:.4f} MAPE={v_mape:.2f}%")
        if v_rmse < best_val:
            best_val = v_rmse
            bad = 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print("early stopping")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device))
    va = run_eval(model, val_loader, device, target_scaler)
    te = run_eval(model, test_loader, device, target_scaler)
    print(f"best checkpoint: {ckpt}")
    print(f"val  | MAE={va[0]:.4f} RMSE={va[1]:.4f} MAPE={va[2]:.2f}%")
    print(f"test | MAE={te[0]:.4f} RMSE={te[1]:.4f} MAPE={te[2]:.2f}%")


if __name__ == "__main__":
    main()
