"""Rolling ContextTab (SAP RPT) baseline: tune / val / eval metrics from a YAML config."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sap_rpt_oss import SAP_RPT_OSS_Regressor


def _load_numeric(cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(cfg["data_path"])
    date_col = cfg.get("date_column")
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    if cfg.get("max_rows") is not None:
        df = df.head(cfg["max_rows"]).reset_index(drop=True)
    numeric_feats = list(cfg["numeric_features"])
    text_feats = list(cfg.get("text_columns") or [])
    X_num = df[numeric_feats].astype(np.float64).reset_index(drop=True)
    if text_feats:
        X_text = df[text_feats].fillna("").astype(str).reset_index(drop=True)
        X_df = pd.concat([X_num, X_text], axis=1)
    else:
        X_df = X_num
    y = df[cfg["target_column"]].astype(np.float32).to_numpy()
    return X_df, y


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return mae, rmse, mape


def _predict_one(reg: SAP_RPT_OSS_Regressor, X_ctx: pd.DataFrame, x_row: pd.DataFrame) -> float:
    if len(X_ctx) > 0:
        batch = pd.concat([X_ctx.iloc[-1:], x_row], ignore_index=True)
    else:
        batch = x_row
    out = reg.predict(batch)
    out = np.asarray(out)
    if out.ndim == 0:
        return float(out)
    return float(out.reshape(-1)[-1])


def rolling_predict(
    reg: SAP_RPT_OSS_Regressor,
    X_ctx: pd.DataFrame,
    y_ctx: np.ndarray,
    X_tgt: pd.DataFrame,
    y_tgt: np.ndarray,
    max_context: int | None,
    *,
    label: str = "",
    log_every: int = 100,
) -> np.ndarray:
    n = len(y_tgt)
    preds = np.zeros(n, dtype=np.float64)
    Xw, yw = X_ctx.copy(), y_ctx.copy()
    if label:
        print(f"[{label}] rolling {n} steps | prefix_rows={len(yw)} | cap={max_context}", flush=True)
    for i in range(n):
        if max_context is not None and len(yw) > max_context:
            ex = len(yw) - max_context
            Xw = Xw.iloc[ex:].reset_index(drop=True)
            yw = yw[ex:]
        reg.fit(Xw, yw)
        xi = X_tgt.iloc[i : i + 1]
        preds[i] = _predict_one(reg, Xw, xi)
        Xw = pd.concat([Xw, xi], ignore_index=True)
        yw = np.append(yw, y_tgt[i])
        done = i + 1
        if label:
            if done == 1 or done == n or (done % log_every == 0 and done < n):
                print(f"[{label}] {done}/{n} | fit_rows={len(yw)}", flush=True)
    return preds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    X_df, y = _load_numeric(cfg)

    n = len(y)
    cr, tr, er = cfg["context_ratio"], cfg["tune_ratio"], cfg["eval_ratio"]
    vr = float(cfg.get("val_ratio") or 0)
    n_ctx = int(n * cr)
    n_tune = int(n * tr)
    if vr > 0:
        n_val = int(n * vr)
        n_eval = n - n_ctx - n_tune - n_val
    else:
        n_val = 0
        n_eval = n - n_ctx - n_tune

    i1, i2, i3 = n_ctx, n_ctx + n_tune, n_ctx + n_tune + n_val
    Xc, yc = X_df.iloc[:i1], y[:i1]
    Xt, yt = X_df.iloc[i1:i2], y[i1:i2]
    Xv, yv = X_df.iloc[i2:i3], y[i2:i3]
    Xe, ye = X_df.iloc[i3:], y[i3:]

    ct = cfg.get("contexttab") or {}
    mc_tune = ct.get("max_context_for_tune")
    mc_val = ct.get("max_context_for_val")
    mc_eval = ct.get("max_context_for_eval")
    caps = [c for c in (mc_tune, mc_val, mc_eval) if c is not None]
    reg = SAP_RPT_OSS_Regressor(max_context_size=max(caps) if caps else 8192, bagging=8)

    for label, X0, y0, Xp, yp, max_context in (
        ("tune", Xc, yc, Xt, yt, mc_tune),
        (
            "validation",
            pd.concat([Xc, Xt], ignore_index=True),
            np.concatenate([yc, yt]),
            Xv,
            yv,
            mc_val,
        ),
        (
            "evaluation",
            pd.concat([Xc, Xt, Xv], ignore_index=True),
            np.concatenate([yc, yt, yv]),
            Xe,
            ye,
            mc_eval,
        ),
    ):
        if len(yp) == 0:
            continue
        pred = rolling_predict(reg, X0, y0, Xp, yp, max_context, label=label)
        mae, rmse, mape = _metrics(yp, pred)
        print(f"{label} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | MAPE: {mape:.4f}%")


if __name__ == "__main__":
    main()
