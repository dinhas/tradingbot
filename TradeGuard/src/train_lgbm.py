
import os
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, log_loss

def _derive_temporal_features(df):
    ts = pd.to_datetime(df["timestamp"], unit="s")
    hour = ts.dt.hour.astype(int)
    day = ts.dt.dayofweek.astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * day / 7.0)
    df["session_asian"] = ((hour >= 0) & (hour < 8)).astype(int)
    df["session_london"] = ((hour >= 8) & (hour < 16)).astype(int)
    df["session_ny"] = ((hour >= 16) & (hour < 24)).astype(int)
    df["session_overlap"] = ((hour >= 12) & (hour < 16)).astype(int)
    return df

def _select_features(df):
    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    risk_cols = ["sl_mult", "tp_mult", "risk_raw", "sl_dist_atr", "tp_dist_atr", "rr_ratio"]
    account_cols = ["equity_norm", "drawdown", "risk_cap_mult", "num_open_positions"]
    trade_cols = ["direction", "asset_id", "atr"]
    temporal_cols = ["hour_sin", "hour_cos", "day_sin", "day_cos", "session_asian", "session_london", "session_ny", "session_overlap"]
    cols = alpha_cols + risk_cols + account_cols + trade_cols + temporal_cols
    return cols

def _time_split_train_val(df):
    ts = pd.to_datetime(df["timestamp"], unit="s")
    train_end = datetime(2023, 12, 31, 23, 59, 59)
    train_idx = ts <= train_end
    val_idx = ts > train_end
    return train_idx, val_idx

def _train(df, model_out, metrics_out):
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = _derive_temporal_features(df)
    feature_cols = _select_features(df)
    df["asset_id"] = df["asset_id"].astype("int32")
    X = df[feature_cols].astype(np.float32)
    y = df["outcome"].astype(int)
    train_idx, val_idx = _time_split_train_val(df)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    model = LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=0.1,
        is_unbalance=True,
        n_jobs=-1
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=["auc", "binary_logloss"],
        verbose=False
    )
    proba_val = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba_val) if len(np.unique(y_val)) > 1 else float("nan")
    ll = log_loss(y_val, np.clip(proba_val, 1e-6, 1 - 1e-6))
    os.makedirs(Path(model_out).parent, exist_ok=True)
    os.makedirs(Path(metrics_out).parent, exist_ok=True)
    model.booster_.save_model(model_out)
    with open(Path(model_out).with_suffix(".features.json"), "w", encoding="utf-8") as f:
        json.dump({"features": feature_cols}, f)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump({"auc": auc, "logloss": ll, "val_count": int(val_idx.sum())}, f, indent=2)

def _evaluate(df, model_path, features_path, out_metrics_path, out_proba_path):
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df = _derive_temporal_features(df)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)["features"]
    from lightgbm import Booster
    booster = Booster(model_file=model_path)
    df["asset_id"] = df["asset_id"].astype("int32")
    X = df[feature_cols].astype(np.float32)
    y = df["outcome"].astype(int) if "outcome" in df.columns else None
    proba = booster.predict(X)
    os.makedirs(Path(out_metrics_path).parent, exist_ok=True)
    os.makedirs(Path(out_proba_path).parent, exist_ok=True)
    if y is not None and len(np.unique(y)) > 1:
        auc = roc_auc_score(y, proba)
        ll = log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6))
    else:
        auc = float("nan")
        ll = float("nan")
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump({"auc": auc, "logloss": ll, "count": int(len(df))}, f, indent=2)
    out = pd.DataFrame({
        "proba": proba,
        "timestamp": df["timestamp"].values,
        "asset_id": df["asset_id"].values
    })
    if "outcome" in df.columns:
        out["outcome"] = df["outcome"].values
    out.to_parquet(out_proba_path)

def _build_dummy(n=5000, f_dim=140):
    rng = np.random.default_rng(42)
    rows = []
    base_ts = datetime(2016, 1, 1).timestamp()
    for i in range(n):
        row = {}
        for j in range(f_dim):
            row[f"alpha_{j}"] = rng.normal()
        row["sl_mult"] = rng.uniform(0.2, 2.0)
        row["tp_mult"] = rng.uniform(0.5, 4.0)
        row["risk_raw"] = rng.uniform(0.0, 1.0)
        row["atr"] = rng.uniform(0.0005, 0.01)
        row["sl_dist_atr"] = row["sl_mult"]
        row["tp_dist_atr"] = row["tp_mult"]
        row["rr_ratio"] = row["tp_mult"] / max(row["sl_mult"], 1e-9)
        row["equity_norm"] = rng.uniform(0.5, 2.0)
        row["drawdown"] = rng.uniform(0.0, 0.6)
        row["risk_cap_mult"] = rng.uniform(0.2, 1.0)
        row["num_open_positions"] = rng.integers(0, 5)
        row["direction"] = rng.choice([-1, 1])
        row["asset_id"] = rng.integers(0, 5)
        ts = base_ts + i * 60 * 60
        row["timestamp"] = ts if i < n * 0.9 else datetime(2024, 1, 10).timestamp() + i
        z = 0.3 * row["rr_ratio"] + 0.1 * row["tp_mult"] - 0.2 * row["sl_mult"] - 0.1 * row["drawdown"] + rng.normal(scale=0.2)
        row["outcome"] = 1 if z > 0.0 else 0
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", type=str, default="TradeGuard/data/dataset_2016_2024.parquet")
    parser.add_argument("--test-dataset", type=str, default=None)
    parser.add_argument("--model_out", type=str, default="TradeGuard/models/lgbm_tradeguard.txt")
    parser.add_argument("--metrics_out", type=str, default="TradeGuard/models/metrics_train_val.json")
    parser.add_argument("--metrics_test_out", type=str, default="TradeGuard/models/metrics_test_2025.json")
    parser.add_argument("--proba_test_out", type=str, default="TradeGuard/models/test_proba_2025.parquet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        df = _build_dummy()
    else:
        p = Path(args.train_dataset)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        df = pd.read_parquet(p)
    _train(df, args.model_out, args.metrics_out)
    if args.test_dataset:
        tp = Path(args.test_dataset)
        if tp.exists():
            tdf = pd.read_parquet(tp)
            _evaluate(
                tdf,
                args.model_out,
                Path(args.model_out).with_suffix(".features.json"),
                args.metrics_test_out,
                args.proba_test_out
            )

if __name__ == "__main__":
    main()
