import os
import subprocess
from pathlib import Path
import sys
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from TradeGuard.src.generate_dataset import TradeGuardDataGenerator
from TradeGuard.src.train_lgbm import _train, _evaluate
def ensure_training_data(output_dir):
    out = Path(output_dir)
    assets = ["EURUSD","GBPUSD","USDJPY","USDCHF","XAUUSD"]
    missing = []
    for a in assets:
        if not (out / f"{a}_5m.parquet").exists():
            missing.append(a)
    if missing:
        cmd = ["python","TradeGuard/src/download_data.py","--output",str(out)]
        subprocess.run(cmd, check=True)
def ensure_backtest_2025():
    base = Path("Alpha/backtest/data")
    assets = ["EURUSD","GBPUSD","USDJPY","USDCHF","XAUUSD"]
    missing = []
    for a in assets:
        if not (base / f"{a}_5m_2025.parquet").exists():
            missing.append(a)
    if missing:
        cmd = ["python","Alpha/backtest/data_fetcher_backtest.py"]
        subprocess.run(cmd, check=True)
def generate_dataset(data_dir, alpha_model, risk_model, output_path):
    g = TradeGuardDataGenerator(alpha_model_path=str(alpha_model), risk_model_path=str(risk_model), data_dir=str(data_dir))
    g.generate(str(output_path))
def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    tg_data = project_root / "TradeGuard" / "data"
    ensure_training_data(tg_data)
    alpha_model = project_root / "Alpha" / "models" / "checkpoints" / "8.03.zip"
    risk_model = project_root / "RiskLayer" / "models" / "2.15.zip"
    train_dataset = tg_data / "dataset_2016_2024.parquet"
    if not train_dataset.exists():
        generate_dataset(tg_data, alpha_model, risk_model, train_dataset)
    ensure_backtest_2025()
    bt_data = project_root / "Alpha" / "backtest" / "data"
    test_dataset = tg_data / "dataset_2025.parquet"
    if not test_dataset.exists():
        generate_dataset(bt_data, alpha_model, risk_model, test_dataset)
    model_out = project_root / "TradeGuard" / "models" / "lgbm_tradeguard.txt"
    metrics_train_val = project_root / "TradeGuard" / "models" / "metrics_train_val.json"
    _train(pd.read_parquet(train_dataset), str(model_out), str(metrics_train_val))
    metrics_test = project_root / "TradeGuard" / "models" / "metrics_test_2025.json"
    proba_out = project_root / "TradeGuard" / "models" / "test_proba_2025.parquet"
    _evaluate(pd.read_parquet(test_dataset), str(model_out), str(Path(model_out).with_suffix(".features.json")), str(metrics_test), str(proba_out))
if __name__ == "__main__":
    import pandas as pd
    main()
