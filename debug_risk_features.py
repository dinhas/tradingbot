"""Debug script to compare Risk model features between training and backtest."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from RiskLayer.src.feature_engine import RiskFeatureEngine

# Load data the same way RiskTradingEnv does during training
assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
data = {}

for asset in assets:
    labeled_path = f"data/{asset}_alpha_labeled.parquet"
    raw_path = f"data/{asset}_5m.parquet"
    
    df = pd.read_parquet(labeled_path)
    if 'alpha_confidence' in df.columns:
        df.rename(columns={'alpha_confidence': 'alpha_conf'}, inplace=True)
    
    # Check if OHLCV exists
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Merging {asset} with raw data to restore OHLCV...")
        raw_df = pd.read_parquet(raw_path)
        df = df[['alpha_signal', 'alpha_conf']].join(raw_df, how='inner')
    
    data[asset] = df
    print(f"Loaded {asset}: {len(df)} rows, columns: {list(df.columns)}")

# Process with feature engine
fe = RiskFeatureEngine()
processed = fe.preprocess_data(data)

print(f"\n=== PROCESSED DATA ===")
print(f"Shape: {processed.shape}")

# Check columns for one asset
eurusd_cols = [c for c in processed.columns if c.startswith('EURUSD_')]
print(f"\nEURUSD columns ({len(eurusd_cols)}):")
for i, col in enumerate(eurusd_cols):
    print(f"  [{i:2d}] {col}")

# Find alpha signal positions
for asset in assets:
    sig_col = f"{asset}_alpha_signal"
    conf_col = f"{asset}_alpha_conf"
    asset_cols = [c for c in processed.columns if c.startswith(f"{asset}_")]
    
    sig_found = sig_col in asset_cols
    conf_found = conf_col in asset_cols
    
    if sig_found:
        sig_idx = asset_cols.index(sig_col)
    else:
        sig_idx = -1
        
    if conf_found:
        conf_idx = asset_cols.index(conf_col)
    else:
        conf_idx = -1
    
    print(f"\n{asset}: signal_idx={sig_idx}, conf_idx={conf_idx}")
    if sig_idx != -1:
        sample_val = processed[sig_col].iloc[1000]
        print(f"  Sample signal value at row 1000: {sample_val}")

# Compare with what backtest would see
print("\n=== COMPARING WITH BACKTEST PIPELINE ===")
from Alpha.src.trading_env import TradingEnv

# Create TradingEnv (this is what backtest uses first)
env = TradingEnv(data_dir='data', stage=1, is_training=False)
print(f"TradingEnv loaded {len(env.data)} assets")

# Now process with Risk Feature Engine (same as backtest)
risk_processed = fe.preprocess_data(env.data)
print(f"Risk processed shape: {risk_processed.shape}")

# Check if columns match
backtest_eurusd_cols = [c for c in risk_processed.columns if c.startswith('EURUSD_')]
print(f"\nBacktest EURUSD columns ({len(backtest_eurusd_cols)}):")

# Compare
if len(eurusd_cols) == len(backtest_eurusd_cols):
    print("✅ Column count matches!")
else:
    print(f"❌ Column count MISMATCH: Training={len(eurusd_cols)}, Backtest={len(backtest_eurusd_cols)}")

# Check if alpha columns exist
for asset in assets:
    sig_col = f"{asset}_alpha_signal"
    conf_col = f"{asset}_alpha_conf"
    
    in_training = sig_col in processed.columns
    in_backtest = sig_col in risk_processed.columns
    
    print(f"{asset} alpha_signal: Training={in_training}, Backtest={in_backtest}")
