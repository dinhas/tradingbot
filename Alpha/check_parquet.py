
import pandas as pd
from pathlib import Path

# Construct a path to the parquet file, assuming this script is in Alpha/
# and the data directory is inside Alpha/backtest/data
file_path = Path(__file__).resolve().parent / "backtest" / "data" / "EURUSD_5m_2025.parquet"

try:
    df = pd.read_parquet(file_path)
    print(df.columns.tolist())
    print(df.head())
except Exception as e:
    print(e)
