import pandas as pd
from pathlib import Path

project_root = Path(r"e:\tradingbot")
assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']

for asset in assets:
    df = pd.read_parquet(project_root / "data" / f"{asset}_5m.parquet").tail(1)
    print(f"{asset}: {df.index[0]}")
