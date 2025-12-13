import pandas as pd
import os

data_dir = r"e:\tradingbot\data"
files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]

for f in files:
    path = os.path.join(data_dir, f)
    try:
        df = pd.read_parquet(path)
        print(f"File: {f}")
        print(f"  Shape: {df.shape}")
        if not df.empty:
            print(f"  Start: {df.index.min()}")
            print(f"  End: {df.index.max()}")
            print(f"  Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
