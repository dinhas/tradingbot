import pandas as pd
import os
import glob

data_dir = r"E:\tradingbot\Alpha\backtest\data"
parquet_files = glob.glob(os.path.join(data_dir, "*_2025.parquet"))

print(f"Checking files in {data_dir}...")

for f in parquet_files:
    try:
        df = pd.read_parquet(f)
        if not df.empty:
            start_date = df.index.min()
            end_date = df.index.max()
            print(f"{os.path.basename(f)}: {len(df)} rows. Start: {start_date}, End: {end_date}")
        else:
            print(f"{os.path.basename(f)}: EMPTY")
    except Exception as e:
        print(f"{os.path.basename(f)}: Error reading - {e}")
