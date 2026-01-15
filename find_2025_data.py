import os
import pandas as pd
from pathlib import Path

for path in Path('.').rglob('*.parquet'):
    try:
        df = pd.read_parquet(path)
        if not df.empty and hasattr(df, 'index') and isinstance(df.index, pd.DatetimeIndex):
            print(f"{path}: {df.index.min()} to {df.index.max()}")
    except Exception:
        pass
