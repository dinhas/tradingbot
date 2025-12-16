from pathlib import Path
from .feature_engine import FeatureEngine
import pandas as pd
import os

def test_preprocessing():
    print("Starting test...")
    fe = FeatureEngine()
    # Construct a path to the data directory at the project root
    data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    data = {}
    
    print("Loading data...")
    for asset in assets:
        file_path = data_dir / f"{asset}_5m.parquet"
        if file_path.exists():
            print(f"Loading {asset}...")
            data[asset] = pd.read_parquet(file_path)
        else:
            print(f"File not found: {file_path}")
            
    print("Preprocessing data...")
    try:
        raw, processed = fe.preprocess_data(data)
        print("Preprocessing successful!")
        print("Processed shape:", processed.shape)
    except Exception as e:
        print("Preprocessing failed!")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_preprocessing()
