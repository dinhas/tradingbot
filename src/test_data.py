from .feature_engine import FeatureEngine
import pandas as pd
import os

def test_preprocessing():
    print("Starting test...")
    fe = FeatureEngine()
    data_dir = "data"
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    data = {}
    
    print("Loading data...")
    for asset in assets:
        file_path = f"{data_dir}/{asset}_5m.parquet"
        if os.path.exists(file_path):
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
