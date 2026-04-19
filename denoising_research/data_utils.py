import pandas as pd
import numpy as np

def load_research_data(filepath='data/EURUSD_5m.parquet', nrows=50000):
    df = pd.read_parquet(filepath)
    # Take the most recent data
    df = df.tail(nrows).copy()
    return df

if __name__ == "__main__":
    df = load_research_data()
    print(f"Loaded {len(df)} rows of data.")
    print(df.head())
