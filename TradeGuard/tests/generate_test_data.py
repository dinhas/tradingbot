import pandas as pd
import numpy as np

def generate_test_dataset():
    df = pd.DataFrame(np.random.randn(10, 105), columns=[f'f_{i}' for i in range(105)])
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    for a in assets:
        df[f'target_{a}'] = np.random.choice([0, 1], 10)
        df[f'pnl_{a}'] = np.random.uniform(-10, 10, 10)
    df['timestamp'] = pd.date_range('2024-01-01', periods=10, freq='5min')
    df.to_parquet('TradeGuard/data/test_dataset.parquet')
    print("Generated test_dataset.parquet")

if __name__ == "__main__":
    generate_test_dataset()
