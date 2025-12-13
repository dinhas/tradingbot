
import pandas as pd
try:
    df = pd.read_parquet(r"e:\tradingbot\backtest\data\EURUSD_5m_2025.parquet")
    print(df.columns.tolist())
    print(df.head())
except Exception as e:
    print(e)
