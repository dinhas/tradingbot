import pandas as pd
import numpy as np
from pathlib import Path
from ta.trend import EMAIndicator, MACD
import logging

def generate_proxy_signals(df):
    """
    Generates realistic Trend Following signals to simulate an Alpha Model.
    Strategy:
    1. Trend Filter: Price > EMA 200 (Bullish), Price < EMA 200 (Bearish)
    2. Entry Trigger: MACD Crossover
    """
    # 1. Indicators
    close = df['close']
    ema_200 = EMAIndicator(close, window=200).ema_indicator()
    macd = MACD(close)
    macd_hist = macd.macd_diff()
    
    # 2. Logic
    signal = np.zeros(len(df))
    
    # Longs
    # Trend is up AND MACD flips positive
    long_condition = (close > ema_200) & (macd_hist > 0) & (macd_hist.shift(1) <= 0)
    
    # Shorts
    # Trend is down AND MACD flips negative
    short_condition = (close < ema_200) & (macd_hist < 0) & (macd_hist.shift(1) >= 0)
    
    signal[long_condition] = 1.0
    signal[short_condition] = -1.0
    
    # Apply signal "hold" or just triggers?
    # The environment expects a signal at the current step to indicate "Alpha says BUY".
    # We can leave it as triggers (sparse) or fill it forward for a few bars.
    # Triggers are better for the "Sniper" - it sees the call and decides.
    
    df['alpha_signal'] = signal
    return df

def build_training_set(data_dir='data', output_dir='RiskLayer/data'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    
    for asset in assets:
        # Try finding source data
        src_path_2025 = Path(data_dir) / f"{asset}_5m_2025.parquet"
        src_path = Path(data_dir) / f"{asset}_5m.parquet"
        
        df = None
        if src_path_2025.exists():
            df = pd.read_parquet(src_path_2025)
        elif src_path.exists():
            df = pd.read_parquet(src_path)
            
        if df is not None:
            logging.info(f"Processing {asset}...")
            # Generate Signals
            df = generate_proxy_signals(df)
            
            # Save to RiskLayer data
            out_path = Path(output_dir) / f"{asset}_5m_labeled.parquet"
            df.to_parquet(out_path)
            logging.info(f"Saved {out_path} with {len(df)} rows and {df['alpha_signal'].abs().sum()} signals.")
        else:
            logging.warning(f"Source data not found for {asset}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_training_set()
