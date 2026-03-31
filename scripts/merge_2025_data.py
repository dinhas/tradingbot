import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BACKTEST_DATA_DIR = PROJECT_ROOT / "backtest" / "data"

ASSETS = ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDCHF', 'USDJPY']

def merge_data():
    for asset in ASSETS:
        main_file = DATA_DIR / f"{asset}_5m.parquet"
        new_file = BACKTEST_DATA_DIR / f"{asset}_5m_2025.parquet"
        
        if not main_file.exists():
            logger.warning(f"Main file missing for {asset}: {main_file}")
            continue
            
        if not new_file.exists():
            logger.warning(f"New 2025 file missing for {asset}: {new_file}")
            continue
            
        # Load data
        df_main = pd.read_parquet(main_file)
        df_2025 = pd.read_parquet(new_file)
        
        logger.info(f"--- {asset} ---")
        logger.info(f"Main data range: {df_main.index.min()} to {df_main.index.max()} ({len(df_main)} rows)")
        logger.info(f"2025 data range: {df_2025.index.min()} to {df_2025.index.max()} ({len(df_2025)} rows)")
        
        # Combine
        df_combined = pd.concat([df_main, df_2025])
        
        # Drop duplicates based on index (timestamp)
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        
        # Sort by index
        df_combined.sort_index(inplace=True)
        
        logger.info(f"Combined data range: {df_combined.index.min()} to {df_combined.index.max()} ({len(df_combined)} rows)")
        
        # Save
        df_combined.to_parquet(main_file)
        logger.info(f"Successfully merged and saved {main_file}")

if __name__ == "__main__":
    merge_data()
