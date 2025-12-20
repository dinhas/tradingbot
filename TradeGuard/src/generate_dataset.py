import logging
import os
from pathlib import Path
import pandas as pd

class DatasetGenerator:
    """
    Generates training dataset for TradeGuard LightGBM model.
    """
    def __init__(self):
        self.logger = self.setup_logging()
        self.data_dir = Path("data") 
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.logger.info("DatasetGenerator initialized")

    def setup_logging(self):
        """
        Sets up logging for the DatasetGenerator.
        """
        logger = logging.getLogger("TradeGuard.DatasetGenerator")
        # Only add handler if not already present to avoid duplicate logs
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_data(self):
        """
        Loads 5-minute OHLCV data for all defined assets.
        
        Returns:
            dict: Dictionary mapping asset names to pandas DataFrames.
        """
        data = {}
        self.logger.info("Loading historical data...")
        
        for asset in self.assets:
            file_path = self.data_dir / f"{asset}_5m.parquet"
            if file_path.exists():
                self.logger.info(f"Loading {asset} from {file_path}")
                try:
                    df = pd.read_parquet(file_path)
                    data[asset] = df
                except Exception as e:
                    self.logger.error(f"Failed to load {asset}: {e}")
            else:
                self.logger.warning(f"File not found: {file_path}")
        
        return data

if __name__ == "__main__":
    generator = DatasetGenerator()