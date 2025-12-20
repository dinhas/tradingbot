import logging
import os
from pathlib import Path

class DatasetGenerator:
    """
    Generates training dataset for TradeGuard LightGBM model.
    """
    def __init__(self):
        self.logger = self.setup_logging()
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

if __name__ == "__main__":
    generator = DatasetGenerator()
