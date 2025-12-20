import unittest
import sys
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

class TestDatasetGeneratorInit(unittest.TestCase):
    def setUp(self):
        # We expect the module to exist
        try:
            from generate_dataset import DatasetGenerator
            self.DatasetGenerator = DatasetGenerator
        except ImportError:
            self.DatasetGenerator = None

    def test_module_exists(self):
        """Test that the generate_dataset module and DatasetGenerator class exist."""
        if self.DatasetGenerator is None:
            self.fail("Could not import DatasetGenerator from generate_dataset")

    def test_initialization(self):
        """Test that the DatasetGenerator initializes correctly."""
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
        
        generator = self.DatasetGenerator()
        self.assertIsNotNone(generator.logger)
        self.assertTrue(isinstance(generator.logger, logging.Logger))
        self.assertEqual(generator.logger.name, "TradeGuard.DatasetGenerator")

    @patch('pathlib.Path.exists')
    @patch('pandas.read_parquet')
    def test_load_data(self, mock_read_parquet, mock_exists):
        """Test loading data for all assets."""
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
            
        # Mock return value
        mock_df = pd.DataFrame({'close': [1.1, 1.2], 'time': [0, 1]})
        mock_read_parquet.return_value = mock_df
        mock_exists.return_value = True # Pretend files exist
        
        generator = self.DatasetGenerator()
        # Override data_dir for test to avoid looking in real path
        generator.data_dir = Path("dummy/path")
        
        # Call the method
        try:
            data = generator.load_data()
        except AttributeError:
            self.fail("DatasetGenerator has no method load_data")
        
        assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.assertEqual(len(data), 5)
        for asset in assets:
            self.assertIn(asset, data)
            # Ensure the value is our mock df
            pd.testing.assert_frame_equal(data[asset], mock_df)
            
        self.assertEqual(mock_read_parquet.call_count, 5)

    @patch('stable_baselines3.PPO.load')
    def test_load_model(self, mock_ppo_load):
        """Test loading the Alpha model."""
        self.skipTest("Implementation pending")
        
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
            
        mock_model = MagicMock()
        mock_ppo_load.return_value = mock_model
        
        generator = self.DatasetGenerator()
        
        with patch('pathlib.Path.exists', return_value=True):
            model = generator.load_model("dummy_path.zip")
            
        self.assertIsNotNone(model)
        self.assertEqual(model, mock_model)
        mock_ppo_load.assert_called_once()

    def test_get_trade_signals(self):
        """Test generating trade signals from data."""
        self.skipTest("Implementation pending")


if __name__ == '__main__':
    unittest.main()