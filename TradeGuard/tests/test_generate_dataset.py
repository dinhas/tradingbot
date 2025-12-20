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

    @patch('pandas.read_parquet')
    def test_load_data(self, mock_read_parquet):
        """Test loading data for all assets."""
        self.skipTest("Implementation pending")
        
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
            
        # Mock return value
        mock_df = pd.DataFrame({'close': [1.1, 1.2], 'time': [0, 1]})
        mock_read_parquet.return_value = mock_df
        
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

if __name__ == '__main__':
    unittest.main()