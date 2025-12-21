import unittest
import sys
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

class TestTrainGuard(unittest.TestCase):
    
    def setUp(self):
        # Attempt import
        try:
            from train_guard import DataLoader
            self.DataLoader = DataLoader
        except ImportError:
            self.DataLoader = None

    def test_import_structure(self):
        """Test that the module and class exist."""
        if self.DataLoader is None:
            self.fail("Could not import DataLoader from train_guard.py. The module or class does not exist yet.")

    @patch('pandas.read_parquet')
    def test_load_and_split_logic(self, mock_read_parquet):
        """Test that data is loaded and split correctly by time."""
        if self.DataLoader is None:
            self.fail("DataLoader not implemented")
        
        # Create mock data covering 2016-2024
        # 2016-01-01 to 2024-12-31
        dates = pd.date_range(start='2016-01-01', end='2024-12-31', freq='D')
        mock_df = pd.DataFrame({
            'date': dates, 
            'feature1': range(len(dates)),
            'label': [0] * len(dates)
        })
        mock_df.set_index('date', inplace=True)
        
        mock_read_parquet.return_value = mock_df
        
        loader = self.DataLoader(file_path="dummy.parquet")
        train_df, val_df = loader.get_train_val_split()
        
        # Check validation set (Hold-out 2024)
        self.assertEqual(len(train_df) + len(val_df), len(mock_df))
        
        # Verify dates
        # Training set should end on or before 2023-12-31
        self.assertTrue(train_df.index.max() <= pd.Timestamp('2023-12-31'))
        
        # Validation set should start on or after 2024-01-01
        self.assertTrue(val_df.index.min() >= pd.Timestamp('2024-01-01'))
        
        # Verify specific counts
        # 2024 is a leap year (366 days)
        self.assertEqual(len(val_df), 366)

    @patch('pandas.read_parquet')
    def test_load_with_timestamp_column(self, mock_read_parquet):
        """Test that data with 'timestamp' column is correctly handled."""
        if self.DataLoader is None:
            self.fail("DataLoader not implemented")
            
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        mock_df = pd.DataFrame({
            'timestamp': dates,
            'val': range(10)
        })
        # Reset index so it's a RangeIndex, mimicking the parquet file
        
        mock_read_parquet.return_value = mock_df
        
        loader = self.DataLoader(file_path="dummy_ts.parquet")
        train_df, val_df = loader.get_train_val_split()
        
        self.assertTrue(isinstance(train_df.index, pd.DatetimeIndex))
        self.assertEqual(len(train_df), 10) # All are 2023


if __name__ == '__main__':
    unittest.main()
