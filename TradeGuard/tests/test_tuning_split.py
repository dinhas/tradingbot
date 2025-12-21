import unittest
import sys
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

class TestTuningSplit(unittest.TestCase):
    
    def setUp(self):
        # Attempt import
        try:
            from train_guard import DataLoader
            self.DataLoader = DataLoader
        except ImportError:
            self.DataLoader = None

    def test_internal_tuning_split(self):
        """Test that the development set is split into Training (2016-2021) and Tuning (2022-2023)."""
        if self.DataLoader is None:
            self.fail("Could not import DataLoader from train_guard.py")
        
        # Create mock data covering 2016-2023 (The development set range)
        dates = pd.date_range(start='2016-01-01', end='2023-12-31', freq='D')
        mock_dev_df = pd.DataFrame({
            'date': dates, 
            'feature1': range(len(dates)),
            'label': [0] * len(dates)
        })
        mock_dev_df.set_index('date', inplace=True)
        
        loader = self.DataLoader(file_path="dummy.parquet")
        
        # We expect a method to split this internal data
        if not hasattr(loader, 'get_internal_tuning_split'):
            self.fail("DataLoader does not have method 'get_internal_tuning_split'")
            
        train_sub, tune_sub = loader.get_internal_tuning_split(mock_dev_df)
        
        # Verify Training Set (2016-2021)
        self.assertTrue(train_sub.index.min() >= pd.Timestamp('2016-01-01'))
        self.assertTrue(train_sub.index.max() <= pd.Timestamp('2021-12-31'))
        
        # Verify Tuning Set (2022-2023)
        self.assertTrue(tune_sub.index.min() >= pd.Timestamp('2022-01-01'))
        self.assertTrue(tune_sub.index.max() <= pd.Timestamp('2023-12-31'))
        
        # Verify no data loss
        self.assertEqual(len(train_sub) + len(tune_sub), len(mock_dev_df))

if __name__ == '__main__':
    unittest.main()
