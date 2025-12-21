import unittest
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

class TestLogging(unittest.TestCase):
    
    def setUp(self):
        try:
            from train_guard import ModelTrainer
            self.ModelTrainer = ModelTrainer
        except ImportError:
            self.ModelTrainer = None
            
        # Configure logging to capture output
        logging.basicConfig(level=logging.INFO)

    def test_optimization_logging(self):
        """Test that optimization process logs the best parameters."""
        if self.ModelTrainer is None:
            self.fail("Could not import ModelTrainer")
            
        # Dummy data
        dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
        X = pd.DataFrame(np.random.rand(10, 2), columns=['f1', 'f2'], index=dates)
        y = pd.Series([0, 1] * 5, index=dates, name='label')
        
        train_df = X.copy(); train_df['label'] = y
        tune_df = X.copy(); tune_df['label'] = y
        
        trainer = self.ModelTrainer()
        
        with patch('lightgbm.train') as mock_train:
            mock_booster = MagicMock()
            mock_booster.predict.return_value = np.array([0.5] * 10)
            mock_train.return_value = mock_booster
            
            # Use assertLogs to capture logs from the root logger or specific logger
            # Assuming the code uses logging.info() or similar
            with self.assertLogs() as cm:
                trainer.optimize_hyperparameters(train_df, tune_df)
                
                # Check if logs contain expected info
                # We expect something like "Best parameters found: ..."
                self.assertTrue(any("Best parameters found" in output for output in cm.output), 
                                f"Logs did not contain 'Best parameters found'. Got: {cm.output}")

if __name__ == '__main__':
    unittest.main()
