import unittest
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

class TestOptimization(unittest.TestCase):
    
    def setUp(self):
        try:
            from train_guard import ModelTrainer
            self.ModelTrainer = ModelTrainer
        except ImportError:
            self.ModelTrainer = None

    def test_optimize_hyperparameters(self):
        """Test that optimize_hyperparameters returns a dictionary of best params."""
        if self.ModelTrainer is None:
            self.fail("Could not import ModelTrainer from train_guard.py")
            
        # Create dummy data
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)], index=dates)
        y = pd.Series(np.random.randint(0, 2, 100), index=dates, name='label')
        
        # Combine for passing to trainer (assuming trainer expects DFs with label)
        train_df = X.copy()
        train_df['label'] = y
        
        tune_df = X.copy()
        tune_df['label'] = y
        
        trainer = self.ModelTrainer()
        
        # We verify that the method exists
        if not hasattr(trainer, 'optimize_hyperparameters'):
            self.fail("ModelTrainer does not have method 'optimize_hyperparameters'")
            
        # Mock lightgbm to avoid actual training time and dependency issues in unit test
        with patch('lightgbm.train') as mock_train:
            # Setup mock to return a dummy booster with a best_score or valid_0 score
            mock_booster = MagicMock()
            # Configure predict to return a numpy array of probabilities
            # matching the length of the tuning set (100 rows)
            mock_booster.predict.return_value = np.array([0.5] * 100)
            
            mock_train.return_value = mock_booster
            
            best_params = trainer.optimize_hyperparameters(train_df, tune_df)
            
            self.assertIsInstance(best_params, dict)
            self.assertIn('learning_rate', best_params)
            self.assertIn('num_leaves', best_params)
            
            # Verify lightgbm.train was called
            self.assertTrue(mock_train.called)

if __name__ == '__main__':
    unittest.main()
