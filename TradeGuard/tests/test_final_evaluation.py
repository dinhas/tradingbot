import unittest
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

class TestFinalEvaluation(unittest.TestCase):
    
    def setUp(self):
        try:
            from train_guard import ModelTrainer
            self.ModelTrainer = ModelTrainer
        except ImportError:
            self.ModelTrainer = None

    def test_train_final_model(self):
        """Test that final model training works with best params."""
        if self.ModelTrainer is None:
            self.fail("Could not import ModelTrainer from train_guard.py")
            
        trainer = self.ModelTrainer()
        
        # Create dummy data
        dates = pd.date_range(start='2016-01-01', periods=100, freq='D')
        df = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)], index=dates)
        df['label'] = np.random.randint(0, 2, 100)
        
        params = {'learning_rate': 0.05, 'num_leaves': 31, 'objective': 'binary'}
        
        with patch('lightgbm.train') as mock_train:
            mock_booster = MagicMock()
            mock_train.return_value = mock_booster
            
            model = trainer.train_final_model(df, params)
            
            self.assertEqual(model, mock_booster)
            self.assertTrue(mock_train.called)
            # Check that it was called with the correct parameters
            args, kwargs = mock_train.call_args
            self.assertEqual(args[0], params)

    def test_evaluate_model(self):
        """Test that evaluation returns a dictionary of metrics."""
        if self.ModelTrainer is None:
            self.fail("Could not import ModelTrainer from train_guard.py")
            
        trainer = self.ModelTrainer()
        
        # Create dummy val data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        df = pd.DataFrame(np.random.rand(100, 5), columns=[f'feat_{i}' for i in range(5)], index=dates)
        df['label'] = np.random.randint(0, 2, 100)
        
        mock_model = MagicMock()
        # Mock predictions (probabilities)
        mock_model.predict.return_value = np.array([0.1, 0.8] * 50)
        
        metrics = trainer.evaluate_model(mock_model, df, threshold=0.5)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('auc', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)

    def test_optimize_threshold(self):
        """Test that threshold optimization finds a threshold meeting precision target."""
        if self.ModelTrainer is None:
            self.fail("Could not import ModelTrainer from train_guard.py")
            
        trainer = self.ModelTrainer()
        
        # Create dummy val data where high prob means label 1
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        df = pd.DataFrame(index=dates)
        df['label'] = [0] * 50 + [1] * 50
        
        mock_model = MagicMock()
        # Mock predictions: 0.1 for first 50, 0.9 for last 50
        mock_model.predict.return_value = np.array([0.1] * 50 + [0.9] * 50)
        
        best_threshold, best_metrics = trainer.optimize_threshold(mock_model, df, target_precision=0.6)
        
        self.assertGreaterEqual(best_threshold, 0.0)
        self.assertLessEqual(best_threshold, 1.0)
        self.assertGreaterEqual(best_metrics['precision'], 0.6)

if __name__ == '__main__':
    unittest.main()
