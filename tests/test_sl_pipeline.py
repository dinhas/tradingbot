import unittest
import torch
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Alpha.src.supervised.model import MultiHeadModel
from Alpha.src.supervised.labeler import Labeler

class TestSLPipeline(unittest.TestCase):
    def test_model_forward(self):
        model = MultiHeadModel(input_dim=40)
        x = torch.randn(2, 40)
        d, q, m = model(x)
        self.assertEqual(d.shape, (2, 1))
        self.assertEqual(q.shape, (2, 1))
        self.assertEqual(m.shape, (2, 1))
        self.assertTrue(torch.all(d >= -1) and torch.all(d <= 1))
        self.assertTrue(torch.all(m >= 0) and torch.all(m <= 1))

    def test_labeler_basic(self):
        labeler = Labeler(time_barrier=5)
        # Create dummy raw data
        idx = pd.date_range("2023-01-01", periods=100, freq="5min")
        raw_df = pd.DataFrame({
            'EURUSD_close': np.linspace(1.0, 1.1, 100),
            'EURUSD_high': np.linspace(1.01, 1.11, 100),
            'EURUSD_low': np.linspace(0.99, 1.09, 100),
            'EURUSD_atr_14': np.ones(100) * 0.01
        }, index=idx)

        labels = labeler.label_data(raw_df, 'EURUSD')
        self.assertFalse(labels.empty)
        self.assertIn('direction', labels.columns)
        self.assertIn('quality_score', labels.columns)
        self.assertIn('meta_label', labels.columns)

if __name__ == "__main__":
    unittest.main()
