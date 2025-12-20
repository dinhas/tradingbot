import unittest
import sys
import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

class TestDatasetGeneratorInit(unittest.TestCase):
    def setUp(self):
        try:
            from generate_dataset import DatasetGenerator, FeatureEngine
            self.DatasetGenerator = DatasetGenerator
            self.FeatureEngine = FeatureEngine
        except ImportError:
            self.DatasetGenerator = None
            self.FeatureEngine = None

    def test_module_exists(self):
        """Test that the generate_dataset module and DatasetGenerator class exist."""
        if self.DatasetGenerator is None:
            self.fail("Could not import DatasetGenerator from generate_dataset")

    def test_initialization(self):
        """Test that the DatasetGenerator initializes correctly."""
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
        
        generator = self.DatasetGenerator(n_jobs=1)
        self.assertIsNotNone(generator.logger)
        self.assertTrue(isinstance(generator.logger, logging.Logger))
        self.assertEqual(generator.logger.name, "TradeGuard.DatasetGenerator")

    @patch('pathlib.Path.exists')
    @patch('pandas.read_parquet')
    def test_load_data(self, mock_read_parquet, mock_exists):
        """Test loading data for all assets."""
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
            
        mock_df = pd.DataFrame({
            'open': [1.1]*10, 'high': [1.2]*10, 'low': [1.0]*10, 
            'close': [1.1]*10, 'volume': [100]*10
        })
        mock_read_parquet.return_value = mock_df
        mock_exists.return_value = True
        
        generator = self.DatasetGenerator(n_jobs=1)
        generator.data_dir = Path("dummy/path")
        
        data = generator.load_data()
        
        self.assertEqual(len(data), 5)
        for asset in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']:
            self.assertIn(asset, data)
            
    def test_save_dataset(self):
        """Test saving the generated signals to a Parquet file."""
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
            
        generator = self.DatasetGenerator(n_jobs=1)
        signals = [
            {'timestamp': '2024-01-01', 'asset': 'EURUSD', 'label': 1, 'features': [0.5]*60},
            {'timestamp': '2024-01-02', 'asset': 'GBPUSD', 'label': 0, 'features': [0.1]*60}
        ]
        
        output_path = Path("test_output.parquet")
        if output_path.exists(): output_path.unlink()
        
        generator.save_dataset(signals, output_path)
        self.assertTrue(output_path.exists())
        
        df = pd.read_parquet(output_path)
        self.assertEqual(len(df), 2)
        self.assertIn('label', df.columns)
        self.assertEqual(len([c for c in df.columns if 'feature_' in c]), 60)
        
        output_path.unlink()

class TestFeatureEngine(unittest.TestCase):
    def setUp(self):
        try:
            from generate_dataset import FeatureEngine
            self.engine = FeatureEngine()
        except ImportError:
            self.engine = None

    def test_calculate_alpha_confidence(self):
        if self.engine is None: self.skipTest("FeatureEngine not available")
        portfolio_state = {
            'asset_action_raw': 0.5,
            'asset_recent_actions': [0.5, 0.4, 0.6],
            'asset_signal_persistence': 1.0,
            'asset_signal_reversal': 0.0,
            'equity': 10000,
            'peak_equity': 11000,
            'open_positions_count': 2,
            'total_exposure': 5000,
            'recent_trades': [{'pnl': 100}, {'pnl': -50}]
        }
        features = self.engine.calculate_alpha_confidence(None, portfolio_state)
        self.assertEqual(len(features), 10)
        self.assertIsInstance(features, list)

    def test_calculate_session_edge(self):
        if self.engine is None: self.skipTest("FeatureEngine not available")
        ts = pd.Timestamp('2024-01-01 14:30:00')
        features = self.engine.calculate_session_edge(ts)
        self.assertEqual(len(features), 10)

    def test_calculate_execution_stats(self):
        if self.engine is None: self.skipTest("FeatureEngine not available")
        trade_info = {'entry_price': 1.1000, 'sl': 1.0990, 'tp': 1.1020}
        portfolio_state = {'equity': 10000, 'peak_equity': 10000, 'position_value': 1000}
        features = self.engine.calculate_execution_stats(None, trade_info, portfolio_state, current_atr=0.0010)
        self.assertEqual(len(features), 10)

class TestDatasetGeneratorPrecompute(unittest.TestCase):
    def setUp(self):
        try:
            from generate_dataset import DatasetGenerator
            self.generator = DatasetGenerator(n_jobs=1)
        except ImportError:
            self.generator = None

    def test_precompute_market_features(self):
        if self.generator is None: self.skipTest("DatasetGenerator not available")
        
        # Create dummy data with enough rows for indicators (e.g. SMA 200)
        df = pd.DataFrame({
            'open': np.random.randn(250) + 1.1,
            'high': np.random.randn(250) + 1.2,
            'low': np.random.randn(250) + 1.0,
            'close': np.random.randn(250) + 1.1,
            'volume': np.random.randint(100, 1000, 250)
        }, index=pd.date_range('2024-01-01', periods=250, freq='5min'))
        
        df_dict = {'EURUSD': df}
        precomputed = self.generator.precompute_market_features(df_dict)
        
        self.assertIn('EURUSD', precomputed)
        # B=10, C=10, D=10, F=10 -> 40 features
        self.assertEqual(precomputed['EURUSD'].shape[1], 40)
        self.assertEqual(precomputed['EURUSD'].shape[0], 250)

if __name__ == '__main__':
    unittest.main()
