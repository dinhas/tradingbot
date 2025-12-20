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

    @patch('generate_dataset.DatasetGenerationEnv')
    @patch('stable_baselines3.PPO.load')
    def test_get_trade_signals(self, mock_ppo_load, mock_env_cls):
        """Test generating trade signals from data."""
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
            
        # Mock Model
        mock_model = MagicMock()
        mock_ppo_load.return_value = mock_model
        # Mock predict to return an action
        mock_model.predict.return_value = ([0]*20, None)
        
        # Mock Env
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        # Setup step return values: (obs, reward, done, truncated, info)
        # We simulate 2 steps: 1st continues, 2nd finishes
        mock_env.reset.return_value = ([0]*140, {})
        mock_env.step.side_effect = [
            ([0]*140, 0, False, False, {}),
            ([0]*140, 0, True, False, {})
        ]
        # Mock signals list
        mock_env.signals = [{'id': 1}, {'id': 2}]
        
        generator = self.DatasetGenerator()
        
        with patch('pathlib.Path.exists', return_value=True):
            # Patch load_data to return empty dict or mock data
            with patch.object(generator, 'load_data', return_value={}):
                signals = generator.generate_signals("dummy_model.zip")
            
        self.assertEqual(len(signals), 2)
        self.assertEqual(signals[0]['id'], 1)
        mock_env.reset.assert_called_once()
        self.assertEqual(mock_env.step.call_count, 2)

    def test_save_dataset(self):
        """Test saving the generated signals to a Parquet file."""
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
            
        generator = self.DatasetGenerator()
        
        # Mock signals data
        signals = [
            {'timestamp': '2024-01-01', 'asset': 'EURUSD', 'direction': 1, 'label': 1, 'features': [0.5]*60},
            {'timestamp': '2024-01-02', 'asset': 'GBPUSD', 'direction': -1, 'label': 0, 'features': [0.1]*60}
        ]
        
        output_path = Path("test_output.parquet")
        
        # If the method doesn't exist yet, it will fail
        try:
            generator.save_dataset(signals, output_path)
            self.assertTrue(output_path.exists())
            
            # Read back and verify
            df = pd.read_parquet(output_path)
            self.assertEqual(len(df), 2)
            self.assertIn('label', df.columns)
            self.assertEqual(len([c for c in df.columns if 'feature_' in c]), 60)
            
            # Cleanup
            output_path.unlink()
        except AttributeError:
            self.fail("DatasetGenerator has no method save_dataset")
        except Exception as e:
            self.fail(f"save_dataset failed: {e}")

class TestLabeling(unittest.TestCase):
    def setUp(self):
        try:
            from generate_dataset import DatasetGenerationEnv
            self.env_cls = DatasetGenerationEnv
        except ImportError:
            self.env_cls = None

    def test_label_long_win(self):
        """Test long trade hitting TP."""
        if self.env_cls is None: self.skipTest("Env not available")
        env = MagicMock()
        # Mock lookahead data
        future_highs = [1.1010, 1.1020, 1.1055] # Hits TP 1.1050
        future_lows = [1.0995, 1.0995, 1.0995]
        future_closes = [1.1005, 1.1015, 1.1050]
        
        # We'll implement this method in the env
        from generate_dataset import label_trade
        label = label_trade(1.1000, 1, 1.0990, 1.1050, future_highs, future_lows, future_closes)
        self.assertEqual(label, 1)

    def test_label_long_loss(self):
        """Test long trade hitting SL."""
        if self.env_cls is None: self.skipTest("Env not available")
        future_highs = [1.1010, 1.1010, 1.1010]
        future_lows = [1.0995, 1.0985, 1.0985] # Hits SL 1.0990
        future_closes = [1.1000, 1.0988, 1.0988]
        
        from generate_dataset import label_trade
        label = label_trade(1.1000, 1, 1.0990, 1.1050, future_highs, future_lows, future_closes)
        self.assertEqual(label, 0)

    def test_label_short_win(self):
        """Test short trade hitting TP."""
        if self.env_cls is None: self.skipTest("Env not available")
        # Entry 1.1000, TP 1.0950, SL 1.1010
        future_highs = [1.1005, 1.1005, 1.1005]
        future_lows = [1.0990, 1.0970, 1.0945] # Hits TP 1.0950
        future_closes = [1.0995, 1.0975, 1.0948]
        
        from generate_dataset import label_trade
        label = label_trade(1.1000, -1, 1.1010, 1.0950, future_highs, future_lows, future_closes)
        self.assertEqual(label, 1)

    def test_label_timeout_profit(self):
        """Test trade timing out with profit."""
        if self.env_cls is None: self.skipTest("Env not available")
        # Entry 1.1000, TP 1.1050, SL 1.0950
        future_highs = [1.1010] * 10
        future_lows = [1.0990] * 10
        future_closes = [1.0990] * 9 + [1.1005] # Closes slightly above entry
        
        from generate_dataset import label_trade
        label = label_trade(1.1000, 1, 1.0950, 1.1050, future_highs, future_lows, future_closes)
        self.assertEqual(label, 1)

if __name__ == '__main__':
    unittest.main()