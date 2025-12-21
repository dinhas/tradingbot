
import unittest
import os
import json
import tempfile
import shutil
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

# Mocking the models for testing
class MockAlphaModel:
    def predict(self, obs):
        return [0], None

class MockRiskModel:
    def predict(self, obs):
        return [np.zeros(3)], None

# We need to be able to import from the project root
import sys
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.backtest_full_system import load_tradeguard_model

class TestFullSystemInfra(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.test_dir, "guard_model.txt")
        self.metadata_path = os.path.join(self.test_dir, "model_metadata.json")
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_load_model_missing_files(self):
        """Test that missing files raise FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            load_tradeguard_model("non_existent.txt", self.metadata_path)
            
        with self.assertRaises(FileNotFoundError):
            # Create dummy model file first
            Path(self.model_path).touch()
            load_tradeguard_model(self.model_path, "non_existent.json")

    def test_load_invalid_metadata(self):
        """Test that invalid metadata raises ValueError or KeyError"""
        # Create dummy model
        data = np.random.rand(100, 10)
        label = np.random.randint(2, size=100)
        train_data = lgb.Dataset(data, label=label)
        model = lgb.train({}, train_data, 1)
        model.save_model(self.model_path)
        
        # Missing threshold
        with open(self.metadata_path, 'w') as f:
            json.dump({"metrics": {}}, f)
            
        with self.assertRaises(KeyError):
            load_tradeguard_model(self.model_path, self.metadata_path)
            
        # Invalid JSON
        with open(self.metadata_path, 'w') as f:
            f.write("invalid json")
            
        with self.assertRaises(ValueError):
            load_tradeguard_model(self.model_path, self.metadata_path)

    def test_successful_load(self):
        """Test successful loading of model and metadata"""
        # Create dummy model
        data = np.random.rand(100, 10)
        label = np.random.randint(2, size=100)
        train_data = lgb.Dataset(data, label=label)
        model = lgb.train({}, train_data, 1)
        model.save_model(self.model_path)
        
        # Valid metadata
        metadata = {"metrics": {"precision": 0.65}, "threshold": 0.55}
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        loaded_model, loaded_metadata = load_tradeguard_model(self.model_path, self.metadata_path)
        self.assertIsInstance(loaded_model, lgb.Booster)
        self.assertEqual(loaded_metadata['threshold'], 0.55)

if __name__ == "__main__":
    unittest.main()
