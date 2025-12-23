import unittest
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from TradeGuard.src.train_guard import TradeGuardTrainer
except ImportError:
    TradeGuardTrainer = None

class TestTrainGuard(unittest.TestCase):
    def setUp(self):
        self.config_path = "TradeGuard/config/ppo_config.yaml"
        self.dataset_path = 'TradeGuard/data/test_dataset.parquet'
        
        # Ensure test data exists
        if not os.path.exists(self.dataset_path):
            from TradeGuard.tests.generate_test_data import generate_test_dataset
            generate_test_dataset()

    def test_trainer_initialization(self):
        self.assertIsNotNone(TradeGuardTrainer, "TradeGuardTrainer not implemented yet")
        config_override = {'env': {'dataset_path': self.dataset_path}}
        trainer = TradeGuardTrainer(self.config_path, config_override=config_override)
        self.assertIsNotNone(trainer.config)
        self.assertIsNotNone(trainer.env)
        self.assertIsNotNone(trainer.model)

    def test_training_loop_mock(self):
        # Verify that the model.learn() can be called (with very few steps)
        config_override = {'env': {'dataset_path': self.dataset_path}}
        trainer = TradeGuardTrainer(self.config_path, config_override=config_override)
        # Override total_timesteps for a quick test
        trainer.train(total_timesteps=10)
        
        # Check if model is saved (optional/mocked)
        save_path = "TradeGuard/models/test_model.zip"
        trainer.save(save_path)
        self.assertTrue(os.path.exists(save_path))
        if os.path.exists(save_path):
            os.remove(save_path)

if __name__ == '__main__':
    unittest.main()
