import unittest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

class TestModelLoading(unittest.TestCase):
    def test_model_files_exist(self):
        # Based on PRD and project structure
        alpha_path = os.path.join("checkpoints", "8.03.zip")
        risk_path = os.path.join("RiskLayer", "models", "2.15.zip")
        # TradeGuard path is "manual" in PRD, but let's assume manual_test_model.zip for now
        # OR just check what's in TradeGuard/models
        tg_path = os.path.join("TradeGuard", "models", "manual_test_model.zip")
        
        self.assertTrue(os.path.exists(alpha_path), f"Alpha model missing: {alpha_path}")
        self.assertTrue(os.path.exists(risk_path), f"Risk model missing: {risk_path}")
        self.assertTrue(os.path.exists(tg_path), f"TradeGuard model missing: {tg_path}")

    @patch('LiveExecution.src.models.PPO.load')
    def test_model_loader_logic(self, mock_ppo_load):
        from LiveExecution.src.models import ModelLoader
        loader = ModelLoader()
        
        # Mock load to return a dummy model
        mock_model = MagicMock()
        mock_ppo_load.return_value = mock_model
        
        success = loader.load_all_models()
        
        self.assertTrue(success)
        self.assertEqual(mock_ppo_load.call_count, 3)
        self.assertIsNotNone(loader.alpha_model)
        self.assertIsNotNone(loader.risk_model)
        self.assertIsNotNone(loader.tradeguard_model)

    @patch('LiveExecution.src.models.PPO.load')
    def test_predict_methods(self, mock_ppo_load):
        from LiveExecution.src.models import ModelLoader
        loader = ModelLoader()
        
        mock_model = MagicMock()
        mock_model.predict.return_value = (1, None)
        mock_ppo_load.return_value = mock_model
        
        loader.load_all_models()
        
        obs = [0] * 140 # Dummy obs
        action = loader.get_alpha_action(obs)
        self.assertEqual(action, 1)
        mock_model.predict.assert_called_with(obs, deterministic=True)

if __name__ == '__main__':
    unittest.main()
