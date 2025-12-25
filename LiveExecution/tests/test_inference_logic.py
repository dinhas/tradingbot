import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

class TestInferenceLogic(unittest.TestCase):
    def test_logic_chain_flow(self):
        # We'll test an Orchestrator or similar class that manages the flow
        from LiveExecution.src.orchestrator import Orchestrator
        
        # Mock components
        mock_fm = MagicMock()
        mock_ml = MagicMock()
        mock_client = MagicMock()
        
        # Setup assets in fm
        mock_fm.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        mock_client.symbol_ids = {'EURUSD': 1}
        
        # Mock observations
        mock_fm.get_alpha_observation.return_value = np.array([0]*140)
        mock_fm.get_risk_observation.return_value = np.array([0]*165)
        mock_fm.get_tradeguard_observation.return_value = np.array([0]*105)
        
        # Mock history for price
        mock_fm.history = {'EURUSD': pd.DataFrame([{'close': 1.1}], index=[pd.Timestamp.now()])}
        
        # Mock model outputs
        mock_ml.get_alpha_action.return_value = np.array([1.0, 0, 0, 0, 0]) # Long for EURUSD
        mock_ml.get_risk_action.return_value = np.array([0.5, 0.1, 0.2]) # Size, SL, TP
        mock_ml.get_tradeguard_action.return_value = 1 # Allow
        
        orch = Orchestrator(mock_client, mock_fm, mock_ml)
        
        # Trigger inference for symbol 1 (EURUSD)
        decision = orch.run_inference_chain(1) 
        
        self.assertIsNotNone(decision)
        self.assertEqual(decision['asset'], 'EURUSD')
        self.assertEqual(decision['allowed'], True)
        
        # Verify sequential calls
        mock_ml.get_alpha_action.assert_called_once()
        mock_ml.get_risk_action.assert_called_once()
        mock_ml.get_tradeguard_action.assert_called_once()

if __name__ == '__main__':
    unittest.main()
