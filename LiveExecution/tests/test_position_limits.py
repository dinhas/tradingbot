import unittest
from unittest.mock import MagicMock
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.orchestrator import Orchestrator

class TestPositionLimits(unittest.TestCase):
    def test_asset_locking(self):
        mock_client = MagicMock()
        mock_fm = MagicMock()
        mock_ml = MagicMock()
        
        orch = Orchestrator(mock_client, mock_fm, mock_ml)
        
        # Simulate having an open position for EURUSD (symbolId 1)
        orch.active_positions = {1: {'id': 'pos1', 'size': 1000}}
        
        # Check if asset is locked
        locked = orch.is_asset_locked(1)
        self.assertTrue(locked)
        
        # Check for symbol without position
        locked = orch.is_asset_locked(2)
        self.assertFalse(locked)

if __name__ == '__main__':
    unittest.main()
