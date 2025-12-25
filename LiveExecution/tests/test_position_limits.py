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
        mock_notifier = MagicMock()
        
        # Mock asset list
        mock_fm.assets = ['EURUSD', 'GBPUSD']
        
        orch = Orchestrator(mock_client, mock_fm, mock_ml, mock_notifier)
        
        # 1. No active positions -> Not locked

if __name__ == '__main__':
    unittest.main()
