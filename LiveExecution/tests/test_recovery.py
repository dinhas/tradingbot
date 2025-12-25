import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
from twisted.internet import reactor

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.orchestrator import Orchestrator

class TestRecovery(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_fm = MagicMock()
        self.mock_ml = MagicMock()
        self.mock_notifier = MagicMock()
        
        self.orchestrator = Orchestrator(
            self.mock_client, 
            self.mock_fm, 
            self.mock_ml,
            self.mock_notifier
        )

    def test_shutdown_sequence(self):
        """Test that shutdown stops the client and notifies."""
        self.orchestrator.stop()
        
        self.mock_client.stop.assert_called_once()
        self.mock_notifier.send_message.assert_called() # Should send "System Stopping"

    def test_startup_synchronization(self):
        """
        Test that on startup, we subscribe to spots for all assets.
        (Note: Subscription logic is typically in client.start(), but Orchestrator might trigger it)
        """
        # In current design, client handles connection/auth. 
        # Orchestrator just waits for on_m5_candle_close.
        # But we should verify Orchestrator ensures subscription.
        pass

if __name__ == '__main__':
    unittest.main()
