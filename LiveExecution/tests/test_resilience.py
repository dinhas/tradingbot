import unittest
from unittest.mock import MagicMock, patch, call
import sys
from pathlib import Path
from twisted.internet import defer

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.ctrader_client import CTraderClient

class TestResilience(unittest.TestCase):
    def setUp(self):
        self.config = {
            "CT_APP_ID": "test_app_id",
            "CT_APP_SECRET": "test_app_secret",
            "CT_ACCOUNT_ID": 12345,
            "CT_ACCESS_TOKEN": "test_token",
            "CT_HOST_TYPE": "demo"
        }

    @patch('LiveExecution.src.ctrader_client.reactor.callLater')
    @patch('LiveExecution.src.ctrader_client.Client')
    def test_reconnection_logic(self, mock_client_cls, mock_call_later):
        client = CTraderClient(self.config)
        client.start() # First start
        
        # Reset mock to track reconnection attempts
        mock_client_instance = mock_client_cls.return_value
        mock_client_instance.startService.reset_mock()
        
        # Simulate disconnection
        client._on_disconnected(mock_client_instance, "Connection lost")
        
        # Verify that callLater was called with a delay for reconnection
        mock_call_later.assert_called_once()
        delay, callback, *args = mock_call_later.call_args[0]
        self.assertGreater(delay, 0)
        self.assertEqual(callback, client.start)
        
        # Simulate reactor triggering the callback
        client.start()
        self.assertEqual(mock_client_instance.startService.call_count, 1)

    @patch('LiveExecution.src.ctrader_client.reactor.callLater')
    @patch('LiveExecution.src.ctrader_client.Client')
    def test_max_retries(self, mock_client_cls, mock_call_later):
        client = CTraderClient(self.config)
        mock_client_instance = mock_client_cls.return_value
        
        # Trigger disconnection 6 times (max is 5)
        for i in range(5):
            client._on_disconnected(mock_client_instance, "Fail")
            self.assertEqual(mock_call_later.call_count, i + 1)
            
        mock_call_later.reset_mock()
        client._on_disconnected(mock_client_instance, "Fail 6")
        
        # Should not call callLater again
        mock_call_later.assert_not_called()

if __name__ == '__main__':
    unittest.main()
