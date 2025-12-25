import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
from twisted.internet import reactor
from twisted.internet.defer import Deferred

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.ctrader_client import CTraderClient

class TestCTraderClient(unittest.TestCase):
    def setUp(self):
        self.config = {
            "CT_APP_ID": "test_app_id",
            "CT_APP_SECRET": "test_app_secret",
            "CT_ACCOUNT_ID": 12345,
            "CT_ACCESS_TOKEN": "test_token",
            "CT_HOST_TYPE": "demo"
        }
    
    @patch('LiveExecution.src.ctrader_client.Client')
    def test_client_initialization(self, mock_client_cls):
        client = CTraderClient(self.config)
        
        # Verify that the underlying ctrader-open-api Client was initialized
        # We expect it to be called with demo host for "demo" config
        mock_client_cls.assert_called_once()
        args, _ = mock_client_cls.call_args
        self.assertIn("demo.ctraderapi.com", args[0])
        
        # Verify attributes
        self.assertEqual(client.account_id, 12345)
        self.assertEqual(client.app_id, "test_app_id")

    @patch('LiveExecution.src.ctrader_client.Client')
    def test_start_service(self, mock_client_cls):
        mock_client_instance = mock_client_cls.return_value
        client = CTraderClient(self.config)
        
        client.start()
        
        mock_client_instance.startService.assert_called_once()
        mock_client_instance.setConnectedCallback.assert_called_once()
        mock_client_instance.setDisconnectedCallback.assert_called_once()
        mock_client_instance.setMessageReceivedCallback.assert_called_once()

if __name__ == '__main__':
    unittest.main()
