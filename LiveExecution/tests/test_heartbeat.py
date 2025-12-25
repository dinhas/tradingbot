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
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *

class TestHeartbeat(unittest.TestCase):
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
    def test_heartbeat_loop(self, mock_client_cls, mock_call_later):
        client = CTraderClient(self.config)
        mock_client_instance = mock_client_cls.return_value
        
        # Manually start heartbeat
        client._start_heartbeat()
        
        # Verify callLater was called to schedule next heartbeat
        mock_call_later.assert_called_once()
        delay, callback, *args = mock_call_later.call_args[0]
        self.assertEqual(delay, 25) # Default cTrader heartbeat is usually 25s
        self.assertEqual(callback, client._send_heartbeat)
        
        # Verify heartbeat message sent
        client._send_heartbeat()
        self.assertEqual(mock_client_instance.send.call_count, 1)
        req = mock_client_instance.send.call_args[0][0]
        self.assertIsInstance(req, ProtoHeartbeatEvent)

if __name__ == '__main__':
    unittest.main()
