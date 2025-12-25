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
from ctrader_open_api.messages.OpenApiMessages_pb2 import *

class TestAuth(unittest.TestCase):
    def setUp(self):
        self.config = {
            "CT_APP_ID": "test_app_id",
            "CT_APP_SECRET": "test_app_secret",
            "CT_ACCOUNT_ID": 12345,
            "CT_ACCESS_TOKEN": "test_token",
            "CT_HOST_TYPE": "demo"
        }
        
    @patch('LiveExecution.src.ctrader_client.Client')
    def test_authentication_flow(self, mock_client_cls):
        # Mock client instance and its send method
        mock_client_instance = mock_client_cls.return_value
        
        # Mock send to return a deferred that resolves immediately
        d = defer.Deferred()
        d.callback("success")
        mock_client_instance.send.return_value = d
        
        client = CTraderClient(self.config)
        
        # Manually trigger on_connected to test the auth flow
        # We need to yield this if it's a generator (inlineCallbacks)
        # But for unit testing without a reactor loop, we can inspect calls if we mock correctly.
        
        # We need to ensure _on_connected is wrapped or called in a way we can verify.
        # Since _on_connected uses inlineCallbacks, it returns a Deferred.
        
        # However, to test logic inside _on_connected, we can just call it and verify the calls to client.send
        
        # Mocking the client.send to return deferreds is key.
        
        # Manually trigger on_connected to test the auth flow
        try:
             client._on_connected(mock_client_instance)
        except Exception as e:
            pass
            
        # Check calls
        self.assertEqual(mock_client_instance.send.call_count, 2, "Expected 2 auth requests")
        
        # Verify App Auth
        args1, _ = mock_client_instance.send.call_args_list[0]
        req1 = args1[0]
        self.assertIsInstance(req1, ProtoOAApplicationAuthReq)
        self.assertEqual(req1.clientId, "test_app_id")
        self.assertEqual(req1.clientSecret, "test_app_secret")
        
        # Verify Account Auth
        args2, _ = mock_client_instance.send.call_args_list[1]
        req2 = args2[0]
        self.assertIsInstance(req2, ProtoOAAccountAuthReq)
        self.assertEqual(req2.ctidTraderAccountId, 12345)
        self.assertEqual(req2.accessToken, "test_token")

if __name__ == '__main__':
    unittest.main()
