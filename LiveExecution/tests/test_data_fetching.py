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
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

class TestDataFetching(unittest.TestCase):
    def setUp(self):
        self.config = {
            "CT_APP_ID": "test_app_id",
            "CT_APP_SECRET": "test_app_secret",
            "CT_ACCOUNT_ID": 12345,
            "CT_ACCESS_TOKEN": "test_token",
            "CT_HOST_TYPE": "demo"
        }

    @patch('LiveExecution.src.ctrader_client.Protobuf.extract')
    @patch('LiveExecution.src.ctrader_client.Client')
    def test_fetch_ohlcv(self, mock_client_cls, mock_extract):
        client = CTraderClient(self.config)
        mock_client_instance = mock_client_cls.return_value
        
        # Mock successful trendbar response
        res = ProtoOAGetTrendbarsRes()
        res.symbolId = 1
        res.period = ProtoOATrendbarPeriod.M5
        
        d = defer.Deferred()
        d.callback("dummy_msg")
        mock_client_instance.send.return_value = d
        mock_extract.return_value = res
        
        if hasattr(client, 'fetch_ohlcv'):
            # fetch_ohlcv is now @inlineCallbacks, returns a Deferred
            d_result = client.fetch_ohlcv(1, 100)
            
            # Verify send was called
            mock_client_instance.send.assert_called_once()
            req = mock_client_instance.send.call_args[0][0]
            self.assertIsInstance(req, ProtoOAGetTrendbarsReq)
            self.assertEqual(req.symbolId, 1)
        else:
            self.fail("CTraderClient does not have fetch_ohlcv method")

    @patch('LiveExecution.src.ctrader_client.Protobuf.extract')
    @patch('LiveExecution.src.ctrader_client.Client')
    def test_fetch_account_summary(self, mock_client_cls, mock_extract):
        client = CTraderClient(self.config)
        mock_client_instance = mock_client_cls.return_value
        
        res = ProtoOATraderRes()
        
        d = defer.Deferred()
        d.callback("dummy_msg")
        mock_client_instance.send.return_value = d
        mock_extract.return_value = res
        
        if hasattr(client, 'fetch_account_summary'):
            client.fetch_account_summary()
            mock_client_instance.send.assert_called_once()
            req = mock_client_instance.send.call_args[0][0]
            self.assertIsInstance(req, ProtoOATraderReq)
        else:
            self.fail("CTraderClient does not have fetch_account_summary method")

if __name__ == '__main__':
    unittest.main()
