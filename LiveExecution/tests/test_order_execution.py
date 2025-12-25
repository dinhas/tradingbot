import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path
from twisted.internet.defer import Deferred

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.ctrader_client import CTraderClient
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

class TestOrderExecution(unittest.TestCase):
    def setUp(self):
        self.config = {
            "CT_APP_ID": "test_app_id",
            "CT_APP_SECRET": "test_app_secret",
            "CT_ACCOUNT_ID": 12345,
            "CT_ACCESS_TOKEN": "test_token",
            "CT_HOST_TYPE": "demo"
        }
        self.client = CTraderClient(self.config)
        # Mock the internal twisted client
        self.client.client = MagicMock()

    def test_execute_market_order_structure(self):
        """Test that execute_market_order sends the correct ProtoOANewOrderReq."""
        symbol_id = 1
        volume = 1000
        side = ProtoOATradeSide.BUY
        
        # Mock the send method to return a Deferred
        d = Deferred()
        self.client.client.send.return_value = d
        
        # Call the method (which doesn't exist yet)
        self.client.execute_market_order(symbol_id, volume, side)
        
        # Verify the call
        self.client.client.send.assert_called_once()
        sent_msg = self.client.client.send.call_args[0][0]
        
        self.assertIsInstance(sent_msg, ProtoOANewOrderReq)
        self.assertEqual(sent_msg.ctidTraderAccountId, 12345)
        self.assertEqual(sent_msg.symbolId, symbol_id)
        self.assertEqual(sent_msg.volume, volume)
        self.assertEqual(sent_msg.tradeSide, side)
        self.assertEqual(sent_msg.orderType, ProtoOAOrderType.MARKET)

    def test_execute_market_order_with_sltp(self):
        """Test that SL and TP are correctly included in the request."""
        symbol_id = 1
        volume = 1000
        side = ProtoOATradeSide.SELL
        sl_price = 1.1000
        tp_price = 1.0500
        
        d = Deferred()
        self.client.client.send.return_value = d
        
        self.client.execute_market_order(symbol_id, volume, side, sl_price=sl_price, tp_price=tp_price)
        
        sent_msg = self.client.client.send.call_args[0][0]
        self.assertEqual(sent_msg.stopLoss, sl_price)
        self.assertEqual(sent_msg.takeProfit, tp_price)

if __name__ == '__main__':
    unittest.main()
