import unittest
from unittest.mock import MagicMock, patch, call
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.ctrader_client import CTraderClient
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *
from ctrader_open_api import Protobuf

class TestEventHandling(unittest.TestCase):
    def setUp(self):
        self.config = {
            "CT_APP_ID": "test_app_id",
            "CT_APP_SECRET": "test_app_secret",
            "CT_ACCOUNT_ID": 12345,
            "CT_ACCESS_TOKEN": "test_token",
            "CT_HOST_TYPE": "demo"
        }

    @patch('LiveExecution.src.ctrader_client.Client')
    def test_trendbar_event_listener(self, mock_client_cls):
        client = CTraderClient(self.config)
        client.on_candle_closed = MagicMock()
        
        # Create a mock ProtoOASpotEvent
        event = ProtoOASpotEvent()
        event.symbolId = 1 # EURUSD
        
        # Add a trendbar to the event
        trendbar = event.trendbar.add()
        trendbar.period = ProtoOATrendbarPeriod.M5
        
        with patch('LiveExecution.src.ctrader_client.Protobuf.extract') as mock_extract:
            mock_extract.return_value = event
            
            # Simulate receiving the message
            client._on_message(None, "dummy_payload")
            
            # Verify that on_candle_closed was called
            client.on_candle_closed.assert_called_once_with(1)

    @patch('LiveExecution.src.ctrader_client.Client')
    def test_ignore_non_m5_trendbar(self, mock_client_cls):
        client = CTraderClient(self.config)
        client.on_candle_closed = MagicMock()
        
        event = ProtoOASpotEvent()
        event.symbolId = 1
        trendbar = event.trendbar.add()
        trendbar.period = ProtoOATrendbarPeriod.M1 # NOT M5
        
        with patch('LiveExecution.src.ctrader_client.Protobuf.extract') as mock_extract:
            mock_extract.return_value = event
            client._on_message(None, "dummy_payload")
            client.on_candle_closed.assert_not_called()

if __name__ == '__main__':
    unittest.main()
