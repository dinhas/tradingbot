import unittest
from unittest.mock import Mock, MagicMock
from datetime import datetime
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.market_data import MarketDataService
from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASpotEvent

class TestMarketDataService(unittest.TestCase):
    def setUp(self):
        self.mock_client = Mock()
        self.mock_client.account_id = 12345  # Set an integer account_id
        self.market_data_service = MarketDataService(self.mock_client)

    def test_subscribe_to_spots(self):
        symbol_id = 1
        self.market_data_service.subscribe_to_spots(symbol_id)
        self.mock_client.send.assert_called_once()
        sent_request = self.mock_client.send.call_args[0][0]
        self.assertEqual(sent_request.ctidTraderAccountId, 12345)
        self.assertIn(symbol_id, sent_request.symbolId)

    def test_on_spot_event(self):
        # Create a mock spot event message with integer prices
        spot_event = ProtoOASpotEvent()
        spot_event.ctidTraderAccountId = 12345
        spot_event.symbolId = 1
        spot_event.bid = 112340
        spot_event.ask = 112360

        mock_message = MagicMock()
        mock_message.payload = spot_event.SerializeToString()

        # Simulate receiving the message
        self.market_data_service._on_spot_event(mock_message)

        # Check if a candle was created
        self.assertIn(1, self.market_data_service.live_candles)
        candle = self.market_data_service.live_candles[1]
        expected_price = 112350
        self.assertEqual(candle['open'], expected_price)
        self.assertEqual(candle['high'], expected_price)
        self.assertEqual(candle['low'], expected_price)
        self.assertEqual(candle['close'], expected_price)

        # Simulate another tick
        spot_event.bid = 112380
        spot_event.ask = 112400
        mock_message.payload = spot_event.SerializeToString()
        self.market_data_service._on_spot_event(mock_message)

        # Check if the candle was updated
        candle = self.market_data_service.live_candles[1]
        self.assertEqual(candle['high'], 112390)
        self.assertEqual(candle['close'], 112390)


if __name__ == '__main__':
    unittest.main()
