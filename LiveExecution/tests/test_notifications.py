import unittest
from unittest.mock import MagicMock, patch, ANY
import json
import sys
from pathlib import Path
from twisted.internet.defer import Deferred
from twisted.web.client import Agent
from twisted.web.http_headers import Headers

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.notifications import DiscordNotifier

class TestDiscordNotifier(unittest.TestCase):
    def setUp(self):
        self.config = {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/test"}
        # Patch Agent at the source if possible, or just instantiate and replace
        self.notifier = DiscordNotifier(self.config)
        self.notifier.agent = MagicMock()

    def test_send_message_simple(self):
        """Test sending a simple text message."""
        d = Deferred()
        self.notifier.agent.request.return_value = d
        
        message = "System Started"
        self.notifier.send_message(message)
        
        self.notifier.agent.request.assert_called_once()
        args, kwargs = self.notifier.agent.request.call_args
        
        # Args: method, uri, headers, bodyProducer
        self.assertEqual(args[0], b'POST')
        self.assertEqual(args[1], self.config["DISCORD_WEBHOOK_URL"].encode('utf-8'))
        
        # Verify Headers (Content-Type: application/json)
        headers = args[2]
        self.assertIsInstance(headers, Headers)
        self.assertEqual(headers.getRawHeaders(b'Content-Type'), [b'application/json'])

    def test_send_trade_event(self):
        """Test sending a structured trade event."""
        d = Deferred()
        self.notifier.agent.request.return_value = d
        
        trade_details = {
            "symbol": "EURUSD",
            "action": "BUY",
            "size": 1000
        }
        self.notifier.send_trade_event(trade_details)
        
        # Just ensure it calls request
        self.notifier.agent.request.assert_called_once()

if __name__ == '__main__':
    unittest.main()
