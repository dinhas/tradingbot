import json
import io
import logging
from twisted.internet import reactor
from twisted.web.client import Agent, FileBodyProducer
from twisted.web.http_headers import Headers

class DiscordNotifier:
    def __init__(self, config):
        self.webhook_url = config.get("DISCORD_WEBHOOK_URL")
        self.agent = Agent(reactor)
        self.logger = logging.getLogger("LiveExecution")

    def send_message(self, content):
        """Sends a simple text message to Discord."""
        if not self.webhook_url:
            self.logger.warning("Discord Webhook URL not set. Notification skipped.")
            return

        payload = {"content": content}
        return self._send_payload(payload)

    def send_trade_event(self, details):
        """Formats and sends a trade execution alert."""
        # Simple formatting for now. Can be upgraded to Embeds later.
        symbol = details.get('symbol', 'Unknown')
        action = details.get('action', 'Unknown')
        size = details.get('size', 0)
        
        msg = f"🚀 **TRADE EXECUTED**\n**Symbol:** {symbol}\n**Action:** {action}\n**Size:** {size}"
        return self.send_message(msg)

    def send_block_event(self, details):
        """Formats and sends a TradeGuard block alert."""
        symbol = details.get('symbol', 'Unknown')
        reason = details.get('reason', 'TradeGuard Filter')
        
        msg = f"🛡️ **TRADE BLOCKED**\n**Symbol:** {symbol}\n**Reason:** {reason}"
        return self.send_message(msg)

    def send_error(self, error_msg):
        """Sends an error alert."""
        msg = f"⚠️ **SYSTEM ERROR**\n{error_msg}"
        return self.send_message(msg)

    def _send_payload(self, payload):
        """Internal method to POST JSON payload."""
        body = json.dumps(payload).encode('utf-8')
        producer = FileBodyProducer(io.BytesIO(body))
        
        headers = Headers({'Content-Type': ['application/json']})
        
        d = self.agent.request(
            b'POST',
            self.webhook_url.encode('utf-8'),
            headers,
            producer
        )
        
        def handle_response(response):
            if response.code not in (200, 204):
                self.logger.error(f"Discord API returned status {response.code}")
            return response
        
        def handle_error(failure):
            self.logger.error(f"Failed to send Discord notification: {failure.getErrorMessage()}")
            return failure

        d.addCallback(handle_response)
        d.addErrback(handle_error)
        return d
