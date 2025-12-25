from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
import logging

class CTraderClient:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("LiveExecution")
        
        self.app_id = config["CT_APP_ID"]
        self.app_secret = config["CT_APP_SECRET"]
        self.account_id = config["CT_ACCOUNT_ID"]
        self.access_token = config["CT_ACCESS_TOKEN"]
        
        # Determine host
        self.host = EndPoints.PROTOBUF_LIVE_HOST if config["CT_HOST_TYPE"] == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.port = EndPoints.PROTOBUF_PORT
        
        # Initialize client
        self.client = Client(self.host, self.port, TcpProtocol)
        
    def start(self):
        """Starts the Twisted client service and sets callbacks."""
        self.client.setConnectedCallback(self._on_connected)
        self.client.setDisconnectedCallback(self._on_disconnected)
        self.client.setMessageReceivedCallback(self._on_message)
        self.client.startService()
        self.logger.info(f"Connecting to cTrader ({self.host}:{self.port})...")
        
    def _on_connected(self, client):
        self.logger.info("Connected to cTrader.")
        
    def _on_disconnected(self, client, reason):
        self.logger.warning(f"Disconnected from cTrader: {reason}")
        
    def _on_message(self, client, message):
        # Placeholder for message handling
        pass
        
    def stop(self):
        """Stops the client service."""
        self.client.stopService()
