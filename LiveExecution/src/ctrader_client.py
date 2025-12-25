from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
import logging

class CTraderClient:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("LiveExecution")
        
        self.app_id = config["CT_APP_ID"]
        self.app_secret = config["CT_APP_SECRET"]
        self.account_id = config["CT_ACCOUNT_ID"]
        self.access_token = config["CT_ACCESS_TOKEN"]
        
        # Reconnection parameters
        self.max_retries = 5
        self.retry_count = 0
        self.base_delay = 5.0 # Seconds
        
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
    
    @inlineCallbacks
    def _on_connected(self, client):
        self.logger.info("Connected to cTrader. Authenticating...")
        self.retry_count = 0 # Reset retry count on successful connection
        
        try:
            # 1. Application Auth
            auth_req = ProtoOAApplicationAuthReq()
            auth_req.clientId = self.app_id
            auth_req.clientSecret = self.app_secret
            yield self.client.send(auth_req)
            self.logger.info("App Auth Success.")

            # 2. Account Auth
            acc_auth_req = ProtoOAAccountAuthReq()
            acc_auth_req.ctidTraderAccountId = self.account_id
            acc_auth_req.accessToken = self.access_token
            yield self.client.send(acc_auth_req)
            self.logger.info("Account Auth Success.")
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            self.stop()
        
    def _on_disconnected(self, client, reason):
        self.logger.warning(f"Disconnected from cTrader: {reason}")
        
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            delay = self.base_delay * (2 ** (self.retry_count - 1))
            self.logger.info(f"Reconnecting in {delay} seconds (Attempt {self.retry_count}/{self.max_retries})...")
            reactor.callLater(delay, self.start)
        else:
            self.logger.error("Max reconnection retries reached. Stopping system.")
            # In a real scenario, we might want to trigger a notification here
            self.stop()
        
    def _on_message(self, client, message):
        # Placeholder for message handling
        pass
        
    def stop(self):
        """Stops the client service."""
        self.client.stopService()
