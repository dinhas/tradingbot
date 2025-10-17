import os
from dotenv import load_dotenv
from twisted.internet import reactor
from ctrader_open_api import Client, TcpProtocol
from ctrader_open_api.endpoints import EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq, ProtoOAAccountAuthReq
)
from utils.logger import system_logger as log

# Load environment variables
load_dotenv()

class CTraderClient:
    def __init__(self, host, port, client_id, client_secret, access_token, account_id):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.account_id = account_id

        self._client = Client(self.host, self.port, TcpProtocol)
        self._is_connected = False
        self._is_authorized = False
        self._auth_deferred = None
        self._message_listeners = []

        # Register callbacks
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._dispatch_message)

    def add_message_listener(self, listener):
        """Adds a listener for incoming messages."""
        if listener not in self._message_listeners:
            self._message_listeners.append(listener)
            log.info(f"Added message listener: {listener.__class__.__name__}")

    def _dispatch_message(self, client, message):
        """Dispatches incoming messages to all registered listeners."""
        # Always handle auth messages internally
        self._on_message_received(client, message)
        # Notify external listeners
        for listener in self._message_listeners:
            try:
                listener(client, message)
            except Exception as e:
                log.error(f"Error in message listener {listener}: {e}")


    def _on_connected(self, client):
        log.info("Successfully connected to cTrader.")
        self._is_connected = True
        self._authorize_app()

    def _on_disconnected(self, client, reason):
        log.warning(f"Disconnected from cTrader: {reason}. Attempting to reconnect...")
        self._is_connected = False
        self._is_authorized = False
        # You might want to add a reconnection logic here
        if reactor.running:
            reactor.stop() # For now, we stop the reactor

    def _on_message_received(self, client, message):
        # This will be expanded to handle various message types
        log.debug(f"Received message: {message.payloadType}")
        if message.payloadType == 2101: # ProtoOAApplicationAuthRes
             self._on_app_auth_response(message)
        elif message.payloadType == 2103: # ProtoOAAccountAuthRes
             self._on_account_auth_response(message)

    def _authorize_app(self):
        log.info("Authorizing application...")
        request = ProtoOAApplicationAuthReq()
        request.clientId = self.client_id
        request.clientSecret = self.client_secret
        self._client.send(request)

    def _on_app_auth_response(self, response):
        log.info("Application authorized. Authorizing account...")
        self._authorize_account()

    def _authorize_account(self):
        request = ProtoOAAccountAuthReq()
        request.ctidTraderAccountId = self.account_id
        request.accessToken = self.access_token
        self._client.send(request)

    def _on_account_auth_response(self, response):
        log.info(f"Account {self.account_id} authorized successfully.")
        self._is_authorized = True
        if self._auth_deferred:
            self._auth_deferred.callback(self)

    def connect(self):
        if not self._is_connected:
            log.info(f"Connecting to {self.host}:{self.port}...")
            self._client.startService()
            self._auth_deferred = reactor.deferred()
            return self._auth_deferred
        return reactor.deferred.succeed(self)

    def disconnect(self):
        if self._is_connected:
            log.info("Disconnecting from cTrader...")
            self._client.stopService()

    def send(self, message):
        if not self._is_authorized:
            raise ConnectionError("Client is not authorized.")
        return self._client.send(message)

def create_ctrader_client():
    """Factory function to create and configure a cTrader client."""
    client_id = os.getenv("CTRADER_CLIENT_ID")
    client_secret = os.getenv("CTRADER_CLIENT_SECRET")
    access_token = os.getenv("CTRADER_ACCESS_TOKEN")
    account_id = int(os.getenv("CTRADER_ACCOUNT_ID"))

    if not all([client_id, client_secret, access_token, account_id]):
        log.error("Missing required cTrader environment variables.")
        raise ValueError("cTrader API credentials not found in .env file.")

    host = EndPoints.PROTOBUF_DEMO_HOST
    port = EndPoints.PROTOBUF_PORT

    return CTraderClient(host, port, client_id, client_secret, access_token, account_id)

if __name__ == '__main__':
    # Example usage:
    log.info("Starting cTrader client for testing...")

    def on_auth_success(client_instance):
        log.info("Authentication sequence complete.")
        # Now you can send other requests
        # For example, let's disconnect after 5 seconds
        reactor.callLater(5, client_instance.disconnect)

    def on_auth_error(failure):
        log.error(f"Authentication failed: {failure}")
        if reactor.running:
            reactor.stop()

    client = create_ctrader_client()
    deferred = client.connect()
    deferred.addCallbacks(on_auth_success, on_auth_error)

    if not reactor.running:
        reactor.run()

    log.info("Reactor stopped.")