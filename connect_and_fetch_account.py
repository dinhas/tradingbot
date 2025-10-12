import os
import json
import logging
from dotenv import load_dotenv
from twisted.internet import reactor
from ctrader_open_api import Client, Protobuf, TcpProtocol
from ctrader_open_api.endpoints import EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv("CTRADER_CLIENT_ID")
CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN")
ACCOUNT_ID = os.getenv("CTRADER_ACCOUNT_ID")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global state variables
client = None
is_connected = False

def on_error(failure):
    """Callback for handling errors."""
    logging.error(f"An error occurred: {failure}")
    if reactor.running:
        reactor.stop()

def on_connected(client_instance):
    """Callback for when the client connects."""
    global is_connected
    is_connected = True
    logging.info("Client connected. Authorizing application...")
    request = ProtoOAApplicationAuthReq()
    request.clientId = CLIENT_ID
    request.clientSecret = CLIENT_SECRET
    deferred = client_instance.send(request)
    deferred.addCallbacks(on_app_auth_response, on_error)

def on_disconnected(client_instance, reason):
    """Callback for when the client disconnects."""
    logging.info(f"Client disconnected: {reason}")

def on_app_auth_response(response):
    """Callback for application authentication response."""
    logging.info("Application authorized. Authorizing account...")
    request = ProtoOAAccountAuthReq()
    request.ctidTraderAccountId = int(ACCOUNT_ID)
    request.accessToken = ACCESS_TOKEN
    deferred = client.send(request)
    deferred.addCallbacks(on_account_auth_response, on_error)

def on_account_auth_response(response):
    """Callback for account authentication response."""
    logging.info("Account authorized. Fetching account list...")
    request = ProtoOAGetAccountListReq()
    request.accessToken = ACCESS_TOKEN
    deferred = client.send(request)
    deferred.addCallbacks(on_account_list_response, on_error)

def on_account_list_response(response):
    """Callback for account list response."""
    logging.info("Received account list.")
    account_list_res = ProtoOAGetAccountListRes()
    response.payload.Unpack(account_list_res)

    account_found = False
    for account in account_list_res.ctidTraderAccount:
        if account.ctidTraderAccountId == int(ACCOUNT_ID):
            account_info = {
                "accountId": account.ctidTraderAccountId,
                "balance": account.balance,
                "leverage": account.leverageInCents / 100,
                "marginLevel": account.marginLevel,
                "accountType": "real" if account.isLive else "demo"
            }
            logging.info(f"Account Info: {account_info}")

            # Log the full response to a JSON file
            with open("logs/account_info.json", "w") as f:
                json.dump(account_info, f, indent=4)
            logging.info("Account information saved to logs/account_info.json")
            logging.info("✅ Connected successfully to cTrader — Account info fetched.")
            account_found = True
            break

    if not account_found:
        logging.warning(f"The specified account ID ({ACCOUNT_ID}) was not found.")

    if reactor.running:
        reactor.stop()

def main():
    """Main function to start the client."""
    global client
    if not all([CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID]):
        logging.error("Missing required environment variables. Please check your .env file.")
        return

    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)

    # Create and configure the client
    host = EndPoints.PROTOBUF_DEMO_HOST
    client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
    client.setConnectedCallback(on_connected)
    client.setDisconnectedCallback(on_disconnected)

    # Start the client and the reactor
    logging.info("Starting cTrader client...")
    client.startService()

    # Add a connection timeout
    timeout_seconds = 20
    def connection_timeout():
        if not is_connected:
            logging.error(f"Connection timed out after {timeout_seconds} seconds. Please check network connectivity and credentials.")
            if reactor.running:
                reactor.stop()

    reactor.callLater(timeout_seconds, connection_timeout)
    reactor.run()

if __name__ == "__main__":
    main()