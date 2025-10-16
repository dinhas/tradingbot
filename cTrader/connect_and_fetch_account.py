import os
import json
import logging
from dotenv import load_dotenv
from twisted.internet import reactor
from ctrader_open_api import Client, Protobuf, TcpProtocol
from ctrader_open_api.endpoints import EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoErrorRes
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrader
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOATraderReq,
    ProtoOATraderRes
)

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv("CTRADER_CLIENT_ID")
CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN")
ACCOUNT_ID = os.getenv("CTRADER_ACCOUNT_ID")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../ctrader_connect.log"),
        logging.StreamHandler()
    ]
)

# Global client variable
client = None

def on_error(failure):
    """Callback for handling errors."""
    logging.error(f"An error occurred: {failure}")
    if reactor.running:
        reactor.stop()

def on_connected(client_instance):
    """Callback for when the client connects."""
    logging.info("Client connected. Authorizing application...")
    request = ProtoOAApplicationAuthReq()
    request.clientId = CLIENT_ID
    request.clientSecret = CLIENT_SECRET
    deferred = client_instance.send(request)
    deferred.addCallbacks(on_app_auth_response, on_error)

def on_disconnected(client_instance, reason):
    """Callback for when the client disconnects."""
    logging.info(f"Client disconnected: {reason}")
    if reactor.running:
        reactor.stop()

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
    logging.info("Account authorized. Fetching trader information...")
    request = ProtoOATraderReq()
    request.ctidTraderAccountId = int(ACCOUNT_ID)
    deferred = client.send(request)
    deferred.addCallbacks(on_trader_response, on_error)

def on_trader_response(response):
    """Callback for trader response."""
    logging.info("Received trader information.")

    trader_res = ProtoOATraderRes()
    trader_res.ParseFromString(response.payload)
    trader = trader_res.trader

    account_info = {
        "accountId": trader.ctidTraderAccountId,
        "balance": trader.balance,
        "leverage": trader.leverageInCents / 100
    }
    logging.info(f"Account Info: {account_info}")

    # Log the full response to a JSON file
    os.makedirs('cTrader/logs', exist_ok=True)
    with open("cTrader/logs/account_info.json", "w") as f:
        json.dump(account_info, f, indent=4)
    logging.info("Account information saved to cTrader/logs/account_info.json")
    logging.info("✅ Connected successfully to cTrader — Account info fetched.")

    if reactor.running:
        reactor.stop()

def main():
    """Main function to start the client."""
    global client
    if not all([CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID]):
        logging.error("Missing required environment variables. Please check your .env file.")
        return

    # Create and configure the client
    host = EndPoints.PROTOBUF_DEMO_HOST
    client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
    client.setConnectedCallback(on_connected)
    client.setDisconnectedCallback(on_disconnected)

    # Add a 60-second timeout as a safeguard
    reactor.callLater(60, on_error, "Timeout: The script ran for more than 60 seconds.")

    # Start the client and the reactor
    logging.info("Starting cTrader client...")
    client.startService()
    reactor.run()

if __name__ == "__main__":
    main()