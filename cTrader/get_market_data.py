import os
import json
import logging
from dotenv import load_dotenv
from twisted.internet import reactor
from ctrader_open_api import Client, TcpProtocol
from ctrader_open_api.endpoints import EndPoints
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOAGetTrendbarsReq,
    ProtoOAGetTrendbarsRes,
)
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv("CTRADER_CLIENT_ID")
CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN")
ACCOUNT_ID = os.getenv("CTRADER_ACCOUNT_ID")

# Define the root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# Create directories if they don't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "ctrader_market_data.log")),
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
    logging.info("Account authorized. Fetching market data...")
    # Load symbol from account_info.json
    try:
        account_info_path = os.path.join(LOGS_DIR, "account_info.json")
        with open(account_info_path, "r") as f:
            account_info = json.load(f)
            symbol_id = account_info.get("symbolId")
            if not symbol_id:
                logging.warning("symbolId not found in account_info.json, using placeholder 1 (EURUSD).")
                symbol_id = 1
    except FileNotFoundError:
        logging.warning("account_info.json not found. Using placeholder symbolId 1 (EURUSD).")
        symbol_id = 1

    # Request trendbars for the last 30 days
    now = datetime.utcnow()
    thirty_days_ago = now - timedelta(days=30)
    request = ProtoOAGetTrendbarsReq()
    request.ctidTraderAccountId = int(ACCOUNT_ID)
    request.symbolId = symbol_id
    request.period = ProtoOATrendbarPeriod.D1
    request.fromTimestamp = int(thirty_days_ago.timestamp() * 1000)
    request.toTimestamp = int(now.timestamp() * 1000)

    deferred = client.send(request)
    deferred.addCallbacks(on_market_data_response, on_error)

def on_market_data_response(response):
    """Callback for market data response."""
    market_data_res = ProtoOAGetTrendbarsRes()
    market_data_res.ParseFromString(response.payload)
    logging.info("Received market data.")

    trendbars = [{
        "timestamp": bar.utcTimestampInMinutes * 60 * 1000,
        "open": bar.low + bar.deltaOpen,
        "high": bar.low + bar.deltaHigh,
        "low": bar.low,
        "close": bar.low + bar.deltaClose,
        "volume": bar.volume
    } for bar in market_data_res.trendbar]

    market_data_path = os.path.join(LOGS_DIR, "market_data.json")
    with open(market_data_path, "w") as f:
        json.dump(trendbars, f, indent=4)
    logging.info(f"Market data saved to {market_data_path}")

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
    logging.info("Starting cTrader client for market data...")
    client.startService()
    reactor.run()

if __name__ == "__main__":
    main()