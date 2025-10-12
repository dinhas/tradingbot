import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from ctrader_open_api import Client, Protobuf, TcpProtocol
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiCommonModelMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# Load environment variables from .env file
load_dotenv()

# cTrader credentials and endpoint
CLIENT_ID = os.getenv("CTRADER_CLIENT_ID")
CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN")
ACCOUNT_ID = os.getenv("CTRADER_ACCOUNT_ID")
CTRADING_API_HOST = "demo.ctraderapi.com"
CTRADING_API_PORT = 5035

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

async def connect_and_fetch():
    """
    Connects to cTrader Open API, authenticates, and fetches account information.
    """
    logging.info("Starting connection to cTrader...")
    task_completed_event = asyncio.Event()

    client = Client(CTRADING_API_HOST, CTRADING_API_PORT, TcpProtocol)

    async def on_message(message: Protobuf):
        logging.info(f"Received message with payload type: {message.payloadType}")
        if message.payloadType == ProtoOAGetAccountListRes().payloadType:
            logging.info("Received account list response.")
            process_account_list(message, task_completed_event)
        elif message.payloadType == ProtoOAApplicationAuthRes().payloadType:
            logging.info("Application authenticated successfully.")
            await authorize_account(client)
        elif message.payloadType == ProtoOAAccountAuthRes().payloadType:
            logging.info("Account authenticated successfully.")
            await get_account_list(client)
        elif message.payloadType == ProtoHeartbeatEvent().payloadType:
            logging.info("Heartbeat received.")
        elif message.payloadType == ProtoErrorRes().payloadType:
            error_res = ProtoErrorRes()
            message.payload.Unpack(error_res)
            logging.error(f"An error occurred: {error_res.description} ({error_res.errorCode})")
            task_completed_event.set()

    client.setMessageReceivedCallback(on_message)

    def on_connect():
        logging.info("Connected to cTrader API.")
        asyncio.create_task(authorize_application(client))

    def on_error(reason):
        logging.error(f"Connection error: {reason}")
        task_completed_event.set()

    client.setConnectedCallback(on_connect)
    client.setDisconnectedCallback(on_error)

    # Start the client
    client.startService()
    try:
        await asyncio.wait_for(task_completed_event.wait(), timeout=30)
    except asyncio.TimeoutError:
        logging.error("Timed out waiting for a response from the server. This might be due to incorrect credentials or network issues.")
    finally:
        client.stopService()
        logging.info("Client stopped.")

async def authorize_application(client: Client):
    """
    Sends an application authorization request.
    """
    logging.info("Authorizing application...")
    request = ProtoOAApplicationAuthReq()
    request.clientId = CLIENT_ID
    request.clientSecret = CLIENT_SECRET
    await client.send(request)

async def authorize_account(client: Client):
    """
    Sends an account authorization request.
    """
    logging.info("Authorizing account...")
    request = ProtoOAAccountAuthReq()
    request.ctidTraderAccountId = int(ACCOUNT_ID)
    request.accessToken = ACCESS_TOKEN
    await client.send(request)

async def get_account_list(client: Client):
    """
    Sends a request to get the list of accounts.
    """
    logging.info("Fetching account list...")
    request = ProtoOAGetAccountListReq()
    request.accessToken = ACCESS_TOKEN
    await client.send(request)

def process_account_list(message: Protobuf, task_completed_event: asyncio.Event):
    """
    Processes the account list response and logs the details.
    """
    account_list_res = ProtoOAGetAccountListRes()
    message.payload.Unpack(account_list_res)

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
        logging.warning(f"The specified account ID ({ACCOUNT_ID}) was not found in the list of accounts returned by the server.")

    task_completed_event.set()

if __name__ == "__main__":
    if not all([CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID]):
        logging.error("Missing required environment variables. Please check your .env file.")
    else:
        try:
            asyncio.run(connect_and_fetch())
        except Exception as e:
            logging.error(f"An error occurred during the process: {e}")