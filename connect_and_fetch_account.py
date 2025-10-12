import os
import asyncio
import logging
import json
import websockets
import ssl
import struct
import uuid
from dotenv import load_dotenv
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOAAccountType
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq, ProtoOAApplicationAuthRes,
    ProtoOAAccountAuthReq, ProtoOAAccountAuthRes,
    ProtoOAGetAccountListByAccessTokenReq, ProtoOAGetAccountListByAccessTokenRes,
    ProtoOAErrorRes
)
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import ProtoMessage, ProtoHeartbeatEvent

# Load environment variables from .env file
load_dotenv()

# cTrader credentials and account details from environment variables
CLIENT_ID = os.getenv("CTRADER_CLIENT_ID")
CLIENT_SECRET = os.getenv("CTRADER_CLIENT_SECRET")
ACCESS_TOKEN = os.getenv("CTRADER_ACCESS_TOKEN")
ACCOUNT_ID = int(os.getenv("CTRADER_ACCOUNT_ID"))

# cTrader API connection details
HOST = "demo.ctraderapi.com"
PORT = 5035

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

response_futures = {}

async def send_request(websocket, request, payload_type, client_msg_id=None):
    """Wraps a request in a ProtoMessage, serializes it, and sends it."""
    proto_message = ProtoMessage()
    proto_message.payloadType = payload_type
    proto_message.payload = request.SerializeToString()
    if client_msg_id:
        proto_message.clientMsgId = client_msg_id

    serialized_message = proto_message.SerializeToString()
    length_prefix = struct.pack('>I', len(serialized_message))

    logging.info(f"Sending message with payload type: {payload_type}")
    await websocket.send(length_prefix + serialized_message)
    logging.info("Message sent.")

async def consumer_handler(websocket):
    """Handles incoming messages from the server."""
    logging.info("Consumer handler started.")
    try:
        async for message in websocket:
            logging.info("Received a message from the server.")
            data = message[4:] # Skip the length prefix
            proto_message = ProtoMessage()
            proto_message.ParseFromString(data)

            if proto_message.HasField("clientMsgId") and proto_message.clientMsgId in response_futures:
                future = response_futures.pop(proto_message.clientMsgId)
                future.set_result(proto_message)
                logging.info(f"Matched response for clientMsgId: {proto_message.clientMsgId}")
            else:
                # Handle unsolicited messages
                if proto_message.payloadType == ProtoHeartbeatEvent().payloadType:
                    logging.info("Received heartbeat from server. Sending heartbeat back.")
                    await send_request(websocket, ProtoHeartbeatEvent(), ProtoHeartbeatEvent().payloadType)
                else:
                    logging.warning(f"Received unsolicited message of type {proto_message.payloadType}")
    except websockets.exceptions.ConnectionClosed as e:
        logging.error(f"Connection closed: {e.code} {e.reason}")
    finally:
        logging.info("Consumer handler finished.")


async def send_and_wait(websocket, request, payload_type):
    """Sends a request and waits for a response with a matching clientMsgId."""
    client_msg_id = str(uuid.uuid4())
    future = asyncio.get_running_loop().create_future()
    response_futures[client_msg_id] = future
    logging.info(f"Sending request with clientMsgId: {client_msg_id}")
    await send_request(websocket, request, payload_type, client_msg_id)
    return await asyncio.wait_for(future, timeout=10)


async def main():
    """Main function to connect to cTrader, authenticate, and fetch account info."""
    if not all([CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, ACCOUNT_ID]):
        logging.error("Missing required environment variables. Please check your .env file.")
        return

    ssl_context = ssl.create_default_context()
    uri = f"wss://{HOST}:{PORT}"
    consumer_task = None
    heartbeat_task = None

    try:
        async with websockets.connect(uri, ssl=ssl_context, ping_interval=None, close_timeout=10) as websocket:
            logging.info("Connected to cTrader API.")
            consumer_task = asyncio.create_task(consumer_handler(websocket))

            # Heartbeat task
            async def heartbeat(ws):
                while True:
                    await asyncio.sleep(15)
                    await send_request(ws, ProtoHeartbeatEvent(), ProtoHeartbeatEvent().payloadType)

            heartbeat_task = asyncio.create_task(heartbeat(websocket))

            # Authenticate application
            logging.info("Authenticating application...")
            app_auth_req = ProtoOAApplicationAuthReq(clientId=CLIENT_ID, clientSecret=CLIENT_SECRET)
            response_msg = await send_and_wait(websocket, app_auth_req, ProtoOAApplicationAuthReq().payloadType)
            if response_msg.payloadType != ProtoOAApplicationAuthRes().payloadType:
                logging.error(f"Application authentication failed. Received: {response_msg}")
                return
            logging.info("Application authenticated successfully.")

            # Authenticate account
            logging.info("Authorizing account...")
            acc_auth_req = ProtoOAAccountAuthReq(ctidTraderAccountId=ACCOUNT_ID, accessToken=ACCESS_TOKEN)
            response_msg = await send_and_wait(websocket, acc_auth_req, ProtoOAAccountAuthReq().payloadType)
            if response_msg.payloadType != ProtoOAAccountAuthRes().payloadType:
                logging.error(f"Account authorization failed. Received: {response_msg}")
                return
            logging.info("Account authorized successfully.")

            # Fetch account list
            logging.info("Fetching account list...")
            get_acc_list_req = ProtoOAGetAccountListByAccessTokenReq(accessToken=ACCESS_TOKEN)
            response_msg = await send_and_wait(websocket, get_acc_list_req, ProtoOAGetAccountListByAccessTokenReq().payloadType)

            if response_msg.payloadType == ProtoOAGetAccountListByAccessTokenRes().payloadType:
                get_acc_list_res = ProtoOAGetAccountListByAccessTokenRes()
                get_acc_list_res.ParseFromString(response_msg.payload)
                process_account_list(get_acc_list_res)
            elif response_msg.payloadType == ProtoOAErrorRes().payloadType:
                error_res = ProtoOAErrorRes()
                error_res.ParseFromString(response_msg.payload)
                logging.error(f"Received error: {error_res.description} [{error_res.errorCode}]")
            else:
                logging.error(f"Unknown response received. PayloadType: {response_msg.payloadType}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
        if consumer_task:
            consumer_task.cancel()


def process_account_list(account_list_response):
    """Processes and logs the account list response."""
    if not os.path.exists("logs"):
        os.makedirs("logs")
        logging.info("Created /logs directory.")

    account_list_data = []
    found_account = False
    for account in account_list_response.ctidTraderAccount:
        balance_in_units = account.balance / 100.0 if hasattr(account, 'balance') else 0
        account_data = {
            "accountId": account.ctidTraderAccountId,
            "balance": balance_in_units,
            "leverage": account.leverage if hasattr(account, 'leverage') else 'N/A',
            "accountType": ProtoOAAccountType.Name(account.accountType) if hasattr(account, 'accountType') else 'N/A',
            "isLive": account.isLive,
            "marginLevel": account.marginLevel if hasattr(account, 'marginLevel') else 'N/A'
        }
        account_list_data.append(account_data)

        if account.ctidTraderAccountId == ACCOUNT_ID:
            found_account = True
            logging.info("✅ Connected successfully to cTrader — Account info fetched.")
            logging.info(f"Account ID: {account_data['accountId']}")
            logging.info(f"Balance: {account_data['balance']}")
            logging.info(f"Leverage: {account_data['leverage']}")
            logging.info(f"Margin Level: {account_data.get('marginLevel', 'N/A')}")
            logging.info(f"Account Type: {'Live' if account_data['isLive'] else 'Demo'}")

    if not found_account:
        logging.warning(f"Account with ID {ACCOUNT_ID} not found in the response.")

    with open("logs/account_info.json", "w") as f:
        json.dump(account_list_data, f, indent=4)
    logging.info("Account information saved to logs/account_info.json")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user.")