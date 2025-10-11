from ctrader_open_api import Client, Protobuf, TcpProtocol, Auth, EndPoints
from ctrader_open_api.messages import (
    OpenApiCommonMessages_pb2 as common_msgs,
    OpenApiMessages_pb2 as api_msgs,
    OpenApiModelMessages_pb2 as model_msgs,
)
from twisted.internet import reactor
from google.protobuf.json_format import MessageToDict
from dotenv import load_dotenv
import os
import json

# Construct the path to the .env file
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Load and validate environment variables
CLIENT_ID = os.getenv("CTRADER_APP_CLIENT_ID")
CLIENT_SECRET = os.getenv("CTRADER_APP_CLIENT_SECRET")
CTID_TRADER_ACCOUNT_ID = os.getenv("CTRADER_ACCOUNT_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

if not all([CLIENT_ID, CLIENT_SECRET, CTID_TRADER_ACCOUNT_ID, ACCESS_TOKEN]):
    raise ValueError("Please make sure all required environment variables are set in your .env file with the correct names.")

host = EndPoints.PROTOBUF_DEMO_HOST
client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

def onError(failure): # Call back for errors
    print("Message Error: ", failure)

from ctrader.market_data import get_symbols

def onProtoOAAccountAuthRes(client, message):
    print("ProtoOAAccountAuthRes: \n", Protobuf.extract(message))
    get_symbols(client, CTID_TRADER_ACCOUNT_ID, onError)

def onProtoOAApplicationAuthRes(client, message):
    print("ProtoOAApplicationAuthRes: \n", Protobuf.extract(message))
    request = api_msgs.ProtoOAAccountAuthReq()
    request.ctidTraderAccountId = int(CTID_TRADER_ACCOUNT_ID)
    request.accessToken = ACCESS_TOKEN
    deferred = client.send(request)
    deferred.addCallbacks(lambda msg: onProtoOAAccountAuthRes(client, msg), onError)


def connected(client): # Callback for client connection
    print("\nConnected")
    request = api_msgs.ProtoOAApplicationAuthReq()
    request.clientId = CLIENT_ID
    request.clientSecret = CLIENT_SECRET
    deferred = client.send(request)
    deferred.addCallbacks(lambda msg: onProtoOAApplicationAuthRes(client, msg), onError)

def disconnected(client, reason): # Callback for client disconnection
    print("\nDisconnected: ", reason)

from ctrader.market_data import onProtoOASymbolsListRes

message_handlers = {
    2101: onProtoOAApplicationAuthRes,
    2103: onProtoOAAccountAuthRes,
    2127: onProtoOASymbolsListRes
}

def onMessageReceived(client, message):
    if message.payloadType in message_handlers:
        message_handlers[message.payloadType](client, message)
    else:
        print("Message received: \n", Protobuf.extract(message))

# Setting optional client callbacks
client.setConnectedCallback(connected)
client.setDisconnectedCallback(disconnected)
client.setMessageReceivedCallback(onMessageReceived)
# Starting the client service
# client.startService()
# Run Twisted reactor
# reactor.callLater(30, reactor.stop)
# reactor.run()