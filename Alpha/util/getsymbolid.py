import logging
from pathlib import Path
from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# --- Configuration ---
CT_APP_ID = "17481_ejoPRnjMkFdEkcZTHbjYt5n98n6wRE2wESCkSSHbLIvdWzRkRp"
CT_APP_SECRET = "AaIrnTNyz47CC9t5nsCXU67sCXtKOm7samSkpNFIvqKOaz1vJ1"
CT_ACCOUNT_ID = 44663862  # Must be an integer
CT_ACCESS_TOKEN = "INnzhrurLIS2OSQDgzzckzZr1IbSf10VkS0sDx-cEVU"
CT_HOST_TYPE = "demo"     # "live" or "demo"

# Assets to find IDs for
ASSETS = {
    'Crypto': ['BTC', 'ETH', 'SOL'],
    'Forex': ['EUR', 'GBP', 'JPY']
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SymbolIdFetcher:
    def __init__(self):
        host = EndPoints.PROTOBUF_LIVE_HOST if CT_HOST_TYPE.lower() == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

    def start(self):
        self.client.setConnectedCallback(self.on_connected)
        self.client.setDisconnectedCallback(self.on_disconnected)
        self.client.setMessageReceivedCallback(self.on_message)
        self.client.startService()
        reactor.run()

    def on_disconnected(self, client, reason):
        logging.info(f"Disconnected: {reason}")
        try:
            reactor.stop()
        except:
            pass

    def on_message(self, client, message):
        pass

    def send_proto_request(self, request):
        return self.client.send(request)

    @inlineCallbacks
    def on_connected(self, client):
        logging.info("Connected to cTrader. Authenticating...")
        
        try:
            # 1. Application Auth
            auth_req = ProtoOAApplicationAuthReq()
            auth_req.clientId = CT_APP_ID
            auth_req.clientSecret = CT_APP_SECRET
            yield self.send_proto_request(auth_req)
            logging.info("App Auth Success.")

            # 2. Account Auth
            acc_auth_req = ProtoOAAccountAuthReq()
            acc_auth_req.ctidTraderAccountId = CT_ACCOUNT_ID
            acc_auth_req.accessToken = CT_ACCESS_TOKEN
            yield self.send_proto_request(acc_auth_req)
            logging.info("Account Auth Success.")
            
            # 3. Get Symbols Mapping
            logging.info("Fetching Symbols List...")
            symbols_req = ProtoOASymbolsListReq()
            symbols_req.ctidTraderAccountId = CT_ACCOUNT_ID
            symbols_req.includeArchivedSymbols = False
            
            symbols_response = yield self.send_proto_request(symbols_req)
            self.parse_symbols(symbols_response)
            
            logging.info("Done fetching IDs. You can now copy them to ctradercervice.py")
            self.client.stopService()

        except Exception as e:
            logging.error(f"Error: {e}")
            self.client.stopService()

    def parse_symbols(self, response_msg):
        logging.info(f"Response Message Type: {type(response_msg)}")
        # logging.info(f"Response Message: {response_msg}") # Uncomment if needed, might be huge

        payload = Protobuf.extract(response_msg)
        logging.info(f"Extracted Payload Type: {type(payload)}")
        
        if payload:
             logging.info(f"Payload fields: {payload.ListFields() if hasattr(payload, 'ListFields') else 'No ListFields'}")

        # Check if it's an error response (ProtoOAErrorRes has errorCode and description)
        if hasattr(payload, 'errorCode'):
            logging.error(f"Server returned an error: {payload.errorCode} - {payload.description}")
            if "authorized" in payload.description:
                logging.error("!!! PLEASE SET A VALID ACCESS TOKEN IN THE CONFIGURATION !!!")
            return

        if not hasattr(payload, 'symbol'):
            logging.error("Failed to parse symbols list. Payload is missing 'symbol' field.")
            if payload:
                logging.info(f"Payload content: {payload}")
            return

        logging.info(f"Received {len(payload.symbol)} symbols from server.")
        
        target_assets = ASSETS['Crypto'] + ASSETS['Forex']
        found_map = {}

        print("\n" + "="*50)
        print("FOUND SYMBOL IDS - COPY THESE TO CONFIG")
        print("="*50)
        
        output_path = Path(__file__).resolve().parent / "all_symbols.txt"
        with open(output_path, "w") as f:
            for s in payload.symbol:
                f.write(f"{s.symbolId}: {s.symbolName}\n")
                
                name = s.symbolName.upper()
                clean_name = name.replace("/", "").replace("-", "")
                
                for asset in target_assets:
                    target_clean = f"{asset}USD" if asset in ASSETS['Crypto'] else asset
                    if asset == 'USD/JPY': target_clean = 'USDJPY'
                    
                    if clean_name == target_clean:
                        print(f"'{asset}': {s.symbolId},  # {name}")
                        found_map[asset] = s.symbolId

        print("="*50 + "\n")

if __name__ == "__main__":
    fetcher = SymbolIdFetcher()
    print("Starting Symbol ID Fetcher...")
    try:
        fetcher.start()
    except KeyboardInterrupt:
        pass
