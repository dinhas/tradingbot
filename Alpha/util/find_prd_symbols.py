import logging
import json
import os
from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# --- Configuration ---
CT_APP_ID = "17481_ejoPRnjMkFdEkcZTHbjYt5n98n6wRE2wESCkSSHbLIvdWzRkRp"
CT_APP_SECRET = "AaIrnTNyz47CC9t5nsCXU67sCXtKOm7samSkpNFIvqKOaz1vJ1"
CT_ACCOUNT_ID = 45036604
CT_ACCESS_TOKEN = "HXfytBrtElk7sI3eIscePQTs5ZDxMmIlCSvXKY1Of8k"
CT_HOST_TYPE = "demo"

# Assets to find IDs for (Custom Configuration)
TARGET_ASSETS = ['EURUSD', 'GBPUSD', 'XAUUSD', 'USDCHF', 'USDJPY']

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
            
            self.client.stopService()

        except Exception as e:
            logging.error(f"Error: {e}")
            self.client.stopService()

    def parse_symbols(self, response_msg):
        payload = Protobuf.extract(response_msg)
        
        if not hasattr(payload, 'symbol'):
            logging.error("Failed to parse symbols list.")
            return

        logging.info(f"Received {len(payload.symbol)} symbols from server.")
        
        # Save all symbols to JSON
        all_symbols = []
        for s in payload.symbol:
            symbol_data = {
                'symbolId': s.symbolId,
                'symbolName': s.symbolName,
                'enabled': s.enabled if hasattr(s, 'enabled') else None,
                'baseAssetId': s.baseAssetId if hasattr(s, 'baseAssetId') else None,
                'quoteAssetId': s.quoteAssetId if hasattr(s, 'quoteAssetId') else None,
                'symbolCategoryId': s.symbolCategoryId if hasattr(s, 'symbolCategoryId') else None,
                'description': s.description if hasattr(s, 'description') else None
            }
            all_symbols.append(symbol_data)
            
        json_path = os.path.join(os.path.dirname(__file__), 'symbols.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(all_symbols, f, indent=4)
            logging.info(f"Saved {len(all_symbols)} symbols to {json_path}")
        except Exception as e:
            logging.error(f"Failed to save symbols.json: {e}")

        print("\n" + "="*50)
        print("FOUND SYMBOL IDS - UPDATE src/data_fetcher.py")
        print("="*50)
        
        found_count = 0
        for s in payload.symbol:
            # Normalize name for comparison (remove separators, uppercase)
            name = s.symbolName.upper()
            clean_name = name.replace("/", "").replace("-", "").replace("_", "")
            
            # Check against our target list
            for target in TARGET_ASSETS:
                target_clean = target.upper()
                
                # Direct match
                if clean_name == target_clean:
                    print(f"'{target}': {s.symbolId},  # {s.symbolName}")
                    found_count += 1
                    break

        print("="*50 + "\n")

if __name__ == "__main__":
    fetcher = SymbolIdFetcher()
    print("Starting Symbol Finder...")
    try:
        fetcher.start()
    except KeyboardInterrupt:
        pass
