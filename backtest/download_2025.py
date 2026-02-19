import logging
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# Load environment variables from .env
load_dotenv()

# --- Configuration ---
CT_APP_ID = os.getenv("CT_APP_ID")
CT_APP_SECRET = os.getenv("CT_APP_SECRET")
CT_ACCOUNT_ID = int(os.getenv("CT_ACCOUNT_ID", "0"))
CT_ACCESS_TOKEN = os.getenv("CT_ACCESS_TOKEN")
CT_HOST_TYPE = os.getenv("CT_HOST_TYPE", "demo")

SYMBOL_IDS = {
    'EURUSD': 1,
    'GBPUSD': 2,
    'XAUUSD': 41,
    'USDCHF': 6,
    'USDJPY': 4
}

START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 2, 1) # Just one month for testing
TIMEFRAME = ProtoOATrendbarPeriod.M5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataFetcherBacktest:
    def __init__(self):
        host = EndPoints.PROTOBUF_LIVE_HOST if CT_HOST_TYPE.lower() == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.request_delay = 0.5
        self.data_dir = Path("backtest/data")

    def start(self):
        self.client.setConnectedCallback(self.on_connected)
        self.client.setDisconnectedCallback(self.on_disconnected)
        self.client.setMessageReceivedCallback(self.on_message)
        self.client.startService()
        reactor.run()

    def on_disconnected(self, client, reason):
        logging.info(f"Disconnected: {reason}")
        try: reactor.stop()
        except: pass

    def on_message(self, client, message): pass

    def send_proto_request(self, request):
        return self.client.send(request)

    @inlineCallbacks
    def on_connected(self, client):
        logging.info("Connected to cTrader. Authenticating...")
        try:
            auth_req = ProtoOAApplicationAuthReq()
            auth_req.clientId = CT_APP_ID
            auth_req.clientSecret = CT_APP_SECRET
            yield self.send_proto_request(auth_req)

            acc_auth_req = ProtoOAAccountAuthReq()
            acc_auth_req.ctidTraderAccountId = CT_ACCOUNT_ID
            acc_auth_req.accessToken = CT_ACCESS_TOKEN
            yield self.send_proto_request(acc_auth_req)

            self.data_dir.mkdir(parents=True, exist_ok=True)

            for asset_name, symbol_id in SYMBOL_IDS.items():
                yield self.fetch_asset_history(asset_name, symbol_id)

            logging.info("All data downloads complete.")
            reactor.stop()
        except Exception as e:
            logging.error(f"Error: {e}")
            reactor.stop()

    @inlineCallbacks
    def fetch_asset_history(self, asset_name, symbol_id):
        logging.info(f"📥 Fetching {asset_name}...")
        current_start = START_DATE
        all_bars = []

        while current_start < END_DATE:
            chunk_end = current_start + timedelta(days=7)
            if chunk_end > END_DATE: chunk_end = END_DATE

            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = CT_ACCOUNT_ID
            req.symbolId = int(symbol_id)
            req.period = TIMEFRAME
            req.fromTimestamp = int(current_start.timestamp() * 1000)
            req.toTimestamp = int(chunk_end.timestamp() * 1000)

            try:
                res_msg = yield self.send_proto_request(req)
                payload = Protobuf.extract(res_msg)
                if hasattr(payload, 'trendbar') and payload.trendbar:
                    bars_data = []
                    for bar in payload.trendbar:
                        DIVISOR = 100000.0
                        low = bar.low / DIVISOR
                        bars_data.append({
                            'timestamp': datetime.fromtimestamp(bar.utcTimestampInMinutes * 60),
                            'open': low + (bar.deltaOpen / DIVISOR),
                            'high': low + (bar.deltaHigh / DIVISOR),
                            'low': low,
                            'close': low + (bar.deltaClose / DIVISOR),
                            'volume': bar.volume
                        })
                    df_chunk = pd.DataFrame(bars_data).set_index('timestamp')
                    all_bars.append(df_chunk)
                current_start = chunk_end
                # Small delay
                d = defer.Deferred()
                reactor.callLater(self.request_delay, d.callback, None)
                yield d
            except Exception as e:
                logging.error(f"Error fetching {asset_name}: {e}")
                break

        if all_bars:
            full_df = pd.concat(all_bars)
            full_df = full_df[~full_df.index.duplicated(keep='first')]
            fname = self.data_dir / f"{asset_name}_5m_2025.parquet"
            full_df.to_parquet(fname)
            logging.info(f"✅ Saved {fname}")

if __name__ == "__main__":
    fetcher = DataFetcherBacktest()
    fetcher.start()
