import logging
import os
import pandas as pd
from datetime import datetime, timedelta
from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

# --- Configuration ---
CT_APP_ID = "17481_ejoPRnjMkFdEkcZTHbjYt5n98n6wRE2wESCkSSHbLIvdWzRkRp"
CT_APP_SECRET = "AaIrnTNyz47CC9t5nsCXU67sCXtKOm7samSkpNFIvqKOaz1vJ1"
CT_ACCOUNT_ID = 44663862
CT_ACCESS_TOKEN = "INnzhrurLIS2OSQDgzzckzZr1IbSf10VkS0sDx-cEVU"
CT_HOST_TYPE = "demo"

# Custom Asset Universe (Modified from PRD)
# User selected forex pairs instead of PRD specification
SYMBOL_IDS = {
    'EURUSD': 1,    # EUR/USD
    'GBPUSD': 2,    # GBP/USD
    'XAUUSD': 41,   # Gold/USD
    'USDCHF': 6,    # USD/CHF
    'USDJPY': 4     # USD/JPY
}

# PRD 2.1 Timeframe & Data
START_DATE = datetime(2016, 1, 1)
END_DATE = datetime(2024, 12, 31)
TIMEFRAME = ProtoOATrendbarPeriod.M5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataFetcher:
    def __init__(self):
        host = EndPoints.PROTOBUF_LIVE_HOST if CT_HOST_TYPE.lower() == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.request_delay = 0.25
        self.downloaded_data = {}

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
            
            # 3. Fetch Data
            os.makedirs("data", exist_ok=True)
            
            fetch_tasks = []
            for asset_name, symbol_id in SYMBOL_IDS.items():
                if symbol_id == 0:
                    logging.warning(f"Skipping {asset_name}: Symbol ID not set. Please update SYMBOL_IDS.")
                    continue
                
                # Check if file already exists
                fname = f"data/{asset_name}_5m.parquet"
                if os.path.exists(fname):
                    logging.info(f"✅ Found existing data for {asset_name}. Skipping download.")
                    continue
                    
                # Store task for parallel execution
                d = self.fetch_asset_history(asset_name, symbol_id)
                fetch_tasks.append(d)
            
            if fetch_tasks:
                logging.info(f"🚀 Starting parallel fetch for {len(fetch_tasks)} assets...")
                yield defer.gatherResults(fetch_tasks, consumeErrors=True)
                
            logging.info("All downloads complete. Stopping reactor.")
            reactor.stop()

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            reactor.stop()

    @inlineCallbacks
    def fetch_asset_history(self, asset_name, symbol_id):
        logging.info(f"📥 Starting fetch for {asset_name} (ID: {symbol_id})")
        
        current_start = START_DATE
        all_bars = []
        
        while current_start < END_DATE:
            # Request 90 days at a time (approx 3 months)
            chunk_end = current_start + timedelta(days=90)
            if chunk_end > END_DATE:
                chunk_end = END_DATE
            
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = CT_ACCOUNT_ID
            req.symbolId = int(symbol_id)
            req.period = TIMEFRAME
            req.fromTimestamp = int(current_start.timestamp() * 1000)
            req.toTimestamp = int(chunk_end.timestamp() * 1000)
            
            # Retry loop for the current chunk
            max_retries = 5
            retry_count = 0
            chunk_success = False

            while retry_count < max_retries:
                try:
                    # Rate limit delay
                    d = defer.Deferred()
                    reactor.callLater(self.request_delay, d.callback, None)
                    yield d
                    
                    res_msg = yield self.send_proto_request(req)
                    payload = Protobuf.extract(res_msg)
                    
                    if not hasattr(payload, 'trendbar') or not payload.trendbar:
                        # No data in this chunk, but request was successful
                        chunk_success = True
                        break
                    
                    bars_data = []
                    for bar in payload.trendbar:
                        # cTrader Price = value / 100,000
                        DIVISOR = 100000.0
                        
                        low = bar.low / DIVISOR
                        open_p = (bar.low + bar.deltaOpen) / DIVISOR
                        high = (bar.low + bar.deltaHigh) / DIVISOR
                        close = (bar.low + bar.deltaClose) / DIVISOR
                        
                        bars_data.append({
                            'open': open_p, 
                            'high': high, 
                            'low': low, 
                            'close': close, 
                            'volume': bar.volume
                        })
                    
                    if not bars_data:
                        chunk_success = True
                        break

                    df_chunk = pd.DataFrame(bars_data)
                    
                    # Create time index (Approximation based on M5)
                    time_index = pd.date_range(start=current_start, periods=len(df_chunk), freq='5min')
                    df_chunk.index = time_index
                    
                    all_bars.append(df_chunk)
                    logging.info(f"   {asset_name}: Fetched {len(df_chunk)} bars. Next: {chunk_end}")
                    
                    chunk_success = True
                    break # Success, exit retry loop

                except Exception as e:
                    retry_count += 1
                    logging.error(f"⚠️ Error fetching chunk for {asset_name} (Attempt {retry_count}/{max_retries}): {e}")
                    
                    # Exponential backoff: 2s, 4s, 8s, 16s, 32s
                    backoff_time = 2.0 ** retry_count
                    logging.info(f"   Retrying in {backoff_time} seconds...")
                    
                    d_wait = defer.Deferred()
                    reactor.callLater(backoff_time, d_wait.callback, None)
                    yield d_wait
            
            if not chunk_success:
                logging.error(f"❌ Failed to fetch chunk starting {current_start} for {asset_name} after {max_retries} attempts. Skipping to next chunk.")
            
            # Move to next chunk regardless of success/failure to avoid infinite loop, 
            # but only after retries are exhausted or success achieved.
            current_start = chunk_end
        
        if all_bars:
            full_df = pd.concat(all_bars)
            full_df = full_df[~full_df.index.duplicated(keep='first')]
            
            # Save Raw Data
            fname = f"data/{asset_name}_5m.parquet"
            full_df.to_parquet(fname)
            logging.info(f"✅ Saved {fname}: {len(full_df)} rows.")
        else:
            logging.warning(f"⚠️ No data fetched for {asset_name}!")

if __name__ == "__main__":
    fetcher = DataFetcher()
    print("Starting Data Fetcher... (Press Ctrl+C to stop manually)")
    try:
        fetcher.start()
    except KeyboardInterrupt:
        logging.info("Interrupted.")
