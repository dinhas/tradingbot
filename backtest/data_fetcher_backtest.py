import logging
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *
from dotenv import load_dotenv

load_dotenv()

CT_APP_ID = os.getenv("CT_APP_ID")
CT_APP_SECRET = os.getenv("CT_APP_SECRET")
CT_ACCOUNT_ID = int(os.getenv("CT_ACCOUNT_ID", 0))
CT_ACCESS_TOKEN = os.getenv("CT_ACCESS_TOKEN")
CT_HOST_TYPE = os.getenv("CT_HOST_TYPE", "demo")

# Asset Universe (same as training)
SYMBOL_IDS = {
    'EURUSD': 1,    # EUR/USD
    'GBPUSD': 2,    # GBP/USD
    'XAUUSD': 41,   # Gold/USD
    'USDCHF': 6,    # USD/CHF
    'USDJPY': 4     # USD/JPY
}

# Backtesting Data Range: 2025 (Jan 1 to Dec 14, 2025)
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 1, 20)  # Fetch until end of Dec 19
TIMEFRAME = ProtoOATrendbarPeriod.M5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataFetcherBacktest:
    def __init__(self):
        host = EndPoints.PROTOBUF_LIVE_HOST if CT_HOST_TYPE.lower() == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.request_delay = 1.0  # Increased delay to 1.0s to avoid rate limits
        self.downloaded_data = {}
        # Define a robust path to the backtest data directory at the project root
        # Define a robust path to the backtest data directory
        self.data_dir = Path(__file__).resolve().parent / "data"

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
            
            # 2a. Brief pause to stabilize connection
            d_wait = defer.Deferred()
            reactor.callLater(2.0, d_wait.callback, None)
            yield d_wait
            
            # 3. Fetch Backtesting Data (2025)
            # Ensure we write to Alpha/backtest/data
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            fetch_tasks = []
            # Use enumeration to stagger start times
            for i, (asset_name, symbol_id) in enumerate(SYMBOL_IDS.items()):
                if symbol_id == 0:
                    logging.warning(f"Skipping {asset_name}: Symbol ID not set. Please update SYMBOL_IDS.")
                    continue
                
                # Check if file already exists
                fname = self.data_dir / f"{asset_name}_5m_2025.parquet"
                if fname.exists():
                    logging.info(f"✅ Found existing backtest data for {asset_name}. Skipping download.")
                    continue
                    
                # Store task for parallel execution with staggered start
                # Delay each asset by 2.0s relative to previous
                d = self.fetch_asset_history(asset_name, symbol_id, initial_delay=i*2.0)
                fetch_tasks.append(d)
            
            if fetch_tasks:
                logging.info(f"🚀 Starting parallel fetch for {len(fetch_tasks)} assets (Staggered)...")
                yield defer.gatherResults(fetch_tasks, consumeErrors=True)
                
            logging.info("All backtest data downloads complete. Stopping reactor.")
            reactor.stop()

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            reactor.stop()

    @inlineCallbacks
    def fetch_asset_history(self, asset_name, symbol_id, initial_delay=0.0):
        # Initial stagger delay
        if initial_delay > 0:
            d_delay = defer.Deferred()
            reactor.callLater(initial_delay, d_delay.callback, None)
            yield d_delay
            
        logging.info(f"📥 Starting fetch for {asset_name} (ID: {symbol_id}) - 2025 BACKTEST DATA")
        
        current_start = START_DATE
        all_bars = []
        
        while current_start < END_DATE:
            # Request 5 days at a time to be SAFE from size limits
            chunk_end = current_start + timedelta(days=5)
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
                    # Rate limit delay (between chunks)
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
            
            # Move to next chunk
            current_start = chunk_end
        
        if all_bars:
            full_df = pd.concat(all_bars)
            full_df = full_df[~full_df.index.duplicated(keep='first')]
            
            # Save Backtest Data
            fname = self.data_dir / f"{asset_name}_5m_2025.parquet"
            full_df.to_parquet(fname)
            logging.info(f"✅ Saved {fname}: {len(full_df)} rows (2025 backtest data).")
        else:
            logging.warning(f"⚠️ No data fetched for {asset_name}!")

if __name__ == "__main__":
    fetcher = DataFetcherBacktest()
    print("Starting Backtest Data Fetcher for 2025... (Press Ctrl+C to stop manually)")
    try:
        fetcher.start()
    except KeyboardInterrupt:
        logging.info("Interrupted.")




