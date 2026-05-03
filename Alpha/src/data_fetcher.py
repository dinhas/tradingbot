import logging
import os
import argparse
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

# Load environment variables
load_dotenv()

# --- Configuration ---
CT_APP_ID = "17481_ejoPRnjMkFdEkcZTHbjYt5n98n6wRE2wESCkSSHbLIvdWzRkRp"
CT_APP_SECRET = "AaIrnTNyz47CC9t5nsCXU67sCXtKOm7samSkpNFIvqKOaz1vJ1"
CT_ACCOUNT_ID = 46341684
CT_ACCESS_TOKEN = "3BU4QPtH9lE2T3cdBP2az9xTIJYmHU-uPQEjLqp2wus"
CT_HOST_TYPE = "demo"

# Custom Asset Universe
SYMBOL_IDS = {
    'EURUSD': 1,    # EUR/USD
    'GBPUSD': 2,    # GBP/USD
    'XAUUSD': 41,   # Gold/USD
    'USDCHF': 6,    # USD/CHF
    'USDJPY': 4     # USD/JPY
}

# Mapping for timeframes
TIMEFRAME_MAP = {
    '1m': ProtoOATrendbarPeriod.M1,
    '5m': ProtoOATrendbarPeriod.M5,
    '30m': ProtoOATrendbarPeriod.M30,
    '1h': ProtoOATrendbarPeriod.H1,
    '4h': ProtoOATrendbarPeriod.H4,
    '1d': ProtoOATrendbarPeriod.D1
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataFetcher:
    def __init__(self, timeframe='30m', start_date='2016-01-01', end_date='2025-12-31'):
        host = EndPoints.PROTOBUF_LIVE_HOST if CT_HOST_TYPE.lower() == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.request_delay = 1.0  # Increased delay to avoid rate limits
        self.downloaded_data = {}
        # Define a robust path to the data directory at the project root
        self.data_dir = Path(__file__).resolve().parent.parent.parent / "data"
        
        self.timeframe_str = timeframe.lower()
        if self.timeframe_str not in TIMEFRAME_MAP:
            logging.warning(f"Unknown timeframe '{timeframe}', defaulting to '5m'")
            self.timeframe_str = '5m'
        
        self.timeframe = TIMEFRAME_MAP[self.timeframe_str]
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')

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
            acc_auth_res = yield self.send_proto_request(acc_auth_req)
            acc_auth_payload = Protobuf.extract(acc_auth_res)
            
            if acc_auth_payload.DESCRIPTOR.name == "ProtoOAErrorRes":
                logging.error(f"Account Auth Failed: {acc_auth_payload.errorCode} - {acc_auth_payload.description}")
                reactor.stop()
                return
            logging.info("Account Auth Success.")
            
            # 3. Fetch Data
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            fetch_tasks = []
            for asset_name, symbol_id in SYMBOL_IDS.items():
                if symbol_id == 0:
                    logging.warning(f"Skipping {asset_name}: Symbol ID not set. Please update SYMBOL_IDS.")
                    continue
                
                # Check if file already exists
                fname = self.data_dir / f"{asset_name}_{self.timeframe_str}.parquet"
                if fname.exists():
                    logging.info(f"✅ Found existing data for {asset_name} ({self.timeframe_str}). Skipping download.")
                    continue
                    
                # Store task for parallel execution
                d = self.fetch_asset_history(asset_name, symbol_id)
                fetch_tasks.append(d)
            
            if fetch_tasks:
                logging.info(f"🚀 Starting parallel fetch for {len(fetch_tasks)} assets ({self.timeframe_str})...")
                yield defer.gatherResults(fetch_tasks, consumeErrors=True)
                
            logging.info("All downloads complete. Stopping reactor.")
            reactor.stop()

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            reactor.stop()

    @inlineCallbacks
    def fetch_asset_history(self, asset_name, symbol_id):
        logging.info(f"📥 Starting fetch for {asset_name} (ID: {symbol_id}) at {self.timeframe_str}")
        
        # Define chunk sizes based on timeframe to avoid silent truncation (cTrader limits)
        CHUNK_LIMITS = {
            '1m': timedelta(days=14),
            '5m': timedelta(days=30),
            '30m': timedelta(days=60),
            '1h': timedelta(days=90),
            '4h': timedelta(days=300),
            '1d': timedelta(days=1000)
        }
        chunk_delta = CHUNK_LIMITS.get(self.timeframe_str, timedelta(days=30))

        current_end = self.end_date
        all_bars = []
        
        while current_end > self.start_date:
            current_start = current_end - chunk_delta
            if current_start < self.start_date:
                current_start = self.start_date
            
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = CT_ACCOUNT_ID
            req.symbolId = int(symbol_id)
            req.period = self.timeframe
            req.fromTimestamp = int(current_start.timestamp() * 1000)
            req.toTimestamp = int(current_end.timestamp() * 1000)
            
            # Retry loop for the current chunk
            max_retries = 5
            retry_count = 0
            chunk_success = False

            while retry_count < max_retries:
                try:
                    # Rate limit delay (5 req/sec limit)
                    d = defer.Deferred()
                    reactor.callLater(0.2, d.callback, None)
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
                        
                        # Use actual timestamp if available
                        timestamp = pd.to_datetime(bar.utcTimestampInMinutes, unit='m')
                        
                        bars_data.append({
                            'timestamp': timestamp,
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
                    df_chunk.set_index('timestamp', inplace=True)
                    
                    all_bars.append(df_chunk)
                    logging.info(f"   {asset_name}: Fetched {len(df_chunk)} bars. Range: {current_start} to {current_end}")
                    
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
            
            # Move to next chunk (walking backward)
            current_end = current_start
        
        if all_bars:
            full_df = pd.concat(all_bars)
            full_df = full_df[~full_df.index.duplicated(keep='first')]
            full_df.sort_index(inplace=True)
            
            # Save Raw Data
            fname = self.data_dir / f"{asset_name}_{self.timeframe_str}.parquet"
            full_df.to_parquet(fname)
            logging.info(f"✅ Saved {fname}: {len(full_df)} rows.")
        else:
            logging.warning(f"⚠️ No data fetched for {asset_name}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cTrader Data Fetcher")
    parser.add_argument("--timeframe", type=str, default="30m", choices=["1m", "5m", "30m", "1h", "4h", "1d"], help="Timeframe to fetch (1m, 5m, 30m, 1h, 4h, 1d)")
    parser.add_argument("--start", type=str, default="2016-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    fetcher = DataFetcher(timeframe=args.timeframe, start_date=args.start, end_date=args.end)
    print(f"Starting Data Fetcher for {args.timeframe} from {args.start} to {args.end}... (Press Ctrl+C to stop manually)")
    try:
        fetcher.start()
    except KeyboardInterrupt:
        logging.info("Interrupted.")
