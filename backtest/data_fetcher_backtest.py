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

# Configure logging at the very top
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Hardcoded Credentials
CT_APP_ID = "17481_ejoPRnjMkFdEkcZTHbjYt5n98n6wRE2wESCkSSHbLIvdWzRkRp"
CT_APP_SECRET = "AaIrnTNyz47CC9t5nsCXU67sCXtKOm7samSkpNFIvqKOaz1vJ1"
CT_ACCOUNT_ID = 45559627
CT_ACCESS_TOKEN = "nHEnxubVAIjSkisNbbPAd4gWCVrX0hG8aKbteZrX4GY"
CT_HOST_TYPE = "demo"

logger.info(f"Script Started - Account ID: {CT_ACCOUNT_ID}, Host: {CT_HOST_TYPE}")


# Asset Universe (same as training)
SYMBOL_IDS = {
    'EURUSD': 1,    # EUR/USD
    'GBPUSD': 2,    # GBP/USD
    'XAUUSD': 41,   # Gold/USD
    'USDCHF': 6,    # USD/CHF
    'USDJPY': 4     # USD/JPY
}

# Backtest Range: Jan 1, 2026 to Apr 9, 2026 (END_DATE is exclusive)
START_DATE = datetime(2026, 1, 1)
END_DATE = datetime(2026, 4, 10)  # Fetches until 2026-04-09 23:59:59
TIMEFRAME = ProtoOATrendbarPeriod.M5

class DataFetcherBacktest:
    def __init__(self):
        host = EndPoints.PROTOBUF_LIVE_HOST if CT_HOST_TYPE.lower() == "live" else EndPoints.PROTOBUF_DEMO_HOST
        port = EndPoints.PROTOBUF_PORT
        logger.info(f"Initializing Client for {host}:{port}")
        self.client = Client(host, port, TcpProtocol)
        self.request_delay = 1.0
        self.downloaded_data = {}
        self.data_dir = Path(__file__).resolve().parent / "data"

    def start(self):
        logger.info("Setting callbacks and starting service...")
        self.client.setConnectedCallback(self.on_connected)
        self.client.setDisconnectedCallback(self.on_disconnected)
        self.client.setMessageReceivedCallback(self.on_message)
        self.client.startService()
        logger.info("Reactor running...")
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
            acc_auth_req.ctidTraderAccountId = int(CT_ACCOUNT_ID)
            acc_auth_req.accessToken = CT_ACCESS_TOKEN
            res_auth = yield self.send_proto_request(acc_auth_req)
            payload_auth = Protobuf.extract(res_auth)
            logging.info(f"Account Auth Response: {payload_auth}")
            logging.info(f"Authenticated for Account ID: {CT_ACCOUNT_ID}")
            
            # 2a. Brief pause to stabilize connection
            d_wait = defer.Deferred()
            reactor.callLater(2.0, d_wait.callback, None)
            yield d_wait
            
            # 3. Fetch Backtesting Data (Continuous from 2025 up to today)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            fetch_tasks = []
            for i, (asset_name, symbol_id) in enumerate(SYMBOL_IDS.items()):
                if symbol_id == 0:
                    logging.warning(f"Skipping {asset_name}: Symbol ID not set.")
                    continue
                
                # We check for a consolidated backtest file
                fname = self.data_dir / f"{asset_name}_5m_backtest.parquet"
                if fname.exists():
                    df_check = pd.read_parquet(fname)
                    # If it ends within the last 24 hours of our target, we skip
                    if df_check.index[-1] >= (END_DATE - timedelta(days=1)):
                        logging.info(f"✅ Backtest data for {asset_name} is already up to date. Skipping.")
                        continue
                    
                # Delay each asset by 2.0s relative to previous to avoid rate limits
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
            
        logging.info(f"📥 Starting fetch for {asset_name} (ID: {symbol_id}) - UPDATING BACKTEST DATA")
        
        fname_final = self.data_dir / f"{asset_name}_5m_backtest.parquet"
        fname_2025 = self.data_dir / f"{asset_name}_5m_2025.parquet"
        
        all_bars = []
        current_start = START_DATE
        
        # Smart Resume Logic:
        # 1. Check if consolidated backtest file exists
        if fname_final.exists():
            existing_df = pd.read_parquet(fname_final)
            all_bars.append(existing_df)
            current_start = existing_df.index[-1].to_pydatetime()
            logging.info(f"   {asset_name}: Resuming from consolidated file (ends at {current_start})")
        # 2. Else check if legacy 2025 file exists to use as base
        elif fname_2025.exists():
            df_2025 = pd.read_parquet(fname_2025)
            all_bars.append(df_2025)
            current_start = df_2025.index[-1].to_pydatetime()
            logging.info(f"   {asset_name}: Found 2025 data. Resuming from {current_start}")
        
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
                        chunk_success = True
                        break
                    
                    bars_data = []
                    for bar in payload.trendbar:
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
                    time_index = pd.date_range(start=current_start, periods=len(df_chunk), freq='5min')
                    df_chunk.index = time_index
                    
                    all_bars.append(df_chunk)
                    logging.info(f"   {asset_name}: Fetched {len(df_chunk)} bars up to {chunk_end}")
                    
                    chunk_success = True
                    break 

                except Exception as e:
                    retry_count += 1
                    logging.error(f"⚠️ Error {asset_name} ({retry_count}/{max_retries}): {e}")
                    backoff_time = 2.0 ** retry_count
                    d_wait = defer.Deferred()
                    reactor.callLater(backoff_time, d_wait.callback, None)
                    yield d_wait
            
            if not chunk_success:
                logging.error(f"❌ Failed chunk for {asset_name} starting {current_start}.")
            
            current_start = chunk_end
        
        if all_bars:
            full_df = pd.concat(all_bars)
            full_df = full_df[~full_df.index.duplicated(keep='last')]
            full_df.sort_index(inplace=True)
            
            # Save to consolidated backtest file
            full_df.to_parquet(fname_final)
            logging.info(f"✅ Saved {fname_final}: {len(full_df)} rows.")
        else:
            logging.warning(f"⚠️ No data fetched for {asset_name}!")

if __name__ == "__main__":
    fetcher = DataFetcherBacktest()
    print(f"Starting Data Fetcher until {END_DATE.date()}...")
    try:
        fetcher.start()
    except KeyboardInterrupt:
        logging.info("Interrupted.")
