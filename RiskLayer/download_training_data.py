import logging
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from twisted.internet import reactor, defer
from twisted.internet.defer import inlineCallbacks
from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *

class Tee:
    """Redirect stdout/stderr to both console and file."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = TeeStderr(self.file, self.stderr)
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

class TeeStderr:
    """Handle stderr separately."""
    def __init__(self, file, stderr):
        self.file = file
        self.stderr = stderr
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stderr.write(text)
        self.stderr.flush()
    
    def flush(self):
        self.file.flush()
        self.stderr.flush()
        
# --- Configuration ---
CT_APP_ID = "17481_ejoPRnjMkFdEkcZTHbjYt5n98n6wRE2wESCkSSHbLIvdWzRkRp"
CT_APP_SECRET = "AaIrnTNyz47CC9t5nsCXU67sCXtKOm7samSkpNFIvqKOaz1vJ1"
CT_ACCOUNT_ID = 45587524
CT_ACCESS_TOKEN = "LdaPB-XNiM9fNkw53AqUXQU82xuq6vo5GVu9eYSEEDU"
CT_HOST_TYPE = "demo"

# Asset Universe
SYMBOL_IDS = {
    'EURUSD': 1,
    'GBPUSD': 2,
    'XAUUSD': 41,
    'USDCHF': 6,
    'USDJPY': 4
}

# Training Data Range: 2016 to 2024
START_DATE = datetime(2016, 1, 1)
END_DATE = datetime(2025, 1, 1) # Until start of 2025
TIMEFRAME = ProtoOATrendbarPeriod.M5

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataFetcherTraining:
    def __init__(self, output_dir="Alpha/data", force=False):
        host = EndPoints.PROTOBUF_LIVE_HOST if CT_HOST_TYPE.lower() == "live" else EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)
        self.request_delay = 0.25
        self.output_dir = output_dir
        self.force = force

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
            os.makedirs(self.output_dir, exist_ok=True)
            
            fetch_tasks = []
            for asset_name, symbol_id in SYMBOL_IDS.items():
                if symbol_id == 0: continue
                
                fname = os.path.join(self.output_dir, f"{asset_name}_5m.parquet") # Standard naming for training data
                if os.path.exists(fname) and not self.force:
                    logging.info(f"✅ Found existing data for {asset_name}. Skipping download.")
                    continue
                    
                d = self.fetch_asset_history(asset_name, symbol_id, fname)
                fetch_tasks.append(d)
            
            if fetch_tasks:
                logging.info(f"🚀 Starting parallel fetch for {len(fetch_tasks)} assets...")
                yield defer.gatherResults(fetch_tasks, consumeErrors=True)
            else:
                logging.info("No downloads needed.")
                
            logging.info("All data downloads complete. Stopping reactor.")
            reactor.stop()

        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            reactor.stop()

    @inlineCallbacks
    def fetch_asset_history(self, asset_name, symbol_id, output_file):
        logging.info(f"📥 Starting fetch for {asset_name} (ID: {symbol_id}) - 2016-2024")
        
        current_start = START_DATE
        all_bars = []
        
        while current_start < END_DATE:
            chunk_end = current_start + timedelta(days=90)
            if chunk_end > END_DATE:
                chunk_end = END_DATE
            
            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = CT_ACCOUNT_ID
            req.symbolId = int(symbol_id)
            req.period = TIMEFRAME
            req.fromTimestamp = int(current_start.timestamp() * 1000)
            req.toTimestamp = int(chunk_end.timestamp() * 1000)
            
            max_retries = 30  # Increased from 5 to 30 to handle flaky connections
            retry_count = 0
            chunk_success = False
            
            while retry_count < max_retries:
                try:
                    d = defer.Deferred()
                    # Exponential backoff with cap
                    delay = min(self.request_delay * (1.5 ** retry_count), 30.0)
                    reactor.callLater(delay, d.callback, None)
                    yield d
                    
                    res_msg = yield self.send_proto_request(req)
                    payload = Protobuf.extract(res_msg)
                    
                    if not hasattr(payload, 'trendbar') or not payload.trendbar:
                        # Empty data from server = Valid gap (e.g. holiday or no history)
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
                            'open': open_p, 'high': high, 'low': low, 'close': close, 'volume': bar.volume
                        })
                    
                    if not bars_data:
                        # Empty list = Valid gap
                        chunk_success = True
                        break

                    df_chunk = pd.DataFrame(bars_data)
                    time_index = pd.date_range(start=current_start, periods=len(df_chunk), freq='5min')
                    df_chunk.index = time_index
                    
                    all_bars.append(df_chunk)
                    logging.info(f"   {asset_name}: Fetched {len(df_chunk)} bars. Next: {chunk_end}")
                    chunk_success = True
                    break

                except Exception as e:
                    retry_count += 1
                    logging.error(f"⚠️ Error {asset_name} retry {retry_count}/{max_retries}: {e}")
                    
                    if retry_count >= max_retries:
                        logging.error(f"❌ CRITICAL FAIL: Could not fetch chunk {current_start}. Data will be MISSING.")
            
            if not chunk_success:
                logging.error(f"❌ GAP created at {current_start} for {asset_name} due to exhaust retries.")
            
            current_start = chunk_end
        
        if all_bars:
            full_df = pd.concat(all_bars)
            full_df = full_df[~full_df.index.duplicated(keep='first')]
            full_df.to_parquet(output_file)
            logging.info(f"✅ Saved {output_file}: {len(full_df)} rows.")
        else:
            logging.warning(f"⚠️ No data fetched for {asset_name}!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Force redownload even if files exist")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory (default: logs in script directory)")
    args = parser.parse_args()
    
    # Setup logging to file - capture all terminal output
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = args.log_dir if args.log_dir else os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"download_training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_file)
    
    try:
        print(f"All terminal output will be saved to: {log_file}")
        fetcher = DataFetcherTraining(output_dir=args.output, force=args.force)
        print("Starting Training Data Fetcher... (Press Ctrl+C to stop manually)")
        fetcher.start()
    except KeyboardInterrupt:
        logging.info("Interrupted.")
        print("Interrupted.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.error(f"CRITICAL ERROR: {e}")
        print(f"CRITICAL ERROR: {e}")
    finally:
        tee.close()
        logging.info(f"Log saved to: {log_file}")
        print(f"Log saved to: {log_file}")
