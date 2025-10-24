import os
import json
import sys
from datetime import datetime, timedelta
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Process
from twisted.internet import reactor, defer
from services.cTrader.client import create_ctrader_client
from services.market_data import MarketDataService
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATrendbarPeriod
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

async def fetch_ohlcv():
    client = None
    try:
        # Load config
        with open("config.json", "r") as f:
            config = json.load(f)

        # Create cTrader client
        client = create_ctrader_client()

        # Connect and authorize
        await client.connect()
        print("cTrader client connected and authorized.")

        # Create MarketDataService
        market_data_service = MarketDataService(client)

        # Fetch symbols to find EURUSD
        print("Fetching symbols to find EURUSD...")
        symbols_path = await market_data_service.get_symbols()

        with open(symbols_path, 'r') as f:
            symbols = json.load(f)

        eurusd_symbol = next((s for s in symbols if s['symbolName'] == 'EURUSD'), None)
        if not eurusd_symbol:
            print("EURUSD symbol not found.")
            return

        symbol_id = eurusd_symbol['symbolId']
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=180) # 6 months of data

        print(f"Fetching 5-minute OHLCV data for EURUSD (symbol ID: {symbol_id}) from {start_date} to {end_date}...")

        file_path = await market_data_service.get_trendbars(symbol_id, ProtoOATrendbarPeriod.M5, start_date, end_date)

        if file_path:
            print(f"OHLCV data saved to {file_path}")
        else:
            print("Could not fetch OHLCV data.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if client:
            client.disconnect()
            print("cTrader client disconnected.")
        if reactor.running:
            reactor.stop()

def run_fetch_process():
    deferred = defer.ensureDeferred(fetch_ohlcv())
    reactor.callWhenRunning(lambda: deferred)
    if not reactor.running:
        reactor.run()

if __name__ == "__main__":
    process = Process(target=run_fetch_process)
    process.start()
    process.join(timeout=300) # 5-minute timeout

    if process.is_alive():
        print("Script execution timed out after 300 seconds. Terminating...")
        process.terminate()
        process.join()
        sys.exit(1)

    if process.exitcode != 0:
        print(f"Script exited with error code: {process.exitcode}")
        sys.exit(process.exitcode)

    print("Script finished successfully.")
