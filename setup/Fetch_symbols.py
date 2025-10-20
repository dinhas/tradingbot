import os
import json
import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Process
from twisted.internet import reactor, defer
from services.cTrader.client import create_ctrader_client
from services.market_data import MarketDataService
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

async def fetch_symbols():
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

        # Fetch symbols
        print("Fetching symbols...")
        file_path = await market_data_service.get_symbols()
        if file_path:
            print(f"Symbols saved to {file_path}")
        else:
            print("Could not fetch symbols.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if client:
            client.disconnect()
            print("cTrader client disconnected.")
        if reactor.running:
            reactor.stop()

def run_fetch_process():
    deferred = defer.ensureDeferred(fetch_symbols())
    reactor.callWhenRunning(lambda: deferred)
    if not reactor.running:
        reactor.run()

if __name__ == "__main__":
    process = Process(target=run_fetch_process)
    process.start()
    process.join(timeout=120) # 120-second timeout

    if process.is_alive():
        print("Script execution timed out after 120 seconds. Terminating...")
        process.terminate()
        process.join()
        sys.exit(1)

    if process.exitcode != 0:
        print(f"Script exited with error code: {process.exitcode}")
        sys.exit(process.exitcode)

    print("Script finished successfully.")
