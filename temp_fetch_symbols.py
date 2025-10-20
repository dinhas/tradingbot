import os
import json
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
        symbols = await market_data_service.get_symbols()
        print("Fetched Symbols:")
        for symbol in symbols:
            print(f"  Symbol ID: {symbol['symbolId']}, Name: {symbol['symbolName']}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if client:
            client.disconnect()
            print("cTrader client disconnected.")
        if reactor.running:
            reactor.stop()

if __name__ == "__main__":
    deferred = defer.ensureDeferred(fetch_symbols())
    # Add a timeout to the operation
    reactor.callLater(60, deferred.cancel)
    deferred.addErrback(lambda f: print(f"Operation timed out or failed: {f.value}"))

    reactor.callWhenRunning(lambda: deferred)
    if not reactor.running:
        reactor.run()
