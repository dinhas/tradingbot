import os
import logging
from twisted.internet import reactor
from dotenv import load_dotenv
from LiveExecution.src.ctrader_client import CTraderClient
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import ProtoOATradeSide

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ManualTradeTest")

def main():
    load_dotenv()
    
    # Validation
    if os.getenv("CT_HOST_TYPE") != "demo":
        logger.error("❌ SAFETY CHECK FAILED: CT_HOST_TYPE is not 'demo'.")
        logger.error("Please set CT_HOST_TYPE=demo in your .env file for this test.")
        return

    config = {
        "CT_APP_ID": os.getenv("CT_APP_ID"),
        "CT_APP_SECRET": os.getenv("CT_APP_SECRET"),
        "CT_ACCOUNT_ID": int(os.getenv("CT_ACCOUNT_ID")),
        "CT_ACCESS_TOKEN": os.getenv("CT_ACCESS_TOKEN"),
        "CT_HOST_TYPE": "demo"
    }

    client = CTraderClient(config)

    def trigger_trade():
        logger.info("⏳ Attempting to open a test trade (EURUSD, Buy, 0.01 lots / 1000 units)...")
        
        # EURUSD Symbol ID is usually 1
        symbol_id = 1 
        # 0.01 lots = 1,000 units. 
        # cTrader Open API expects volume in units * 100.
        volume = 100000 
        
        d = client.execute_market_order(symbol_id, volume, ProtoOATradeSide.BUY)
        
        def on_success(result):
            logger.info(f"✅ Trade Executed Successfully!")
            logger.info(f"Response: {result}")
            cleanup()
            
        def on_failure(failure):
            logger.error(f"❌ Trade Failed: {failure}")
            cleanup()

        d.addCallback(on_success)
        d.addErrback(on_failure)

    def cleanup():
        logger.info("Stopping reactor...")
        client.stop()
        if reactor.running:
            reactor.stop()

    # Start client
    client.start()

    # Schedule trade execution after 5 seconds to allow for connection & auth
    reactor.callLater(5, trigger_trade)
    
    # Timeout after 15 seconds
    reactor.callLater(15, lambda: logger.error("Timeout reached") or cleanup())

    logger.info("🚀 Starting Trade Test... (Ctrl+C to cancel)")
    reactor.run()

if __name__ == "__main__":
    main()
