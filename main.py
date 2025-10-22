import json
from twisted.internet import reactor
from services.cTrader.client import create_ctrader_client
from services.market_data import MarketDataService
from services.ai_engine import AIEngine
from services.risk_manager import RiskManager
from services.executor import TradeExecutor
from utils.logger import system_logger as log

class TradingBot:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.market_data_service = None
        self.ai_engine = None
        self.risk_manager = None
        self.executor = None
        self.is_running = False

    def start(self):
        """Initializes services and starts the bot."""
        log.info("Starting the AI Trading Bot...")

        # Initialize services
        self.client = create_ctrader_client()
        self.market_data_service = MarketDataService(self.client)
        self.ai_engine = AIEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        self.executor = TradeExecutor(self.client, self.config)

        # Connect to cTrader
        deferred = self.client.connect()
        deferred.addCallbacks(self.on_auth_success, self.on_auth_error)

    def stop(self):
        """Stops the bot and disconnects. This method is idempotent."""
        if not self.is_running:
            return  # Already stopping or stopped
        log.info("Stopping the AI Trading Bot...")
        self.is_running = False
        if self.client:
            self.client.disconnect()
        if reactor.running:
            # This will trigger the 'shutdown' event, which might call this method again.
            # The is_running flag prevents re-entry.
            reactor.stop()

    def on_auth_success(self, client_instance):
        """Callback for successful authentication."""
        log.info("Authentication successful. Bot is now running.")
        self.is_running = True
        # Subscribe to symbols from config
        self._subscribe_to_all_symbols()
        # Start the main decision loop
        self.main_loop()

    def on_auth_error(self, failure):
        """Callback for authentication failure."""
        log.error(f"Authentication failed: {failure}. Shutting down.")
        self.stop()

    def _subscribe_to_all_symbols(self):
        """Subscribes to spot data for all symbols in the config."""
        # This is a simplified approach. A robust implementation would
        # fetch all available symbols and match them against the config.
        # For this example, we'll assume a 'symbol_ids' dictionary in the config.
        symbol_ids = self.config.get("symbol_ids", {})
        if not symbol_ids:
            log.warning("No symbol_ids found in config.json. Cannot subscribe to symbols.")
            return

        for symbol_name, symbol_id in symbol_ids.items():
            log.info(f"Subscribing to {symbol_name} (ID: {symbol_id})")
            self.market_data_service.subscribe_to_spots(symbol_id)

    def main_loop(self):
        """The main decision-making loop, compatible with Twisted."""
        if not self.is_running:
            log.info("Main loop stopped.")
            return

        try:
            log.info("Executing 5-minute decision cycle...")

            for symbol_id, candle in self.market_data_service.live_candles.items():
                log.info(f"Latest candle for symbol {symbol_id}: {candle}")
                # Placeholder for AI decision logic
                # ai_decision = self.ai_engine.get_decision(candle)
                # if self.risk_manager.validate_trade(ai_decision):
                #     self.executor.execute_trade(ai_decision)
            log.info("AI decision is 'PASS' (placeholder). No action taken.")

        except Exception as e:
            log.error(f"Error in main loop: {e}")

        # Schedule the next run of the loop
        loop_interval_seconds = 300 # 5 minutes
        reactor.callLater(loop_interval_seconds, self.main_loop)


def load_config(file_path='config.json'):
    """Loads the configuration from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        log.error(f"Configuration file not found at {file_path}.")
        return None
    except json.JSONDecodeError:
        log.error(f"Error decoding JSON from {file_path}.")
        return None

if __name__ == "__main__":
    config = load_config()
    if config:
        bot = TradingBot(config)
        # Add a shutdown hook
        reactor.addSystemEventTrigger('before', 'shutdown', bot.stop)
        try:
            # Start the bot, which in turn will start the reactor if not running
            bot.start()
            # Schedule the bot to stop after 60 seconds
            reactor.callLater(60, bot.stop)
            if not reactor.running:
                reactor.run()
        except Exception as e:
            log.error(f"An unhandled exception occurred: {e}")
        finally:
            log.info("Bot has been shut down.")
    else:
        log.error("Could not load configuration. Exiting.")