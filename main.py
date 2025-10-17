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
        """Stops the bot and disconnects."""
        log.info("Stopping the AI Trading Bot...")
        self.is_running = False
        if self.client:
            self.client.disconnect()
        if reactor.running:
            # This will stop the reactor, ending the script
            reactor.stop()

    def on_auth_success(self, client_instance):
        """Callback for successful authentication."""
        log.info("Authentication successful. Bot is now running.")
        self.is_running = True
        # Start the main decision loop
        self.main_loop()

    def on_auth_error(self, failure):
        """Callback for authentication failure."""
        log.error(f"Authentication failed: {failure}. Shutting down.")
        self.stop()

    def main_loop(self):
        """The main decision-making loop, compatible with Twisted."""
        if not self.is_running:
            log.info("Main loop stopped.")
            return

        try:
            log.info("Executing 5-minute decision cycle...")

            # This is a placeholder for the real logic.
            # In a real implementation, you would trigger data fetching,
            # AI analysis, and potential trade execution here.
            # For example:
            # self.market_data_service.get_symbols().addCallback(self.process_symbols)

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
            if not reactor.running:
                reactor.run()
        except Exception as e:
            log.error(f"An unhandled exception occurred: {e}")
        finally:
            log.info("Bot has been shut down.")
    else:
        log.error("Could not load configuration. Exiting.")