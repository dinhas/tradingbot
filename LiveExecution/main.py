import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from twisted.internet import reactor

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.config import load_config
from LiveExecution.src.logger import setup_logger
from LiveExecution.src.ctrader_client import CTraderClient
from LiveExecution.src.features import FeatureManager
from LiveExecution.src.models import ModelLoader
from LiveExecution.src.notifications import TelegramNotifier
from LiveExecution.src.orchestrator import Orchestrator
from LiveExecution.dashboard.main import DashboardServer

def main():
    # 1. Setup Environment & Logging
    load_dotenv()
    logger = setup_logger()
    logger.info("Starting Live Execution System...")

    # 2. Load Configuration
    try:
        config = load_config()
    except Exception as e:
        logger.critical(f"Configuration Error: {e}")
        return

    # 3. Initialize Components
    try:
        # Notifier
        notifier = TelegramNotifier(config)
        notifier.send_message("🟢 **System Starting Up...**")

        # Core Components
        client = CTraderClient(config)
        feature_manager = FeatureManager()
        model_loader = ModelLoader()
        
        # Load Models
        model_loader.load_all_models()

        # Orchestrator
        orchestrator = Orchestrator(client, feature_manager, model_loader, notifier, config=config)

        # Link notifier to orchestrator for commands
        notifier.set_orchestrator(orchestrator)

        # 3.5 Start Dashboard
        dashboard = DashboardServer(orchestrator)
        dashboard.start()

        # 4. Wiring
        # Connect Client Events to Orchestrator
        client.on_authenticated = orchestrator.bootstrap
        client.on_candle_closed = orchestrator.on_m5_candle_close
        client.on_order_execution = orchestrator.on_order_execution
        client.on_order_error = orchestrator.on_order_error

        # 5. Start Service
        client.start()
        
        # 6. Run Event Loop
        logger.info("Entering main event loop...")
        reactor.run()

    except Exception as e:
        logger.critical(f"Fatal Startup Error: {e}")
        if 'notifier' in locals():
            notifier.send_error(f"Fatal Startup Error: {e}")
        raise e

if __name__ == "__main__":
    main()
