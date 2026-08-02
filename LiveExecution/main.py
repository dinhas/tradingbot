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

from LiveExecution.src.config import load_config, get_thresholds
from LiveExecution.src.logger import setup_logger
from LiveExecution.src.ctrader_client import CTraderClient
from LiveExecution.src.features import FeatureManager
from LiveExecution.src.models import ModelLoader
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
        thresholds = get_thresholds(project_root)
        config["ALPHA_CONFIDENCE_THRESHOLD"] = thresholds["alpha_confidence_threshold"]
        config["FILTER_THRESHOLD"] = thresholds["filter_threshold"]
        config["SL_MULTIPLIER"] = thresholds["sl_multiplier"]
        config["TP_MULTIPLIER"] = thresholds["tp_multiplier"]

        logger.info(
            f"Thresholds: Alpha={config['ALPHA_CONFIDENCE_THRESHOLD']}, "
            f"Filter={config['FILTER_THRESHOLD']}, "
            f"SL={config['SL_MULTIPLIER']}x ATR, TP={config['TP_MULTIPLIER']}x ATR"
        )

    except Exception as e:
        logger.critical(f"Configuration Error: {e}")
        return

    # 3. Initialize Components
    try:
        client = CTraderClient(config)
        feature_manager = FeatureManager()
        model_loader = ModelLoader()

        # Apply thresholds to model loader
        model_loader.alpha_threshold = config["ALPHA_CONFIDENCE_THRESHOLD"]
        model_loader.filter_threshold = config["FILTER_THRESHOLD"]
        model_loader.sl_multiplier = config["SL_MULTIPLIER"]
        model_loader.tp_multiplier = config["TP_MULTIPLIER"]

        # Load Models
        if not model_loader.load_all_models():
            logger.critical("Failed to load models. System cannot proceed.")
            return

        # Orchestrator
        orchestrator = Orchestrator(
            client, feature_manager, model_loader, config=config
        )

        # Dashboard
        dashboard = DashboardServer(orchestrator)
        dashboard.start()
        orchestrator.set_dashboard(dashboard)

        # Start daily macro/COT refresh at midnight
        feature_manager.start_daily_refresh()

        # 4. Wiring
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
        raise e


if __name__ == "__main__":
    main()
