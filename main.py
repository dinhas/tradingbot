import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from twisted.internet import reactor
from twisted.web import server, resource

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.config import load_config, get_thresholds
from LiveExecution.src.logger import setup_logger
from LiveExecution.src.ctrader_client import CTraderClient
from LiveExecution.src.features import FeatureManager
from LiveExecution.src.models import ModelLoader
from LiveExecution.src.orchestrator import Orchestrator
from LiveExecution.dashboard.main import DashboardServer


# HTTP Resource for Back4App Health Check
class HealthCheckResource(resource.Resource):
    isLeaf = True

    def render_GET(self, request):
        request.setHeader(b"content-type", b"text/plain; charset=utf-8")
        return b"Trading Bot is live and running!"


def main():
    # 1. Setup Environment & Logging
    load_dotenv()
    logger = setup_logger()
    logger.info("Starting Live Execution System...")

    # 2. BIND HTTP HEALTH CHECK IMMEDIATELY
    # Opens port 8080 instantly so Back4App passes health checks while models load in background
    try:
        port = int(os.getenv("PORT", 8080))
        site = server.Site(HealthCheckResource())
        reactor.listenTCP(port, site)
        logger.info(f"Health check HTTP server listening on port {port}...")
    except Exception as e:
        logger.error(f"Could not bind health check port: {e}")

    # 3. Load Configuration
    try:
        config = load_config()
        # Load Thresholds (Centralized)
        thresholds = get_thresholds(project_root)
        config["ALPHA_CONFIDENCE_THRESHOLD"] = thresholds["alpha_confidence_threshold"]
        config["FILTER_THRESHOLD"] = thresholds["filter_threshold"]
        config["SL_MULTIPLIER"] = thresholds["sl_multiplier"]
        config["TP_MULTIPLIER"] = thresholds["tp_multiplier"]

        logger.info(
            f"Thresholds Loaded: Alpha={config['ALPHA_CONFIDENCE_THRESHOLD']}, "
            f"Filter={config['FILTER_THRESHOLD']}, "
            f"SL={config['SL_MULTIPLIER']}x ATR, TP={config['TP_MULTIPLIER']}x ATR"
        )

    except Exception as e:
        logger.critical(f"Configuration Error: {e}")
        return

    # 4. Initialize Components & Load Models
    try:
        # Core Components
        client = CTraderClient(config)
        feature_manager = FeatureManager()
        model_loader = ModelLoader()

        # Load Models
        if not model_loader.load_all_models():
            logger.critical("Failed to load models. System cannot proceed.")
            return

        # Wire thresholds from config into model loader
        model_loader.alpha_threshold = config.get("ALPHA_CONFIDENCE_THRESHOLD", 0.60)
        model_loader.filter_threshold = config.get("FILTER_THRESHOLD", 0.72)
        model_loader.sl_multiplier = config.get("SL_MULTIPLIER", 2.0)
        model_loader.tp_multiplier = config.get("TP_MULTIPLIER", 4.0)

        # Orchestrator
        orchestrator = Orchestrator(
            client, feature_manager, model_loader, config=config
        )

        # 4.5 Start Dashboard
        dashboard = DashboardServer(orchestrator)
        dashboard.start()
        orchestrator.set_dashboard(dashboard)

        # 5. Wiring
        # Connect Client Events to Orchestrator
        client.on_authenticated = orchestrator.bootstrap
        client.on_candle_closed = orchestrator.on_m5_candle_close
        client.on_order_execution = orchestrator.on_order_execution
        client.on_order_error = orchestrator.on_order_error

        # 6. Start Service
        client.start()

        # 7. Run Event Loop
        logger.info("Entering main event loop...")
        reactor.run()

    except Exception as e:
        logger.critical(f"Fatal Startup Error: {e}")
        raise e


if __name__ == "__main__":
    main()