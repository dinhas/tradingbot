import os
import sys
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from dotenv import load_dotenv
from twisted.internet import reactor

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


# Lightweight HTTP Health Check Server running on a separate thread
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        # Silence HTTP logs to keep console clean
        return


def start_health_check_server(port):
    server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
    server.serve_forever()


def main():
    # 1. INSTANT HEALTH CHECK START
    # Start HTTP server on a background thread so Back4App passes immediately
    port = int(os.getenv("PORT", 8080))
    health_thread = threading.Thread(
        target=start_health_check_server, args=(port,), daemon=True
    )
    health_thread.start()

    # 2. Setup Environment & Logging
    load_dotenv()
    logger = setup_logger()
    logger.info(f"Health check HTTP server active on port {port}...")
    logger.info("Starting Live Execution System...")

    # 3. Load Configuration
    try:
        config = load_config()
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

    # 4. Initialize Components & Heavy Model Loading
    try:
        client = CTraderClient(config)
        feature_manager = FeatureManager()
        model_loader = ModelLoader()

        # Load PyTorch / Scikit-Learn Models
        if not model_loader.load_all_models():
            logger.critical("Failed to load models. System cannot proceed.")
            return

        # Wire thresholds into model loader
        model_loader.alpha_threshold = config.get("ALPHA_CONFIDENCE_THRESHOLD", 0.60)
        model_loader.filter_threshold = config.get("FILTER_THRESHOLD", 0.72)
        model_loader.sl_multiplier = config.get("SL_MULTIPLIER", 2.0)
        model_loader.tp_multiplier = config.get("TP_MULTIPLIER", 4.0)

        # Orchestrator
        orchestrator = Orchestrator(
            client, feature_manager, model_loader, config=config
        )

        # Start Dashboard
        dashboard = DashboardServer(orchestrator)
        dashboard.start()
        orchestrator.set_dashboard(dashboard)

        # Wiring
        client.on_authenticated = orchestrator.bootstrap
        client.on_candle_closed = orchestrator.on_m5_candle_close
        client.on_order_execution = orchestrator.on_order_execution
        client.on_order_error = orchestrator.on_order_error

        # Start Client
        client.start()

        # 5. Run Twisted Event Loop
        logger.info("Entering main event loop...")
        reactor.run()

    except Exception as e:
        logger.critical(f"Fatal Startup Error: {e}")
        raise e


if __name__ == "__main__":
    main()