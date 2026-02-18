import os
import sys
import logging
import subprocess
import json
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

def run_threshold_optimization(logger):
    """Runs the threshold optimizer and returns the best parameters."""
    optimizer_path = Path(project_root) / "backtest" / "optimize_thresholds.py"
    results_path = Path(project_root) / "backtest" / "results" / "optimal_thresholds.json"
    
    logger.info("Starting Threshold Optimization (this may take a minute)...")
    try:
        # Run the optimizer as a subprocess
        # Use sys.executable to ensure we use the same python environment
        result = subprocess.run([sys.executable, str(optimizer_path)], check=True, capture_output=True, text=True, encoding='utf-8')
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                params = json.load(f)
            logger.info(f"Optimization Complete. Loaded Params: {params}")
            return params
        else:
            logger.warning("Optimization script finished but no JSON result found. Using defaults.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Threshold Optimization Subprocess Failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
    except Exception as e:
        logger.error(f"Threshold Optimization Failed: {e}")
    
    return None

def main():
    # 1. Setup Environment & Logging
    load_dotenv()
    logger = setup_logger()
    logger.info("Starting Live Execution System...")

    # 2. Run Threshold Optimization BEFORE loading components
    # This ensures we have the latest optimal thresholds for 2025
    auto_params = run_threshold_optimization(logger)

    # 3. Load Configuration
    try:
        config = load_config()
        # Inject optimized thresholds into config
        if auto_params:
            config['META_THRESHOLD'] = auto_params.get('meta_threshold', 0.80)
            config['QUAL_THRESHOLD'] = auto_params.get('qual_threshold', 0.35)
            config['RISK_THRESHOLD'] = auto_params.get('risk_threshold', 0.15)
        else:
            # Fallback defaults if optimization fails
            config.setdefault('META_THRESHOLD', 0.80)
            config.setdefault('QUAL_THRESHOLD', 0.35)
            config.setdefault('RISK_THRESHOLD', 0.15)

    except Exception as e:
        logger.critical(f"Configuration Error: {e}")
        return

    # 4. Initialize Components
    try:
        # Notifier
        notifier = TelegramNotifier(config)
        notifier.send_message("🟢 System Starting Up...")
        if auto_params:
            notifier.send_message(f"Thresholds Optimized:\nMeta: {config['META_THRESHOLD']}\nQual: {config['QUAL_THRESHOLD']}\nRisk: {config['RISK_THRESHOLD']}")

        # Core Components
        client = CTraderClient(config)
        feature_manager = FeatureManager()
        model_loader = ModelLoader()
        
        # Load Models
        if not model_loader.load_all_models():
            logger.critical("Failed to load models. System cannot proceed.")
            notifier.send_error("Failed to load models. System stopped.")
            return

        # Orchestrator
        orchestrator = Orchestrator(client, feature_manager, model_loader, notifier, config=config)

        # Link notifier to orchestrator for commands
        notifier.set_orchestrator(orchestrator)

        # 4.5 Start Dashboard
        dashboard = DashboardServer(orchestrator)
        dashboard.start()

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
        if 'notifier' in locals():
            notifier.send_error(f"Fatal Startup Error: {e}")
        raise e

if __name__ == "__main__":
    main()

