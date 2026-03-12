import os
import json
from pathlib import Path
from dotenv import load_dotenv

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

DEFAULT_THRESHOLDS = {
    "meta_threshold": 0.7071,
    "qual_threshold": 0.7,
    "risk_threshold": 0.1,
}

def get_thresholds(project_root=None):
    """Loads thresholds from saved JSON file or returns defaults."""
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    
    results_path = Path(project_root) / "backtest" / "results" / "optimal_thresholds.json"

    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                params = json.load(f)
            return {
                "meta_threshold": params.get("meta_threshold", DEFAULT_THRESHOLDS["meta_threshold"]),
                "qual_threshold": params.get("qual_threshold", DEFAULT_THRESHOLDS["qual_threshold"]),
                "risk_threshold": params.get("risk_threshold", DEFAULT_THRESHOLDS["risk_threshold"]),
            }
        except Exception:
            return DEFAULT_THRESHOLDS
    return DEFAULT_THRESHOLDS

def load_config(override_env=None):
    """Loads and validates configuration from environment variables."""
    if override_env is None:
        # Load .env file if it exists
        load_dotenv()
        env = os.environ
    else:
        env = override_env
    
    required_vars = [
        "CT_APP_ID",
        "CT_APP_SECRET",
        "CT_ACCOUNT_ID",
        "CT_ACCESS_TOKEN",
        "TELEGRAM_BOT_TOKEN"
    ]
    
    config = {}
    
    # Check for missing variables
    missing = [var for var in required_vars if not env.get(var)]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")
    
    # Populate and validate
    config["CT_APP_ID"] = env.get("CT_APP_ID")
    config["CT_APP_SECRET"] = env.get("CT_APP_SECRET")
    config["CT_ACCESS_TOKEN"] = env.get("CT_ACCESS_TOKEN")
    config["TELEGRAM_BOT_TOKEN"] = env.get("TELEGRAM_BOT_TOKEN")
    config["TELEGRAM_CHAT_ID"] = env.get("TELEGRAM_CHAT_ID")
    
    # CT_ACCOUNT_ID must be an integer
    try:
        config["CT_ACCOUNT_ID"] = int(env.get("CT_ACCOUNT_ID"))
    except ValueError:
        raise ConfigError("CT_ACCOUNT_ID must be an integer.")
    
    # CT_HOST_TYPE defaults to 'demo'
    config["CT_HOST_TYPE"] = env.get("CT_HOST_TYPE", "demo").lower()
    if config["CT_HOST_TYPE"] not in ["demo", "live"]:
        raise ConfigError("CT_HOST_TYPE must be either 'demo' or 'live'.")

    # DB Path defaults
    config["DB_PATH"] = env.get("DB_PATH", "LiveExecution/data/live_trading.db")

    # Telegram Alert Thresholds
    config["TELEGRAM_DRAWDOWN_ALERT"] = float(env.get("TELEGRAM_DRAWDOWN_ALERT", "5.0"))
    config["TELEGRAM_PNL_MILESTONE"] = float(env.get("TELEGRAM_PNL_MILESTONE", "1.0"))
    config["TELEGRAM_PULSE_INTERVAL"] = int(env.get("TELEGRAM_PULSE_INTERVAL", "7200"))
    config["TELEGRAM_ERROR_ALERTS"] = env.get("TELEGRAM_ERROR_ALERTS", "true").lower() == "true"
        
    return config
