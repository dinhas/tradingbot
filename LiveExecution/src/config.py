import os
import json
from pathlib import Path
from dotenv import load_dotenv

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

DEFAULT_THRESHOLDS = {
    "alpha_confidence_threshold": 0.60,
    "filter_threshold": 0.72,
    "sl_multiplier": 2.0,
    "tp_multiplier": 4.0,
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
                "alpha_confidence_threshold": params.get("alpha_confidence_threshold", DEFAULT_THRESHOLDS["alpha_confidence_threshold"]),
                "filter_threshold": params.get("filter_threshold", DEFAULT_THRESHOLDS["filter_threshold"]),
                "sl_multiplier": params.get("sl_multiplier", DEFAULT_THRESHOLDS["sl_multiplier"]),
                "tp_multiplier": params.get("tp_multiplier", DEFAULT_THRESHOLDS["tp_multiplier"]),
            }
        except Exception:
            return DEFAULT_THRESHOLDS
    return DEFAULT_THRESHOLDS

def load_config(override_env=None):
    """Loads and validates configuration from environment variables."""
    if override_env is None:
        load_dotenv()
        env = os.environ
    else:
        env = override_env

    required_vars = [
        "CT_APP_ID",
        "CT_APP_SECRET",
        "CT_ACCOUNT_ID",
        "CT_ACCESS_TOKEN",
    ]

    config = {}

    missing = [var for var in required_vars if not env.get(var)]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")

    config["CT_APP_ID"] = env.get("CT_APP_ID")
    config["CT_APP_SECRET"] = env.get("CT_APP_SECRET")
    config["CT_ACCESS_TOKEN"] = env.get("CT_ACCESS_TOKEN")

    try:
        config["CT_ACCOUNT_ID"] = int(env.get("CT_ACCOUNT_ID"))
    except ValueError:
        raise ConfigError("CT_ACCOUNT_ID must be an integer.")

    config["CT_HOST_TYPE"] = env.get("CT_HOST_TYPE", "demo").lower()
    if config["CT_HOST_TYPE"] not in ["demo", "live"]:
        raise ConfigError("CT_HOST_TYPE must be either 'demo' or 'live'.")

    config["DB_PATH"] = env.get("DB_PATH", "LiveExecution/data/live_trading.db")

    config["ALPHA_CONFIDENCE_THRESHOLD"] = float(env.get("ALPHA_CONFIDENCE_THRESHOLD", DEFAULT_THRESHOLDS["alpha_confidence_threshold"]))
    config["FILTER_THRESHOLD"] = float(env.get("FILTER_THRESHOLD", DEFAULT_THRESHOLDS["filter_threshold"]))
    config["SL_MULTIPLIER"] = float(env.get("SL_MULTIPLIER", DEFAULT_THRESHOLDS["sl_multiplier"]))
    config["TP_MULTIPLIER"] = float(env.get("TP_MULTIPLIER", DEFAULT_THRESHOLDS["tp_multiplier"]))

    return config
