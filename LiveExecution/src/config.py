import os
from dotenv import load_dotenv

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

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
    config["TRAILING_STOP_ENABLED"] = env.get("TRAILING_STOP_ENABLED", "true").lower() == "true"
    config["TRAILING_TRIGGER_R"] = float(env.get("TRAILING_TRIGGER_R", "1.25"))
    config["TRAILING_ATR_MULT"] = float(env.get("TRAILING_ATR_MULT", "1.0"))
    config["BREAKEVEN_TRIGGER_R"] = float(env.get("BREAKEVEN_TRIGGER_R", "0.90"))
    config["BREAKEVEN_BUFFER_ATR"] = float(env.get("BREAKEVEN_BUFFER_ATR", "0.10"))
        
    return config
