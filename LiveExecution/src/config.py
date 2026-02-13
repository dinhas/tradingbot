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
        
    return config
