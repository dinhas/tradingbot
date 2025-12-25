import os
from dotenv import load_dotenv

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

def load_config():
    """Loads and validates configuration from environment variables."""
    # Load .env file if it exists
    load_dotenv()
    
    required_vars = [
        "CT_APP_ID",
        "CT_APP_SECRET",
        "CT_ACCOUNT_ID",
        "CT_ACCESS_TOKEN",
        "DISCORD_WEBHOOK_URL"
    ]
    
    config = {}
    
    # Check for missing variables
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")
    
    # Populate and validate
    config["CT_APP_ID"] = os.getenv("CT_APP_ID")
    config["CT_APP_SECRET"] = os.getenv("CT_APP_SECRET")
    config["CT_ACCESS_TOKEN"] = os.getenv("CT_ACCESS_TOKEN")
    config["DISCORD_WEBHOOK_URL"] = os.getenv("DISCORD_WEBHOOK_URL")
    
    # CT_ACCOUNT_ID must be an integer
    try:
        config["CT_ACCOUNT_ID"] = int(os.getenv("CT_ACCOUNT_ID"))
    except ValueError:
        raise ConfigError("CT_ACCOUNT_ID must be an integer.")
    
    # CT_HOST_TYPE defaults to 'demo'
    config["CT_HOST_TYPE"] = os.getenv("CT_HOST_TYPE", "demo").lower()
    if config["CT_HOST_TYPE"] not in ["demo", "live"]:
        raise ConfigError("CT_HOST_TYPE must be either 'demo' or 'live'.")
        
    return config
