import logging
import os
from logging.handlers import RotatingFileHandler

# Define the root directory of the project
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# Create directories if they don't exist
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    handler.setFormatter(formatter)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(stream_handler)

    return logger

# Example of creating a logger
# You can create more loggers for different services
trade_logger = setup_logger('trade_logger', os.path.join(LOGS_DIR, 'trades.log'))
system_logger = setup_logger('system_logger', os.path.join(LOGS_DIR, 'system.log'))
error_logger = setup_logger('error_logger', os.path.join(LOGS_DIR, 'errors.log'))

# A general logger for quick use
log = setup_logger('general_logger', os.path.join(LOGS_DIR, 'general.log'))