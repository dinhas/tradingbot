import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import uuid
import time
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Formats log records as JSON objects."""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

def generate_correlation_id(symbol_name):
    """Generates a unique ID to trace a single candle-to-execution cycle."""
    return f"{symbol_name}-{int(time.time())}-{str(uuid.uuid4())[:8]}"

def setup_logger(name="LiveExecution"):
    """Sets up a rotating file logger with optional JSON formatting."""
    # Determine project root
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "conductor" / "logs"
    
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "live_execution.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        # Check for JSON logging preference
        is_json = os.environ.get("JSON_LOGGING", "false").lower() == "true"
        
        if is_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Rotating File Handler (10MB per file, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
