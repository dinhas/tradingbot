import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name="LiveExecution"):
    """Sets up a rotating file logger targeting conductor/logs/."""
    # Determine project root
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "conductor" / "logs"
    
    # Create log directory if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "live_execution.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers if they already exist (e.g., if setup_logger is called multiple times)
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Rotating File Handler (10MB per file, keep 5 backups)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
