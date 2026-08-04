import os
import logging
from pathlib import Path
import uuid
import time
import json
from datetime import datetime, timedelta

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
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


class DailyFileHandler(logging.FileHandler):
    """
    Custom file handler that rolls over daily.
    The active file is always named tradebot-YYYY-MM-DD.log.
    """
    def __init__(self, log_dir, prefix="tradebot-", encoding="utf-8", retain_days=30):
        self.log_dir = Path(log_dir)
        self.prefix = prefix
        self.encoding = encoding
        self.retain_days = retain_days
        self.current_date = datetime.utcnow().strftime("%Y-%m-%d")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_old_logs()

        filepath = self.log_dir / f"{self.prefix}{self.current_date}.log"
        super().__init__(filepath, encoding=self.encoding)

    def emit(self, record):
        now_date = datetime.utcnow().strftime("%Y-%m-%d")
        if now_date != self.current_date:
            self.current_date = now_date
            self.close()
            filepath = self.log_dir / f"{self.prefix}{self.current_date}.log"
            self.baseFilename = os.path.abspath(filepath)
            self.stream = self._open()
            self._cleanup_old_logs()
        super().emit(record)

    def _cleanup_old_logs(self):
        """Deletes daily logs older than retain_days."""
        try:
            now = datetime.utcnow()
            for log_file in self.log_dir.glob(f"{self.prefix}*.log"):
                date_str = log_file.name[len(self.prefix):-4]
                try:
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if now - file_date > timedelta(days=self.retain_days):
                        log_file.unlink()
                except ValueError:
                    pass
        except Exception:
            pass


def generate_correlation_id(symbol_name):
    """Generates a unique ID to trace a single candle-to-execution cycle."""
    return f"{symbol_name}-{int(time.time())}-{str(uuid.uuid4())[:8]}"


def setup_logger(name="LiveExecution"):
    """Sets up standard logger with a DailyFileHandler and a StreamHandler."""
    # Determine project root
    project_root = Path(__file__).resolve().parent.parent.parent
    log_dir = project_root / "logs"

    logger = logging.getLogger(name)

    # Determine log level from env var
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger.setLevel(log_level)

    # Avoid adding handlers if they already exist
    if not logger.handlers:
        # Check for JSON logging preference
        is_json = os.environ.get("JSON_LOGGING", "false").lower() == "true"

        if is_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # Daily File Handler
        file_handler = DailyFileHandler(log_dir, prefix="tradebot-")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
