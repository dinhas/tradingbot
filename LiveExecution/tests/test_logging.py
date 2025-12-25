import unittest
import os
import logging
from pathlib import Path
import sys

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.logger import setup_logger

class TestLogging(unittest.TestCase):
    def test_logger_setup(self):
        logger = setup_logger("test_logger")
        self.assertEqual(logger.level, logging.DEBUG)
        
        # Check if file handler exists
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
        self.assertTrue(len(file_handlers) > 0, "No RotatingFileHandler found")
        
        # Check log file directory
        log_file = file_handlers[0].baseFilename
        log_dir = os.path.dirname(log_file)
        self.assertTrue(log_dir.endswith(os.path.join("conductor", "logs")), f"Expected log dir conductor/logs, got {log_dir}")
        
        # Test logging
        logger.debug("Test debug message")
        self.assertTrue(os.path.exists(log_file))
        with open(log_file, "r") as f:
            content = f.read()
            self.assertIn("Test debug message", content)

if __name__ == '__main__':
    unittest.main()
