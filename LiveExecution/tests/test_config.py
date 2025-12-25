import unittest
import os
from unittest.mock import patch
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from LiveExecution.src.config import load_config, ConfigError

class TestConfig(unittest.TestCase):
    def test_load_config_success(self):
        mock_env = {
            "CT_APP_ID": "test_app_id",
            "CT_APP_SECRET": "test_app_secret",
            "CT_ACCOUNT_ID": "12345",
            "CT_ACCESS_TOKEN": "test_token",
            "DISCORD_WEBHOOK_URL": "http://test.webhook",
            "CT_HOST_TYPE": "demo"
        }
        with patch.dict(os.environ, mock_env):
            config = load_config()
            self.assertEqual(config["CT_APP_ID"], "test_app_id")
            self.assertEqual(config["CT_ACCOUNT_ID"], 12345)
            self.assertEqual(config["CT_HOST_TYPE"], "demo")

    def test_load_config_missing_var(self):
        mock_env = {
            "CT_APP_ID": "test_app_id",
            # CT_APP_SECRET is missing
        }
        with patch.dict(os.environ, mock_env, clear=True):
            with self.assertRaises(ConfigError) as cm:
                load_config()
            self.assertIn("CT_APP_SECRET", str(cm.exception))

    def test_load_config_invalid_type(self):
        mock_env = {
            "CT_APP_ID": "test_app_id",
            "CT_APP_SECRET": "test_app_secret",
            "CT_ACCOUNT_ID": "not_an_int",
            "CT_ACCESS_TOKEN": "test_token",
            "DISCORD_WEBHOOK_URL": "http://test.webhook",
            "CT_HOST_TYPE": "demo"
        }
        with patch.dict(os.environ, mock_env):
            with self.assertRaises(ConfigError) as cm:
                load_config()
            self.assertIn("CT_ACCOUNT_ID", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
