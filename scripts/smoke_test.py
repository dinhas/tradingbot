import sys
import unittest
from unittest.mock import MagicMock
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

class TestInitialization(unittest.TestCase):
    def test_instantiation(self):
        from LiveExecution.src.ctrader_client import CTraderClient
        from LiveExecution.src.features import FeatureManager
        from LiveExecution.src.orchestrator import Orchestrator
        from LiveExecution.src.models import ModelLoader

        config = {
            "CT_APP_ID": "test", "CT_APP_SECRET": "test", "CT_ACCOUNT_ID": 12345,
            "CT_ACCESS_TOKEN": "test", "CT_HOST_TYPE": "demo",
            "DB_PATH": "LiveExecution/data/test_live_trading.db"
        }
        client = MagicMock(spec=CTraderClient)
        fm = FeatureManager()
        ml = MagicMock(spec=ModelLoader)
        notifier = MagicMock()
        orchestrator = Orchestrator(client, fm, ml, notifier, config=config)
        self.assertIsNotNone(orchestrator)
        print("Orchestrator instantiated successfully.")

if __name__ == "__main__":
    unittest.main()
