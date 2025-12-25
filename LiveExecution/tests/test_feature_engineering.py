import unittest
import sys
from pathlib import Path

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

class TestFeatureEngineering(unittest.TestCase):
    def test_imports(self):
        try:
            from Alpha.src.feature_engine import FeatureEngine as AlphaFE
            from RiskLayer.src.feature_engine import FeatureEngine as RiskFE
            from TradeGuard.src.feature_calculator import TradeGuardFeatureCalculator
            print("Successfully imported all feature engineering modules.")
        except ImportError as e:
            self.fail(f"Failed to import feature engineering modules: {e}")

    def test_logic_integration(self):
        # This will be more detailed once we have the actual integration code
        # For now, we just check if we can instantiate or call basic logic
        from LiveExecution.src.features import FeatureManager
        manager = FeatureManager()
        self.assertIsNotNone(manager)

if __name__ == '__main__':
    unittest.main()
