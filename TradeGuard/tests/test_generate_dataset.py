import unittest
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

class TestDatasetGeneratorInit(unittest.TestCase):
    def setUp(self):
        # We expect the module to exist
        try:
            from generate_dataset import DatasetGenerator
            self.DatasetGenerator = DatasetGenerator
        except ImportError:
            self.DatasetGenerator = None

    def test_module_exists(self):
        """Test that the generate_dataset module and DatasetGenerator class exist."""
        if self.DatasetGenerator is None:
            self.fail("Could not import DatasetGenerator from generate_dataset")

    def test_initialization(self):
        """Test that the DatasetGenerator initializes correctly."""
        if self.DatasetGenerator is None:
            self.skipTest("DatasetGenerator not available")
        
        generator = self.DatasetGenerator()
        self.assertIsNotNone(generator.logger)
        self.assertTrue(isinstance(generator.logger, logging.Logger))
        self.assertEqual(generator.logger.name, "TradeGuard.DatasetGenerator")

if __name__ == '__main__':
    unittest.main()
