import unittest
import os
import sys
from pathlib import Path

class TestScaffolding(unittest.TestCase):
    def test_project_structure(self):
        base_dir = "LiveExecution"
        expected_paths = [
            os.path.join(base_dir, "__init__.py"),
            os.path.join(base_dir, "src", "__init__.py"),
            os.path.join(base_dir, "config"),
            os.path.join(base_dir, "tests", "__init__.py"),
            os.path.join(base_dir, "README.md"),
        ]
        
        for path in expected_paths:
            self.assertTrue(os.path.exists(path), f"Missing path: {path}")

if __name__ == '__main__':
    unittest.main()