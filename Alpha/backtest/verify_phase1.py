
import os
import sys
from pathlib import Path
import json
import lightgbm as lgb
import numpy as np
import tempfile
import shutil

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.backtest.backtest_full_system import load_tradeguard_model

def verify():
    print("Verifying Phase 1: Infrastructure & Model Loading...")
    
    test_dir = tempfile.mkdtemp()
    try:
        model_path = os.path.join(test_dir, "guard_model.txt")
        metadata_path = os.path.join(test_dir, "model_metadata.json")
        
        # 1. Create dummy model
        data = np.random.rand(100, 10)
        label = np.random.randint(2, size=100)
        train_data = lgb.Dataset(data, label=label)
        model = lgb.train({}, train_data, 1)
        model.save_model(model_path)
        
        # 2. Create valid metadata
        metadata = {"metrics": {"precision": 0.65}, "threshold": 0.55}
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
            
        # 3. Load
        loaded_model, loaded_metadata = load_tradeguard_model(model_path, metadata_path)
        print(f"Successfully loaded model and metadata.")
        print(f"Threshold: {loaded_metadata['threshold']}")
        
        # 4. Verify fail-fast
        try:
            load_tradeguard_model("non_existent.txt", metadata_path)
            print("ERROR: Fail-fast for missing model failed!")
            return False
        except FileNotFoundError:
            print("Verified: Fail-fast for missing model works.")
            
        print("Phase 1 Verification SUCCESSFUL.")
        return True
    finally:
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    if verify():
        sys.exit(0)
    else:
        sys.exit(1)
