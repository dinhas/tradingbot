
import os
import sys
import json
import logging
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add numpy 1.x/2.x compatibility shim for SB3 model loading
if not hasattr(np, "_core"):
    import sys
    sys.modules["numpy._core"] = np.core

from stable_baselines3 import PPO
from Alpha.backtest.backtest_combined import CombinedBacktest

logger = logging.getLogger(__name__)

def load_tradeguard_model(model_path, metadata_path):
    """
    Loads the TradeGuard model and its metadata.
    Fails fast if either is missing or invalid.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TradeGuard model not found at {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"TradeGuard metadata not found at {metadata_path}")
    
    try:
        model = lgb.Booster(model_file=model_path)
    except Exception as e:
        raise ValueError(f"Failed to load TradeGuard model: {e}")
        
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load TradeGuard metadata: {e}")
        
    if 'threshold' not in metadata:
        raise KeyError("Metadata must contain 'threshold'")
        
    return model, metadata

class FullSystemBacktest(CombinedBacktest):
    """
    Extends CombinedBacktest to include TradeGuard filtering.
    """
    def __init__(self, alpha_model_path, risk_model_path, guard_model_path, guard_metadata_path, data_dir, initial_equity=10):
        # Load models
        alpha_model = PPO.load(alpha_model_path)
        risk_model = PPO.load(risk_model_path)
        guard_model, guard_metadata = load_tradeguard_model(guard_model_path, guard_metadata_path)
        
        super().__init__(alpha_model, risk_model, data_dir, initial_equity)
        
        self.guard_model = guard_model
        self.guard_threshold = guard_metadata['threshold']
        self.guard_metadata = guard_metadata
        
        logger.info(f"Initialized FullSystemBacktest with TradeGuard threshold: {self.guard_threshold}")

if __name__ == "__main__":
    # Placeholder for CLI entry point
    pass
