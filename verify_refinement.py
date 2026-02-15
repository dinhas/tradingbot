import sys
import os
import numpy as np
import pandas as pd
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'Alpha', 'src'))

from feature_engine import FeatureEngine
from trading_env import TradingEnv

def test_feature_refinement():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("VerifyRefinement")
    
    logger.info("Initializing FeatureEngine...")
    engine = FeatureEngine()
    
    # 1. Verify Feature List
    logger.info(f"Feature count: {len(engine.feature_names)}")
    expected_new_features = [
        "htf_ema_alignment", "htf_rsi_divergence", "swing_structure_proximity",
        "vwap_deviation", "delta_pressure", "volume_shock",
        "volatility_squeeze", "wick_rejection_strength", "breakout_velocity",
        "rsi_slope_divergence", "macd_momentum_quality"
    ]
    
    for feat in expected_new_features:
        if feat in engine.feature_names:
            logger.info(f"CONFIRMED: {feat} present in feature list.")
        else:
            logger.error(f"FAILED: {feat} MISSING from feature list.")
            
    removed_features = ["has_position", "unrealized_pnl", "equity", "drawdown"]
    for feat in removed_features:
        if feat in engine.feature_names:
             logger.error(f"FAILED: {feat} should be REMOVED but is present.")
        else:
             logger.info(f"CONFIRMED: {feat} removed.")

    if len(engine.feature_names) != 40:
        logger.error(f"FAILED: Expected 40 features, got {len(engine.feature_names)}")
    else:
        logger.info("CONFIRMED: Total features = 40")

    # 2. Verify Preprocessing & Env
    logger.info("Initializing TradingEnv (using dummy/file data)...")
    try:
        env = TradingEnv(data_dir='data') # Will fallback to dummy if needed
        logger.info("TradingEnv initialized.")
        
        obs, _ = env.reset()
        logger.info(f"Observation shape: {obs.shape}")
        
        if obs.shape != (40,):
             logger.error(f"FAILED: Observation shape mismatch. Expected (40,), got {obs.shape}")
        else:
             logger.info("CONFIRMED: Observation shape is (40,).")
             
        # Check for NaNs
        if np.isnan(obs).any():
             logger.error("FAILED: NaNs detected in observation!")
        else:
             logger.info("CONFIRMED: No NaNs in observation.")
             
        # Step
        obs, reward, done, trunc, info = env.step([1.0]) # Buy action
        logger.info("Step executed successfully.")
        logger.info(f"Reward: {reward}")
        
    except Exception as e:
        logger.error(f"Env test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_refinement()
