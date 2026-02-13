import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque

# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add numpy 1.x/2.x compatibility shim for SB3 model loading
if not hasattr(np, "_core"):
    import sys
    sys.modules["numpy._core"] = np.core

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from Alpha.src.trading_env import TradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskCollector:
    def __init__(self, alpha_model_path, risk_model_path, data_dir, initial_equity=10):
        self.initial_equity = initial_equity
        self.equity = initial_equity
        self.peak_equity = initial_equity
        
        # Load models
        logger.info(f"Loading Alpha model from {alpha_model_path}")
        self.alpha_model = PPO.load(alpha_model_path)
        logger.info(f"Loading Risk model from {risk_model_path}")
        self.risk_model = PPO.load(risk_model_path)
        
        # Initialize Alpha environment
        self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
        self.env.equity = initial_equity
        self.env.peak_equity = initial_equity
        
        # Per-asset history tracking (needed for risk observation)
        self.asset_histories = {
            asset: {
                'pnl_history': deque([0.0] * 5, maxlen=5),
                'action_history': deque([np.zeros(3, dtype=np.float32) for _ in range(5)], maxlen=5)
            }
            for asset in self.env.assets
        }
        
    def build_risk_observation(self, asset):
        """Build 165-feature observation for risk model"""
        # 1. Market state (140 features)
        alpha_obs = self.env._get_observation()
        market_obs = alpha_obs[:140]
        
        # 2. Account state (5 features)
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        equity_norm = self.equity / self.initial_equity
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        account_obs = np.array([
            equity_norm,
            drawdown,
            0.0,  # Leverage placeholder
            risk_cap_mult,
            0.0   # Padding
        ], dtype=np.float32)
        
        # 3. History (20 features)
        asset_history = self.asset_histories[asset]
        hist_pnl = np.array(asset_history['pnl_history'], dtype=np.float32)
        hist_acts = np.array(asset_history['action_history'], dtype=np.float32).flatten()
        
        # Combine all features
        obs = np.concatenate([market_obs, account_obs, hist_pnl, hist_acts])
        
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs

    def parse_risk_action(self, action):
        """Parse risk model action to SL/TP/sizing"""
        sl_mult = np.clip((action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)
        tp_mult = np.clip((action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)
        risk_raw = np.clip((action[2] + 1) / 2, 0.0, 1.0)
        return float(sl_mult), float(tp_mult), float(risk_raw)

    def collect(self, limit=20):
        collected_data = []
        obs, _ = self.env.reset()
        done = False
        
        logger.info(f"Starting collection of {limit} risk outputs...")
        
        while not done and len(collected_data) < limit:
            # 1. Alpha Prediction
            alpha_action, _ = self.alpha_model.predict(obs, deterministic=True)
            
            directions = {}
            for i, asset in enumerate(self.env.assets):
                direction_raw = alpha_action[i]
                direction = 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0)
                directions[asset] = direction
            
            # 2. Risk Evaluation for each asset with non-zero direction
            for asset, direction in directions.items():
                if direction == 0:
                    continue
                
                risk_obs = self.build_risk_observation(asset)
                risk_action, _ = self.risk_model.predict(risk_obs, deterministic=True)
                sl_mult, tp_mult, risk_raw = self.parse_risk_action(risk_action)
                
                entry = {
                    'step': int(self.env.current_step),
                    'timestamp': str(self.env._get_current_timestamp()),
                    'asset': asset,
                    'alpha_direction': int(direction),
                    'risk_sl_mult': sl_mult,
                    'risk_tp_mult': tp_mult,
                    'risk_raw': risk_raw
                }
                collected_data.append(entry)
                
                if len(collected_data) >= limit:
                    break
            
            if len(collected_data) >= limit:
                break
                
            # Advance environment (minimal steps to get next obs)
            self.env.current_step += 1
            obs = self.env._validate_observation(self.env._get_observation())
            done = self.env.current_step >= self.env.max_steps
            
        return collected_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-model", type=str, default="Alpha/models/checkpoints/8.03.zip")
    parser.add_argument("--risk-model", type=str, default="RiskLayer/models/2.15.zip")
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--output", type=str, default="risk_outputs_sample.json")
    args = parser.parse_args()
    
    collector = RiskCollector(
        alpha_model_path=str(Path(project_root) / args.alpha_model),
        risk_model_path=str(Path(project_root) / args.risk_model),
        data_dir=str(Path(project_root) / args.data_dir)
    )
    
    data = collector.collect(limit=20)
    
    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)
        
    logger.info(f"Successfully saved {len(data)} risk outputs to {args.output}")
