"""
TradeGuard Dataset Generator
----------------------------
Generates a training dataset for the TradeGuard meta-model.
Runs the Alpha + Risk models over historical data and records:
- Input: Market Features (140) + Risk Parameters (SL/TP/Size)
- Output: True Outcome (Win/Loss) based on Oracle simulation

Implements strict Portfolio Management Rules:
- Gold (XAUUSD): No pyramiding (One trade at a time).
- Others: Reversal allowed (Close opposite, Open new).

Memory Optimized: Streams data to Parquet chunks to minimize RAM usage.
"""

import os
import sys
import gc
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import deque
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Numpy 2.x Compatibility Shim
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

# Import Project Modules
from Alpha.src.trading_env import TradingEnv
from RiskLayer.src.risk_env import RiskManagementEnv

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GuardDataGenerator:
    def __init__(self, alpha_path, risk_path, data_dir, output_dir):
        self.alpha_path = alpha_path
        self.risk_path = risk_path
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.chunk_size = 5000
        self.buffer = []
        self.chunk_counter = 0
        
        # Ensure output dir exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Models
        self._load_models()
        
        # Risk Config
        self.MAX_RISK_PER_TRADE = 0.80
        self.MAX_MARGIN_PER_TRADE_PCT = 0.80
        self.MAX_LEVERAGE = 400.0
        self.MIN_LOTS = 0.01
        
        # Portfolio State Tracker (Simulated for Rule Enforcement)
        # { 'ASSET_NAME': { 'direction': 1/-1 } or None }
        self.portfolio = {}

    def _load_models(self):
        """Load Alpha and Risk models."""
        logger.info("Loading models...")
        
        # 1. Alpha Model
        env_alpha = DummyVecEnv([lambda: TradingEnv(data_dir=self.data_dir, stage=1, is_training=False)])
        self.env = env_alpha.envs[0] # Direct access to env instance
        self.alpha_model = PPO.load(self.alpha_path, env=env_alpha)
        
        # Initialize Portfolio State
        self.portfolio = {asset: None for asset in self.env.assets}
        
        # 2. Risk Model (Requires dummy env for init)
        logger.info("Initializing Risk Model...")
        # Create minimal dummy file for RiskEnv init
        import tempfile
        dummy_df = pd.DataFrame({
            'direction': [1]*10, 'entry_price': [1.0]*10, 'atr': [0.01]*10,
            'atr_14': [0.01]*10, 'max_profit_pct': [0.01]*10, 'max_loss_pct': [-0.01]*10,
            'close_1000_price': [1.0]*10, 'features': [np.zeros(140)]*10, 'pair': ['EURUSD']*10
        })
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp:
            dummy_df.to_parquet(tmp.name)
            tmp_path = tmp.name
            
        try:
            risk_env = RiskManagementEnv(dataset_path=tmp_path, is_training=False)
            self.risk_model = PPO.load(self.risk_path, env=DummyVecEnv([lambda: risk_env]))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        logger.info("Models loaded successfully.")

    def _flush_buffer(self):
        """Write buffer to Parquet file."""
        if not self.buffer:
            return
            
        df = pd.DataFrame(self.buffer)
        filename = os.path.join(self.output_dir, f"guard_data_{self.chunk_counter}.parquet")
        df.to_parquet(filename, index=False)
        
        logger.info(f"Saved chunk {self.chunk_counter} ({len(df)} rows)")
        self.buffer = []
        self.chunk_counter += 1
        gc.collect()

    def _build_risk_obs(self, market_obs, asset):
        """Construct observation vector for Risk Model."""
        # Note: In generation mode, we approximate account state as 'Neutral'/Reset
        # because the Risk Model needs stable inputs, and we are evaluating the trade *in isolation*.
        
        # Market State (140)
        # Account State (5) - Default values
        account_obs = np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32) 
        # History (20) - Zeros (Assume no immediate history bias for Guard evaluation)
        hist_pnl = np.zeros(5, dtype=np.float32)
        hist_acts = np.zeros(15, dtype=np.float32)
        
        return np.concatenate([market_obs, account_obs, hist_pnl, hist_acts])

    def _parse_risk_action(self, action):
        sl_mult = np.clip((action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)
        tp_mult = np.clip((action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)
        risk_raw = np.clip((action[2] + 1) / 2, 0.0, 1.0)
        return sl_mult, tp_mult, risk_raw

    def generate(self):
        """Main generation loop."""
        logger.info("Starting Data Generation...")
        
        obs, _ = self.env.reset()
        done = False
        total_generated = 0
        
        # Pre-cache arrays for speed (already done in TradingEnv init, but ensuring access)
        prices = self.env.close_arrays
        atrs = self.env.atr_arrays
        
        while not done:
            # 1. Get Alpha Direction
            alpha_action, _ = self.alpha_model.predict(obs, deterministic=True)
            
            current_idx = self.env.current_step
            current_prices = self.env._get_current_prices()
            current_atrs = self.env._get_current_atrs()
            
            # 2. Iterate Assets
            for i, asset in enumerate(self.env.assets):
                # Parse Alpha Direction
                raw_dir = alpha_action[i]
                direction = 1 if raw_dir > 0.33 else (-1 if raw_dir < -0.33 else 0)
                
                if direction == 0:
                    continue
                    
                # --- MANAGER LOGIC (Rule Enforcement) ---
                current_pos = self.portfolio[asset]
                
                if current_pos is not None:
                    if current_pos['direction'] == direction:
                        # Same direction -> Ignore (No Pyramiding)
                        # Applied to ALL assets (Forex + Gold)
                        continue
                    else:
                        # Opposite direction -> Close current, Open New (Reversal)
                        # Applied to ALL assets
                        self.portfolio[asset] = None # Virtual Close
                        # Proceed to open new trade below...
                
                # --- PREPARE DATA POINT ---
                
                market_features = obs[:140] # Extract the market part
                
                # Get Risk Output
                risk_obs = self._build_risk_obs(market_features, asset)
                risk_action_raw, _ = self.risk_model.predict(risk_obs, deterministic=True)
                sl_mult, tp_mult, risk_raw = self._parse_risk_action(risk_action_raw)
                
                # --- ORACLE SIMULATION (Get Label) ---
                # We temporarily inject the trade into the environment to use its simulation logic
                price = current_prices[asset]
                atr = current_atrs[asset]
                
                # Setup virtual position for simulation
                sl_dist = sl_mult * atr
                tp_dist = tp_mult * atr
                sl = price - (direction * sl_dist)
                tp = price + (direction * tp_dist)
                
                virtual_pos = {
                    'direction': direction,
                    'entry_price': price,
                    'size': 1.0, # Dummy size for PnL calc
                    'sl': sl,
                    'tp': tp,
                    'entry_step': current_idx,
                    'sl_dist': sl_dist,
                    'tp_dist': tp_dist
                }
                
                # Inject
                original_pos = self.env.positions[asset]
                self.env.positions[asset] = virtual_pos
                
                # Simulate (Lookahead)
                pnl = self.env._simulate_trade_outcome(asset)
                
                # Restore
                self.env.positions[asset] = original_pos
                
                # --- UPDATE PORTFOLIO STATE ---
                # If we decided to 'take' this trade (conceptually), we mark it in portfolio
                # so subsequent steps know we have a position.
                self.portfolio[asset] = {'direction': direction}
                
                # --- RECORD DATA ---
                # Label: 1 if PnL > 0 (Winning), 0 if Loss
                # We can also store the raw PnL for regression if needed later
                label = 1 if pnl > 0 else 0
                
                # Flatten features to list
                feat_list = market_features.tolist()
                
                row = {
                    'asset': asset,
                    'direction': direction,
                    'sl_mult': sl_mult,
                    'tp_mult': tp_mult,
                    'risk_raw': risk_raw,
                    'pnl': pnl,
                    'label': label,
                    **{f'f_{k}': v for k, v in enumerate(feat_list)}
                }
                
                self.buffer.append(row)
                total_generated += 1
                
                if len(self.buffer) >= self.chunk_size:
                    self._flush_buffer()
            
            # Advance Environment
            obs, _, _, truncated, _ = self.env.step(alpha_action)
            done = truncated
            
            if total_generated % 1000 == 0:
                print(f"Generated {total_generated} samples... (Step {self.env.current_step})", end='\r')
        
        # Final Flush
        self._flush_buffer()
        logger.info(f"\nGeneration Complete. Total Samples: {total_generated}")

if __name__ == "__main__":
    # Hardcoded paths based on user environment
    ALPHA_PATH = "Alpha/models/checkpoints/8.03.zip"
    RISK_PATH = "RiskLayer/models/2.15.zip"
    DATA_DIR = "Alpha/backtest/data"
    OUTPUT_DIR = "Guard/data"
    
    gen = GuardDataGenerator(ALPHA_PATH, RISK_PATH, DATA_DIR, OUTPUT_DIR)
    gen.generate()