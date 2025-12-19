import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import deque
from stable_baselines3 import PPO
from tqdm import tqdm
import logging
import gc

# Add project root to sys.path to allow absolute imports
# Since this script is now in TradeGuard/src/, parent.parent is the root
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Alpha.src.trading_env import TradingEnv
from RiskLayer.src.risk_env import RiskManagementEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradeGuardDataGenerator:
    """ Generates a dataset for TradeGuard by running Alpha + Risk models historically (2016-2024). """

    def __init__(self, alpha_model_path, risk_model_path, data_dir, initial_equity=10.0):
        self.alpha_model_path = alpha_model_path
        self.risk_model_path = risk_model_path
        self.data_dir = data_dir
        self.initial_equity = initial_equity
        
        # 1. Load Alpha Environment (for data and feature engineering)
        logger.info(f"Initializing Alpha Environment with data from {data_dir}...")
        self.env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
        self.assets = self.env.assets
        
        # 2. Load Models
        logger.info(f"Loading Alpha Model: {alpha_model_path}")
        self.alpha_model = PPO.load(alpha_model_path, device='cpu')
        
        logger.info(f"Loading Risk Model: {risk_model_path}")
        # Use a dummy environment to load the risk model
        self.risk_model = PPO.load(risk_model_path, device='cpu')
        
        # 3. State Tracking
        self.equity = initial_equity
        self.peak_equity = initial_equity
        self.asset_histories = {
            asset: {
                'pnl_history': deque([0.0] * 5, maxlen=5),
                'action_history': deque([np.zeros(3, dtype=np.float32) for _ in range(5)], maxlen=5)
            }
            for asset in self.assets
        }
        
        # Risk Model Constants (match RiskManagementEnv)
        self.MAX_RISK_PER_TRADE = 0.40  # 40%
        self.MAX_MARGIN_PER_TRADE_PCT = 0.80
        self.MAX_LEVERAGE = 400.0
        self.MIN_LOTS = 0.01
        self.CONTRACT_SIZE = 100000
        
        # Cumulative results
        self.collected_data = []

    def build_risk_observation(self, asset, alpha_obs):
        """Build 165-feature observation for Risk Model"""
        market_obs = alpha_obs[:140]
        
        # Account state (5)
        drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
        equity_norm = self.equity / self.initial_equity
        risk_cap_mult = max(0.2, 1.0 - (drawdown * 2.0))
        
        account_obs = np.array([
            equity_norm,
            drawdown,
            0.0, # Leverage placeholder
            risk_cap_mult,
            0.0 # Padding
        ], dtype=np.float32)
        
        # History (20)
        hist_pnl = np.array(self.asset_histories[asset]['pnl_history'], dtype=np.float32)
        hist_acts = np.array(self.asset_histories[asset]['action_history'], dtype=np.float32).flatten()
        
        obs = np.concatenate([market_obs, account_obs, hist_pnl, hist_acts])
        return np.nan_to_num(obs, nan=0.0)

    def calculate_trade_outcome(self, asset, direction, sl_mult, tp_mult):
        """Simulate trade outcome using Alpha Env's look-ahead logic"""
        # Inject virtual position into Alpha Env
        current_prices = self.env._get_current_prices()
        atrs = self.env._get_current_atrs()
        
        entry_price = current_prices[asset]
        atr = atrs[asset]
        
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        
        # Safety min SL
        min_sl = max(0.0001 * entry_price, 0.2 * atr)
        if sl_dist < min_sl: sl_dist = min_sl
        
        sl_price = entry_price - (direction * sl_dist)
        tp_price = entry_price + (direction * tp_dist)
        
        # Outcome simulation
        # Note: _simulate_trade_outcome uses self.positions[asset]
        original_pos = self.env.positions[asset]
        self.env.positions[asset] = {
            'direction': direction,
            'entry_price': entry_price,
            'size': 1.0, # Notional for % pnl
            'sl': sl_price,
            'tp': tp_price,
            'entry_step': self.env.current_step,
            'sl_dist': sl_dist,
            'tp_dist': tp_dist
        }
        
        # This returns pnl which is roughly price_change_pct * 100 (leverage in AlphaEnv)
        pnl_val = self.env._simulate_trade_outcome(asset)
        
        # We want the exit type and R-multiple
        # Logic duplicated from RiskManagementEnv for accuracy
        max_profit_pct = 0
        max_loss_pct = 0 # Not directly returned by env, but we can approximate or use pnl
        
        # Let's get more detail
        # We define outcome as:
        # 1 if pnl > 0 else 0
        outcome = 1 if pnl_val > 0 else 0
        
        # Restore env
        self.env.positions[asset] = original_pos
        
        return pnl_val, outcome, sl_dist, tp_dist

    def generate(self, output_path):
        """Run generation loop"""
        logger.info(f"Starting data generation (Total steps: {self.env.max_steps})")
        
        # Reset Env
        obs, _ = self.env.reset()
        
        for step in tqdm(range(self.env.max_steps)):
            # 1. Get Alpha Action
            alpha_action, _ = self.alpha_model.predict(obs, deterministic=True)
            
            # 2. Iterate Assets for signals
            any_signal = False
            for i, asset in enumerate(self.assets):
                direction_raw = alpha_action[i]
                direction = 1 if direction_raw > 0.33 else (-1 if direction_raw < -0.33 else 0)
                
                if direction == 0:
                    continue
                
                any_signal = True
                
                # 3. Get Risk Action
                risk_obs = self.build_risk_observation(asset, obs)
                risk_action, _ = self.risk_model.predict(risk_obs, deterministic=True)
                
                # Parse Risk Action
                sl_mult = np.clip((risk_action[0] + 1) / 2 * 1.8 + 0.2, 0.2, 2.0)
                tp_mult = np.clip((risk_action[1] + 1) / 2 * 3.5 + 0.5, 0.5, 4.0)
                risk_raw = np.clip((risk_action[2] + 1) / 2, 0.0, 1.0)
                
                # Final Calculations for positions
                pnl_val, outcome, sl_final, tp_final = self.calculate_trade_outcome(asset, direction, sl_mult, tp_mult)
                
                # 4. Construct Row (95-100 Features)
                row = {}
                
                # Alpha Features (Selection or all 140)
                # Let's take the first 140
                for idx, feat in enumerate(obs):
                    row[f'alpha_{idx}'] = feat
                
                # Risk Features
                row['sl_mult'] = sl_mult
                row['tp_mult'] = tp_mult
                row['risk_raw'] = risk_raw
                row['sl_dist_atr'] = sl_final / max(self.env.atr_arrays[asset][step], 1e-9)
                row['tp_dist_atr'] = tp_final / max(self.env.atr_arrays[asset][step], 1e-9)
                row['rr_ratio'] = tp_final / sl_final
                
                # Account State
                drawdown = 1.0 - (self.equity / max(self.peak_equity, 1e-9))
                account_obs = risk_obs[140:145]
                row['equity_norm'] = account_obs[0]
                row['drawdown'] = account_obs[1]
                row['risk_cap_mult'] = account_obs[3]
                row['num_open_positions'] = sum(1 for p in self.env.positions.values() if p is not None)
                
                # Trade Context
                row['direction'] = direction
                row['asset_id'] = i
                row['atr'] = self.env.atr_arrays[asset][step]
                row['timestamp'] = self.env._get_current_timestamp().timestamp()
                
                # Outcome Labels
                row['pnl_simulated'] = pnl_val
                row['outcome'] = outcome # 1 if win, 0 if loss
                
                self.collected_data.append(row)
                
                # 5. Update Portfolio State (Pseudo-Backtest)
                # To keep account features dynamic, we update equity
                # Scaling pnl_val: pnl_val is % return * 100 * leverage in alpha env?
                # Actually alpha env formula: price_change_pct * (size * leverage)
                # where size is normalized.
                # Let's just update equity based on a fixed risk of 1% for the pseudo-portfolio
                self.equity *= (1.0 + (pnl_val / 1000.0)) # scaled down for stability
                self.peak_equity = max(self.peak_equity, self.equity)
                
                # Update Hist
                self.asset_histories[asset]['pnl_history'].append(pnl_val / 100.0)
                self.asset_histories[asset]['action_history'].append(np.array([sl_mult, tp_mult, risk_raw]))

            # Step Env
            obs, _, _, _, _ = self.env.step(alpha_action)
            
            # Periodic Save / GC
            if step % 10000 == 0 and step > 0:
                logger.info(f"Progress: {step}/{self.env.max_steps} | Signals: {len(self.collected_data)}")
                gc.collect()

        # Save Final
        logger.info(f"Generation complete. Total signals: {len(self.collected_data)}")
        df = pd.DataFrame(self.collected_data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path)
        logger.info(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha-model", type=str, default="Alpha/models/checkpoints/8.03.zip")
    parser.add_argument("--risk-model", type=str, default="RiskLayer/models/risk_model_final(1).zip")
    parser.add_argument("--data-dir", type=str, default="TradeGuard/data")
    parser.add_argument("--output", type=str, default="TradeGuard/data/dataset_2016_2024.parquet")
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).resolve().parent.parent.parent
    alpha_path = project_root / args.alpha_model
    risk_path = project_root / args.risk_model
    data_dir = project_root / args.data_dir
    output_path = project_root / args.output

    generator = TradeGuardDataGenerator(
        alpha_model_path=str(alpha_path),
        risk_model_path=str(risk_path),
        data_dir=str(data_dir)
    )
    generator.generate(str(output_path))
