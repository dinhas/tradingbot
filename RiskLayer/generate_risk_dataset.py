import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from stable_baselines3 import PPO
from tqdm import tqdm
import logging
import gc
from src.frozen_alpha_env import TradingEnv

# Add numpy 1.x/2.x compatibility shim for SB3 model loading
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Tee:
    """Redirect stdout/stderr to both console and file."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = TeeStderr(self.file, self.stderr)
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

class TeeStderr:
    """Handle stderr separately."""
    def __init__(self, file, stderr):
        self.file = file
        self.stderr = stderr
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stderr.write(text)
        self.stderr.flush()
    
    def flush(self):
        self.file.flush()
        self.stderr.flush()

# Defaults
# UPDATED: Use the new 40-dim single-pair alpha model
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "checkpoints", "alpha", "ppo_final_model.zip")
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "risk_dataset.parquet")
LOOKAHEAD_STEPS = 6 # 30 mins (5m candles)
BATCH_SIZE = 50000 

def build_asset_obs_matrix(df, asset, assets, start_idx, end_idx):
    """
    Constructs the 40-feature observation matrix for a SPECIFIC asset.
    (25 asset-specific + 15 global)
    """
    num_rows = end_idx - start_idx
    obs_matrix = np.zeros((num_rows, 40), dtype=np.float32)
    df_slice = df.iloc[start_idx:end_idx]
    
    current_col = 0
    
    # 1. Per-Asset Features (25 columns)
    # Technicals (13)
    data_cols = [
        f"{asset}_close", f"{asset}_return_1", f"{asset}_return_12",
        f"{asset}_atr_14", f"{asset}_atr_ratio", f"{asset}_bb_position",
        f"{asset}_ema_9", f"{asset}_ema_21", f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21",
        f"{asset}_rsi_14", f"{asset}_macd_hist", f"{asset}_volume_ratio"
    ]
    obs_matrix[:, 0:13] = df_slice[data_cols].values.astype(np.float32)
    
    # Portfolio State (7) - Zeros for signal generation
    # 13 to 20 are zeros
    
    # Cross-Asset (5)
    cross_cols = [
         f"{asset}_corr_basket", f"{asset}_rel_strength",
         f"{asset}_corr_xauusd", f"{asset}_corr_eurusd", f"{asset}_rank"
    ]
    obs_matrix[:, 20:25] = df_slice[cross_cols].values.astype(np.float32)
    
    # 2. Global Features (15 columns)
    current_col = 25
    
    # Portfolio State (Global) - 4 columns
    obs_matrix[:, current_col] = 1.0 # Equity norm placeholder
    current_col += 4
    
    # Market Regime (3)
    gbp_ret = df_slice.get("GBPUSD_return_1", 0)
    xau_ret = df_slice.get("XAUUSD_return_1", 0)
    if isinstance(gbp_ret, pd.Series):
        obs_matrix[:, current_col] = ((gbp_ret + xau_ret) / 2).values.astype(np.float32)
    else:
        obs_matrix[:, current_col] = 0
    current_col += 1
    
    ret_cols = [f"{a}_return_1" for a in assets]
    obs_matrix[:, current_col] = df_slice[ret_cols].std(axis=1).values.astype(np.float32)
    current_col += 1
    
    atr_ratio_cols = [f"{a}_atr_ratio" for a in assets]
    obs_matrix[:, current_col] = df_slice[atr_ratio_cols].mean(axis=1).values.astype(np.float32)
    current_col += 1
    
    # Session (8)
    sess_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'session_asian', 'session_london', 'session_ny', 'session_overlap']
    obs_matrix[:, current_col:current_col+8] = df_slice[sess_cols].values.astype(np.float32)
    
    return obs_matrix

def generate_dataset_batched(model_path, data_dir, output_file):
    """Generates the risk dataset using optimized batched inference with 40-dim features."""
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        sys.exit(1)

    logger.info(f"Loading Alpha Model from {model_path}...")
    try:
        model = PPO.load(model_path, device='cpu')
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    logger.info(f"Loading and preprocessing data from {data_dir}...")
    try:
        env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        sys.exit(1)
        
    df = env.processed_data
    if df is None or len(df) == 0:
        logger.error("No data loaded or processed data is empty.")
        sys.exit(1)

    total_rows = len(df)
    assets = env.assets
    start_idx = 500
    end_idx = total_rows - LOOKAHEAD_STEPS
    
    if start_idx >= end_idx:
        logger.error(f"Data too short. Total rows: {total_rows}, needed > {start_idx + LOOKAHEAD_STEPS}")
        sys.exit(1)

    close_arrays = {a: env.close_arrays[a] for a in assets}
    high_arrays = {a: env.high_arrays[a] for a in assets}
    low_arrays = {a: env.low_arrays[a] for a in assets}
    atr_arrays = {a: env.atr_arrays[a] for a in assets}
    
    timestamps = df.index[start_idx:end_idx]
    num_rows = end_idx - start_idx
    
    # Final results accumulator
    all_results = {
        'timestamp': [], 'pair': [], 'direction': [], 'entry_price': [], 
        'atr': [], 'features': [], 'max_profit_pct': [], 'max_loss_pct': [], 
        'close_1000_price': []
    }
    
    total_signals = 0
    
    for asset in assets:
        logger.info(f"Processing asset: {asset}...")
        
        # 1. Build 40-dim observation matrix for this asset
        try:
            obs_matrix = build_asset_obs_matrix(df, asset, assets, start_idx, end_idx)
        except Exception as e:
            logger.error(f"Error building observation matrix for {asset}: {e}")
            continue
        
        # 2. Batch Inference
        batch_starts = np.arange(0, num_rows, BATCH_SIZE)
        for b_start in tqdm(batch_starts, desc=f"Inference {asset}"):
            b_end = min(b_start + BATCH_SIZE, num_rows)
            batch_obs = obs_matrix[b_start:b_end]
            
            actions, _ = model.predict(batch_obs, deterministic=True)
            
            # Action is single dimension for new model: [Direction]
            # Clip to identify buy/sell
            buy_mask = actions > 0.33
            sell_mask = actions < -0.33
            
            mask = buy_mask.flatten() | sell_mask.flatten()
            if not np.any(mask):
                continue
                
            active_local_indices = np.where(mask)[0]
            for local_idx in active_local_indices:
                global_idx = start_idx + b_start + local_idx
                direction = 1 if actions[local_idx] > 0.33 else -1
                
                entry_price = close_arrays[asset][global_idx]
                atr = atr_arrays[asset][global_idx]
                
                f_end_step = global_idx + LOOKAHEAD_STEPS
                if f_end_step >= len(high_arrays[asset]): continue
                    
                future_highs = high_arrays[asset][global_idx+1 : f_end_step]
                future_lows = low_arrays[asset][global_idx+1 : f_end_step]
                future_close = close_arrays[asset][f_end_step-1]
                
                if len(future_highs) == 0: continue
                
                max_h = np.max(future_highs)
                min_l = np.min(future_lows)
                
                if direction == 1:
                    max_profit_pct = (max_h - entry_price) / entry_price
                    max_loss_pct = (min_l - entry_price) / entry_price
                else:
                    max_profit_pct = (entry_price - min_l) / entry_price
                    max_loss_pct = (entry_price - max_h) / entry_price

                all_results['timestamp'].append(timestamps[b_start + local_idx])
                all_results['pair'].append(asset) # renamed from 'asset' to 'pair' for consistency
                all_results['direction'].append(direction)
                all_results['entry_price'].append(entry_price)
                all_results['atr'].append(atr)
                all_results['features'].append(batch_obs[local_idx].tolist())
                all_results['max_profit_pct'].append(max_profit_pct)
                all_results['max_loss_pct'].append(max_loss_pct)
                all_results['close_1000_price'].append(future_close)
                total_signals += 1

    if total_signals > 0:
        logger.info(f"Saving {total_signals} signals to {output_file}...")
        pd.DataFrame(all_results).to_parquet(output_file, index=False)
    else:
        logger.warning("No signals generated.")
        sys.exit(1) # Should fail if no data was generated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Risk Dataset (40-dim Single Pair)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    
    args = parser.parse_args()
    
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"gen_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_file)
    
    try:
        generate_dataset_batched(args.model, args.data, args.output)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        tee.close()
