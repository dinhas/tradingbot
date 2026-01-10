import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm
import logging
import gc

# Add numpy 1.x/2.x compatibility shim for SB3 model loading
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add Alpha to path to import environment
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, "Alpha"))
from src.trading_env import TradingEnv

# Constants
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints", "alpha")
LOOKAHEAD_STEPS = 6  # 30 mins
BATCH_SIZE = 50000
MIN_RR = 0.5 
ACTION_THRESHOLD = 0.40 # Higher threshold to reduce noise and memory usage

def get_latest_model_files(models_dir):
    if not os.path.exists(models_dir): return None, None
    files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
    if not files: return None, None
    files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    latest_zip = files[0]
    return os.path.join(models_dir, latest_zip), os.path.join(models_dir, f"{os.path.splitext(latest_zip)[0]}_vecnormalize.pkl")

def build_master_feature_cache(df, assets, start_idx, end_idx):
    num_rows = end_idx - start_idx
    obs_matrix = np.zeros((num_rows, 140), dtype=np.float32)
    df_slice = df.iloc[start_idx:end_idx]
    current_col = 0
    for asset in assets:
        data_cols = [
            f"{asset}_close", f"{asset}_return_1", f"{asset}_return_12",
            f"{asset}_atr_14", f"{asset}_atr_ratio", f"{asset}_bb_position",
            f"{asset}_ema_9", f"{asset}_ema_21", f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21",
            f"{asset}_rsi_14", f"{asset}_macd_hist", f"{asset}_volume_ratio"
        ]
        obs_matrix[:, current_col:current_col+13] = df_slice[data_cols].values.astype(np.float32)
        current_col += 20 # 13 + 7 (portfolio)
        cross_cols = [f"{asset}_corr_basket", f"{asset}_rel_strength", f"{asset}_corr_xauusd", f"{asset}_corr_eurusd", f"{asset}_rank"]
        obs_matrix[:, current_col:current_col+5] = df_slice[cross_cols].values.astype(np.float32)
        current_col += 5
    
    # Global Features
    obs_matrix[:, 125] = 10000.0 # Equity
    gbp_ret = df_slice.get("GBPUSD_return_1", 0)
    xau_ret = df_slice.get("XAUUSD_return_1", 0)
    obs_matrix[:, 129] = ((gbp_ret + xau_ret) / 2).values if hasattr(gbp_ret, "values") else 0
    ret_cols = [f"{a}_return_1" for a in assets]
    obs_matrix[:, 130] = df_slice[ret_cols].std(axis=1).values
    atr_ratio_cols = [f"{a}_atr_ratio" for a in assets]
    obs_matrix[:, 131] = df_slice[atr_ratio_cols].mean(axis=1).values
    sess_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'session_asian', 'session_london', 'session_ny', 'session_overlap']
    obs_matrix[:, 132:140] = df_slice[sess_cols].values
    return obs_matrix

def generate_dataset(data_dir, output_file):
    model_path, norm_path = get_latest_model_files(MODELS_DIR)
    if not model_path:
        logger.error("Alpha model not found.")
        return

    logger.info(f"Loading Alpha Model: {model_path}")
    model = PPO.load(model_path, device='cpu')
    
    norm_env = None
    if norm_path and os.path.exists(norm_path):
        dummy_venv = DummyVecEnv([lambda: None])
        norm_env = VecNormalize.load(norm_path, dummy_venv)
        norm_env.training = False

    try:
        env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
    except Exception as e:
        logger.error(f"Failed to init env: {e}")
        return

    df = env.processed_data
    assets = env.assets
    close_arrays = {a: env.close_arrays[a] for a in assets}
    high_arrays = {a: env.high_arrays[a] for a in assets}
    low_arrays = {a: env.low_arrays[a] for a in assets}
    
    start_idx = 500
    end_idx = len(df) - LOOKAHEAD_STEPS
    
    logger.info("Building feature cache...")
    all_observations = build_master_feature_cache(df, assets, start_idx, end_idx)
    timestamps = df.index[start_idx:end_idx]
    
    del df
    gc.collect()

    num_rows = end_idx - start_idx
    batch_starts = np.arange(0, num_rows, BATCH_SIZE)
    
    all_samples_list = []
    
    for b_start in tqdm(batch_starts, desc="Processing Batches"):
        b_end = min(b_start + BATCH_SIZE, num_rows)
        master_batch_obs = all_observations[b_start:b_end]
        
        for i, asset in enumerate(assets):
            asset_feats = master_batch_obs[:, i*25 : i*25+25]
            global_feats = master_batch_obs[:, 125:140]
            single_pair_obs = np.concatenate([asset_feats, global_feats], axis=1)
            
            if norm_env:
                single_pair_obs = norm_env.normalize_obs(single_pair_obs)
            
            actions, _ = model.predict(single_pair_obs, deterministic=True)
            actions = actions.flatten()
            
            buy_mask = actions > ACTION_THRESHOLD
            sell_mask = actions < -ACTION_THRESHOLD
            combined_mask = buy_mask | sell_mask
            
            if not np.any(combined_mask):
                continue
                
            active_indices = np.where(combined_mask)[0]
            
            for local_idx in active_indices:
                global_idx = start_idx + b_start + local_idx
                direction = 1 if actions[local_idx] > ACTION_THRESHOLD else -1
                
                entry_price = close_arrays[asset][global_idx]
                f_end = global_idx + LOOKAHEAD_STEPS
                
                if f_end >= len(high_arrays[asset]): continue
                
                future_highs = high_arrays[asset][global_idx+1 : f_end+1]
                future_lows = low_arrays[asset][global_idx+1 : f_end+1]
                
                if len(future_highs) < LOOKAHEAD_STEPS: continue
                
                max_h = np.max(future_highs)
                min_l = np.min(future_lows)
                
                if direction == 1: # Buy
                    risk_dist = entry_price - min_l
                    reward_dist = max_h - entry_price
                else: # Sell
                    risk_dist = max_h - entry_price
                    reward_dist = entry_price - min_l
                
                if risk_dist <= 0: risk_dist = 0.00001
                rr = reward_dist / risk_dist
                
                is_trade = 1 if rr >= MIN_RR else 0
                
                all_samples_list.append({
                    'timestamp': timestamps[b_start + local_idx],
                    'asset': asset,
                    'direction': direction,
                    'features': single_pair_obs[local_idx].tolist(),
                    'label': is_trade,
                    'rr': rr,
                    'risk_pips': risk_dist,
                    'reward_pips': reward_dist
                })
        
        gc.collect()

    if all_samples_list:
        logger.info(f"Creating DataFrame from {len(all_samples_list)} samples...")
        final_df = pd.DataFrame(all_samples_list)
        logger.info(f"Saving to {output_file}...")
        final_df.to_parquet(output_file, index=False)
        logger.info(f"Saved {len(final_df)} samples to {output_file}")
        logger.info(f"Class distribution: {final_df['label'].value_counts(normalize=True).to_dict()}")
    else:
        logger.warning("No samples generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=os.path.join(PROJECT_ROOT, "data"))
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "sl_risk_dataset.parquet"))
    args = parser.parse_args()
    generate_dataset(args.data, args.output)
