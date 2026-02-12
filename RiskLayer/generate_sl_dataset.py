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

# Add numpy 1.x/2.x compatibility shim
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "ppo_final_model.zip")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_FILE = os.path.join(BASE_DIR, "data", "sl_risk_dataset.parquet")

# Configuration
LOOKAHEAD_BARS = 30  # 30 * 5m = 150 minutes = 2.5 hours
BATCH_SIZE = 10000

def build_observation_matrix(df, assets, start_idx, end_idx):
    """
    Constructs the full observation matrix (N x 140) efficiently.
    Same logic as generate_risk_dataset.py to ensure feature consistency.
    """
    num_rows = end_idx - start_idx
    obs_matrix = np.zeros((num_rows, 140), dtype=np.float32)
    df_slice = df.iloc[start_idx:end_idx]
    
    current_col = 0
    
    # 1. Per-Asset Features (25 * 5 = 125 columns)
    for asset in assets:
        data_cols = [
            f"{asset}_close", f"{asset}_return_1", f"{asset}_return_12",
            f"{asset}_atr_14", f"{asset}_atr_ratio", f"{asset}_bb_position",
            f"{asset}_ema_9", f"{asset}_ema_21", f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21",
            f"{asset}_rsi_14", f"{asset}_macd_hist", f"{asset}_volume_ratio"
        ]
        obs_matrix[:, current_col:current_col+13] = df_slice[data_cols].values.astype(np.float32)
        current_col += 13
        current_col += 7 # Portfolio Features (Zero)
        
        cross_cols = [
             f"{asset}_corr_basket", f"{asset}_rel_strength",
             f"{asset}_corr_xauusd", f"{asset}_corr_eurusd", f"{asset}_rank"
        ]
        obs_matrix[:, current_col:current_col+5] = df_slice[cross_cols].values.astype(np.float32)
        current_col += 5

    # 2. Global Features (15 columns)
    obs_matrix[:, current_col] = 10000.0 # Equity
    current_col += 1
    current_col += 3 # Margin, DD, Num Pos
    
    gbp_ret = df_slice.get("GBPUSD_return_1", 0)
    xau_ret = df_slice.get("XAUUSD_return_1", 0)
    if isinstance(gbp_ret, (int, float)) and gbp_ret == 0:
        obs_matrix[:, current_col] = 0
    else:
        obs_matrix[:, current_col] = ((gbp_ret + xau_ret) / 2).values.astype(np.float32)
    current_col += 1
    
    ret_cols = [f"{a}_return_1" for a in assets]
    obs_matrix[:, current_col] = df_slice[ret_cols].std(axis=1).values.astype(np.float32)
    current_col += 1
    
    atr_ratio_cols = [f"{a}_atr_ratio" for a in assets]
    obs_matrix[:, current_col] = df_slice[atr_ratio_cols].mean(axis=1).values.astype(np.float32)
    current_col += 1
    
    sess_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'session_asian', 'session_london', 'session_ny', 'session_overlap']
    obs_matrix[:, current_col:current_col+8] = df_slice[sess_cols].values.astype(np.float32)
    current_col += 8
    
    return obs_matrix

def generate_sl_dataset(model_path, data_dir, output_file, limit=None):
    """
    Generates the Supervised Learning dataset with optimal SL/TP/Size labels.
    """
    
    # 1. Load Model
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading Alpha Model from {model_path}...")
    model = PPO.load(model_path, device='cpu')

    # 2. Load Data
    logger.info(f"Loading and preprocessing data from {data_dir}...")
    try:
        env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return
        
    df = env.processed_data
    total_rows = len(df)
    logger.info(f"Total data points: {total_rows}")
    
    start_idx = 500
    end_idx = total_rows - LOOKAHEAD_BARS
    if limit: end_idx = min(start_idx + limit, end_idx)
    
    # Cache raw arrays
    assets = env.assets
    close_arrays = {a: env.close_arrays[a] for a in assets}
    high_arrays = {a: env.high_arrays[a] for a in assets}
    low_arrays = {a: env.low_arrays[a] for a in assets}
    atr_arrays = {a: env.atr_arrays[a] for a in assets}
    
    # 3. Build Features
    logger.info("Constructing feature matrix...")
    all_observations = build_observation_matrix(df, assets, start_idx, end_idx)
    
    # 4. Process Batches
    logger.info(f"Starting batched inference and labeling (Lookahead: {LOOKAHEAD_BARS} bars / 2.5h)...")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    if os.path.exists(output_file): os.remove(output_file)
    
    indices = np.arange(num_rows := end_idx - start_idx)
    batch_starts = np.arange(0, num_rows, BATCH_SIZE)
    
    total_signals = 0
    timestamps = df.index[start_idx:end_idx]
    
    for b_start in tqdm(batch_starts, desc="Processing Batches"):
        b_end = min(b_start + BATCH_SIZE, num_rows)
        batch_obs = all_observations[b_start:b_end]
        
        # Get Model Actions
        asset_actions_all = {}
        for i, asset in enumerate(assets):
            asset_start_idx = i * 25
            asset_obs = batch_obs[:, asset_start_idx : asset_start_idx + 25]
            global_obs = batch_obs[:, 125:140]
            v3_batch_obs = np.concatenate([asset_obs, global_obs], axis=1)
            actions, _ = model.predict(v3_batch_obs, deterministic=True)
            if actions.ndim > 1 and actions.shape[1] > 1:
                asset_actions_all[asset] = actions[:, i]
            else:
                asset_actions_all[asset] = actions.flatten()
        
        # Prepare Batch Results
        batch_results = {
            'timestamp': [], 'asset': [], 'direction': [], 'entry_price': [], 'atr': [],
            'features': [], 
            # Labels
            'target_sl_mult': [], 'target_tp_mult': [], 'target_size': [],
            # Debug Stats
            'mfe': [], 'mae': [], 'quality_ratio': []
        }
        
        has_data = False
        
        for asset in assets:
            asset_actions = asset_actions_all[asset]
            # Filter Signals
            mask = (asset_actions > 0.33) | (asset_actions < -0.33)
            if not np.any(mask): continue
                
            active_local_indices = np.where(mask)[0]
            global_indices = start_idx + b_start + active_local_indices
            
            for local_idx, global_idx in zip(active_local_indices, global_indices):
                direction = 1 if asset_actions[local_idx] > 0.33 else -1
                entry_price = close_arrays[asset][global_idx]
                atr = atr_arrays[asset][global_idx]
                if atr == 0: continue

                # Future Window
                f_end_step = min(global_idx + LOOKAHEAD_BARS, len(high_arrays[asset]))
                future_highs = high_arrays[asset][global_idx+1 : f_end_step]
                future_lows = low_arrays[asset][global_idx+1 : f_end_step]
                
                if len(future_highs) == 0: continue
                
                # --- Feature Extraction (60-dim) ---
                # Move this up so we can use features (like ATR Ratio) for dynamic labeling
                asset_start_idx = i * 25
                asset_slice = batch_obs[local_idx][asset_start_idx : asset_start_idx + 25]
                global_slice = batch_obs[local_idx][125:140]
                
                # Synthetic Account/History (20 dims)
                syn_account = np.array([1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
                syn_pnl = np.zeros(5, dtype=np.float32)
                syn_acts = np.zeros(10, dtype=np.float32)
                
                full_features = np.concatenate([asset_slice, global_slice, syn_account, syn_pnl, syn_acts])

                # --- SNIPER ORACLE LABELING LOGIC ---
                
                # 1. Find the peak profit (MFE) and the bar where it occurred
                if direction == 1: # LONG
                    mfe_idx = np.argmax(future_highs)
                    mfe_price = future_highs[mfe_idx]
                    mfe_dist = mfe_price - entry_price
                    
                    # 2. Look for the maximum drawdown ONLY until that peak was reached
                    sub_lows = future_lows[:mfe_idx+1]
                    mae_price = np.min(sub_lows)
                    mae_dist = entry_price - mae_price
                else: # SHORT
                    mfe_idx = np.argmin(future_lows)
                    mfe_price = future_lows[mfe_idx]
                    mfe_dist = entry_price - mfe_price
                    
                    # 2. Look for the maximum drawdown until that peak
                    sub_highs = future_highs[:mfe_idx+1]
                    mae_price = np.max(sub_highs)
                    mae_dist = mae_price - entry_price
                
                # Normalize by ATR
                mfe_atr = mfe_dist / atr
                mae_atr = mae_dist / atr
                
                # 3. Target SL & TP with DYNAMIC BUFFER
                # ATR Ratio is feature index 4 in the asset_slice
                atr_ratio = asset_slice[4]
                # Dynamic Buffer: Scales from 0.1 to 0.4 depending on market regime
                dynamic_buffer = 0.1 + (0.1 * np.clip(atr_ratio, 0.5, 3.0))
                
                target_sl_mult = mae_atr + dynamic_buffer
                target_tp_mult = mfe_atr
                
                # 4. RR Ratio check: TP / SL must be at least 2.0
                rr_ratio = target_tp_mult / max(target_sl_mult, 1e-9)
                
                # Cap Targets to realistic bounds for model stability
                target_sl_mult = np.clip(target_sl_mult, 0.2, 5.0)
                target_tp_mult = np.clip(target_tp_mult, 0.1, 10.0)
                
                # 5. Target Size / Quality
                if (mfe_atr + mae_atr) > 0:
                    quality_ratio = mfe_atr / (mfe_atr + mae_atr)
                else:
                    quality_ratio = 0.0
                
                # Map quality to size (0.0 to 1.0)
                target_size = np.clip((quality_ratio - 0.4) * 1.66, 0.0, 1.0)
                
                # CRITICAL: If RR < 2.0, set size to 0.00
                if rr_ratio < 2.0:
                    target_size = 0.0
                
                # Hard Filter: If MFE < 0.5 ATR, it's just noise
                if mfe_atr < 0.5:
                    target_size = 0.0
                
                # Append results
                batch_results['timestamp'].append(timestamps[b_start + local_idx])
                batch_results['asset'].append(asset)
                batch_results['direction'].append(direction)
                batch_results['entry_price'].append(entry_price)
                batch_results['atr'].append(atr)
                batch_results['features'].append(full_features.tolist())
                
                batch_results['target_sl_mult'].append(target_sl_mult)
                batch_results['target_tp_mult'].append(target_tp_mult)
                batch_results['target_size'].append(target_size)
                
                batch_results['mfe'].append(mfe_atr)
                batch_results['mae'].append(mae_atr)
                batch_results['quality_ratio'].append(quality_ratio)
                
                has_data = True

        if has_data:
            batch_df = pd.DataFrame(batch_results)
            total_signals += len(batch_df)
            
            if os.path.exists(output_file):
                # Append is not supported by all engines, so we read and concat
                existing_df = pd.read_parquet(output_file)
                combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
                combined_df.to_parquet(output_file, index=False)
                del existing_df, combined_df
            else:
                batch_df.to_parquet(output_file, index=False)
            
            del batch_df
        del batch_results
        gc.collect()

    logger.info(f"Dataset Generation Complete. Saved {total_signals} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    generate_sl_dataset(args.model, args.data, args.output, args.limit)
