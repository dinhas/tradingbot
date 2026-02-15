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

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "ppo_final_model.zip")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_FILE = os.path.join(BASE_DIR, "data", "sl_risk_dataset.parquet")

# Configuration
LOOKAHEAD_BARS = 30  # 2.5 hours
BATCH_SIZE = 10000

def build_observation_matrix(df, assets, start_idx, end_idx):
    """Returns the processed data slice."""
    return df.iloc[start_idx:end_idx]

def generate_sl_dataset(model_path, data_dir, output_file, limit=None):
    # 1. Load Model
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading Alpha Model from {model_path}...")
    model = PPO.load(model_path, device='cpu')

    # 2. Load Data & Env
    logger.info(f"Loading data from {data_dir}...")
    env = TradingEnv(data_dir=data_dir, stage=3, is_training=False)
    df = env.processed_data
    total_rows = len(df)
    
    start_idx = 500
    end_idx = total_rows - LOOKAHEAD_BARS
    if limit: end_idx = min(start_idx + limit, end_idx)
    
    assets = env.assets
    close_arrays = env.close_arrays
    high_arrays = env.high_arrays
    low_arrays = env.low_arrays
    atr_arrays = env.atr_arrays
    
    # 3. Build Observations
    logger.info("Extracting observations...")
    all_observations_df = build_observation_matrix(df, assets, start_idx, end_idx)
    num_rows = len(all_observations_df)
    
    # 4. Process Batches
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    if os.path.exists(output_file): os.remove(output_file)
    
    batch_starts = np.arange(0, num_rows, BATCH_SIZE)
    total_signals = 0
    timestamps = df.index[start_idx:end_idx]
    
    for b_start in tqdm(batch_starts, desc="Processing Batches"):
        b_end = min(b_start + BATCH_SIZE, num_rows)
        batch_df_slice = all_observations_df.iloc[b_start:b_end]
        
        # Get Alpha Model Actions
        asset_actions_all = {}
        for asset in assets:
            # Construct 40-dim observation for this asset across batch
            obs_batch = []
            for _, row in batch_df_slice.iterrows():
                obs_batch.append(env.feature_engine.get_observation(row, asset=asset))
            obs_batch = np.array(obs_batch, dtype=np.float32)
            
            actions, _ = model.predict(obs_batch, deterministic=True)
            asset_actions_all[asset] = actions.flatten()
        
        batch_results = {
            'timestamp': [], 'asset': [], 'direction': [], 'entry_price': [], 'atr': [],
            'features': [], 'target_sl_mult': [], 'target_tp_mult': [], 'target_size': [],
            'mfe': [], 'mae': []
        }
        
        has_data = False
        for asset in assets:
            asset_actions = asset_actions_all[asset]
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
                
                # --- Feature Extraction (40-dim) ---
                row = batch_df_slice.iloc[local_idx]
                full_features = env.feature_engine.get_observation(row, asset=asset)

                # --- Labeling Logic ---
                if direction == 1: # LONG
                    mfe_idx = np.argmax(future_highs)
                    mfe_dist = future_highs[mfe_idx] - entry_price
                    mae_dist = entry_price - np.min(future_lows[:mfe_idx+1])
                else: # SHORT
                    mfe_idx = np.argmin(future_lows)
                    mfe_dist = entry_price - future_lows[mfe_idx]
                    mae_dist = np.max(future_highs[:mfe_idx+1]) - entry_price
                
                mfe_atr, mae_atr = mfe_dist / atr, mae_dist / atr
                
                # Target SL & TP
                target_sl_mult = np.clip(mae_atr + 0.2, 0.2, 5.0)
                target_tp_mult = np.clip(mfe_atr, 0.1, 10.0)
                
                # Confidence / Size
                rr_ratio = target_tp_mult / max(target_sl_mult, 1e-9)
                target_size = np.clip((mfe_atr - 0.5) / 2.0, 0.0, 1.0) if rr_ratio >= 1.5 else 0.0
                
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
                has_data = True

        if has_data:
            batch_df = pd.DataFrame(batch_results)
            total_signals += len(batch_df)
            if os.path.exists(output_file):
                existing_df = pd.read_parquet(output_file)
                pd.concat([existing_df, batch_df], ignore_index=True).to_parquet(output_file, index=False)
            else:
                batch_df.to_parquet(output_file, index=False)
        gc.collect()

    logger.info(f"Dataset Generation Complete. Saved {total_signals} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    generate_sl_dataset(parser.parse_args().model, parser.parse_args().data, parser.parse_args().output, parser.parse_args().limit)
