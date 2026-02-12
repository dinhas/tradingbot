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

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Defaults
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "ppo_final_model.zip")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "risk_dataset.parquet")
LOOKAHEAD_STEPS = 30 # Reduced to 30 steps (2.5 hours) for short-term risk management
BATCH_SIZE = 50000  # Increased batch size as we use less memory now

def build_observation_matrix(df, assets, start_idx, end_idx):
    """
    Constructs the full observation matrix (N x 140) efficiently using vectorized operations.
    Returns:
        np.ndarray: The observation matrix of shape (end_idx - start_idx, 140)
    """
    num_rows = end_idx - start_idx
    # Pre-allocate matrix with float32
    obs_matrix = np.zeros((num_rows, 140), dtype=np.float32)
    
    # Slice the dataframe once
    df_slice = df.iloc[start_idx:end_idx]
    
    current_col = 0
    
    # 1. Per-Asset Features (25 * 5 = 125 columns)
    for asset in assets:
        # Define the 13 data columns in order
        data_cols = [
            f"{asset}_close", f"{asset}_return_1", f"{asset}_return_12",
            f"{asset}_atr_14", f"{asset}_atr_ratio", f"{asset}_bb_position",
            f"{asset}_ema_9", f"{asset}_ema_21", f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21",
            f"{asset}_rsi_14", f"{asset}_macd_hist", f"{asset}_volume_ratio"
        ]
        
        # Copy data columns
        # Using .values ensures we get numpy array
        obs_matrix[:, current_col:current_col+13] = df_slice[data_cols].values.astype(np.float32)
        current_col += 13
        
        # Portfolio Features (Zero State) - 7 columns
        # Already zeros, just skip
        current_col += 7
        
        # Cross-Asset Features - 5 columns
        cross_cols = [
             f"{asset}_corr_basket", f"{asset}_rel_strength",
             f"{asset}_corr_xauusd", f"{asset}_corr_eurusd", f"{asset}_rank"
        ]
        obs_matrix[:, current_col:current_col+5] = df_slice[cross_cols].values.astype(np.float32)
        current_col += 5

    # 2. Global Features (15 columns)
    
    # Portfolio State (Global) - 4 columns
    # Equity = 10000.0
    obs_matrix[:, current_col] = 10000.0
    current_col += 1
    # Margin, Drawdown, Num Pos - 3 columns (zeros)
    current_col += 3
    
    # Market Regime - 3 columns
    # Risk On Score
    gbp_ret = df_slice.get("GBPUSD_return_1", 0)
    xau_ret = df_slice.get("XAUUSD_return_1", 0)
    # Handle scalar or series
    if isinstance(gbp_ret, (int, float)) and gbp_ret == 0:
        obs_matrix[:, current_col] = 0
    else:
        obs_matrix[:, current_col] = ((gbp_ret + xau_ret) / 2).values.astype(np.float32)
    current_col += 1
    
    # Asset Dispersion
    ret_cols = [f"{a}_return_1" for a in assets]
    obs_matrix[:, current_col] = df_slice[ret_cols].std(axis=1).values.astype(np.float32)
    current_col += 1
    
    # Market Volatility
    atr_ratio_cols = [f"{a}_atr_ratio" for a in assets]
    obs_matrix[:, current_col] = df_slice[atr_ratio_cols].mean(axis=1).values.astype(np.float32)
    current_col += 1
    
    # Session Features - 8 columns
    sess_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'session_asian', 'session_london', 'session_ny', 'session_overlap']
    obs_matrix[:, current_col:current_col+8] = df_slice[sess_cols].values.astype(np.float32)
    current_col += 8
    
    # Verification
    if current_col != 140:
        logger.warning(f"Feature matrix construction mismatch! Expected 140 columns, filled {current_col}")
        
    return obs_matrix

def generate_dataset_batched(model_path, data_dir, output_file, limit=None):
    """Generates the risk dataset using optimized batched inference."""
    
    # 1. Load Model
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading Alpha Model from {model_path}...")
    try:
        model = PPO.load(model_path, device='cpu')
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. Load Data
    logger.info(f"Loading and preprocessing data from {data_dir}...")
    
    # Check data availability logic (same as before)
    required_assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    data_missing = False
    for asset in required_assets:
        p1 = os.path.join(data_dir, f"{asset}_5m.parquet")
        p2 = os.path.join(data_dir, f"{asset}_5m_2025.parquet")
        if not os.path.exists(p1) and not os.path.exists(p2):
            data_missing = True
            break
            
    if data_missing:
        # ... (Auto download logic kept same)
        logger.warning(f"Data missing in {data_dir}. Attempting to download...")
        try:
            import subprocess
            script_dir = os.path.dirname(os.path.abspath(__file__))
            downloader = os.path.join(script_dir, "download_training_data.py")
            if os.path.exists(downloader):
                logger.info(f"Running {downloader}...")
                subprocess.check_call([sys.executable, downloader, "--output", data_dir])
            else:
                logger.error(f"Downloader script not found at {downloader}")
        except Exception as e:
            logger.error(f"Failed to auto-download data: {e}")

    try:
        env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return
        
    df = env.processed_data
    total_rows = len(df)
    logger.info(f"Total data points: {total_rows}")
    
    start_idx = 500
    end_idx = total_rows - LOOKAHEAD_STEPS
    
    if limit:
        end_idx = min(start_idx + limit, end_idx)
    
    if start_idx >= end_idx:
        logger.error("Data too short.")
        return

    # Cache raw arrays for fast lookups
    assets = env.assets
    # Using numpy arrays from the environment for speed
    close_arrays = {a: env.close_arrays[a] for a in assets}
    high_arrays = {a: env.high_arrays[a] for a in assets}
    low_arrays = {a: env.low_arrays[a] for a in assets}
    atr_arrays = {a: env.atr_arrays[a] for a in assets}
    
    # 3. Build Observation Matrix (Vectorized)
    logger.info("Constructing feature matrix...")
    # This might take a moment but is much faster than doing it in a loop
    all_observations = build_observation_matrix(df, assets, start_idx, end_idx)
    logger.info(f"Feature matrix built. Shape: {all_observations.shape}. Size: {all_observations.nbytes / 1024**2:.2f} MB")
    
    # 4. Batch Processing Loop
    logger.info(f"Starting batched inference (Batch Size: {BATCH_SIZE})...")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Remove existing file to start fresh
    if os.path.exists(output_file):
        os.remove(output_file)
        
    # Indices relative to start_idx (0 to num_rows)
    num_rows = end_idx - start_idx
    indices = np.arange(num_rows)
    
    # Batches
    batch_starts = np.arange(0, num_rows, BATCH_SIZE)
    
    total_signals = 0
    timestamps = df.index[start_idx:end_idx]
    
    for b_start in tqdm(batch_starts, desc="Processing Batches"):
        b_end = min(b_start + BATCH_SIZE, num_rows)
        batch_indices = indices[b_start:b_end]
        
        # Get observations for this batch
        batch_obs = all_observations[b_start:b_end]
        
        # 5. Inference per Asset (Alpha models now expect 40-dim input)
        # We collect all asset predictions first
        asset_actions_all = {}
        
        for i, asset in enumerate(assets):
            # Extract 40-dim observation for this asset: [25 asset + 15 global]
            asset_start_idx = i * 25
            asset_obs = batch_obs[:, asset_start_idx : asset_start_idx + 25]
            global_obs = batch_obs[:, 125:140]
            v3_batch_obs = np.concatenate([asset_obs, global_obs], axis=1)
            
            # Prediction
            # SB3 model.predict handles batches automatically
            actions, _ = model.predict(v3_batch_obs, deterministic=True)
            # Stage 1 models might return (N, 5) or (N,). 
            # In generate_risk_dataset, we assume the model output index matches asset index?
            # Actually, most Alpha models predict for ALL assets or for THE asset in observation.
            # Based on ppo_final_model.zip being a 40-dim model, it's likely a per-asset classifier
            # that outputs a single value per observation.
            
            if actions.ndim > 1 and actions.shape[1] > 1:
                # If it's a multi-output model that still takes 40-dim (unlikely but possible)
                asset_actions_all[asset] = actions[:, i]
            else:
                # Standard per-asset model
                asset_actions_all[asset] = actions.flatten()
        
        # 6. Collect Signals
        # Use column-oriented storage (dict of lists) instead of row-oriented (list of dicts)
        # to save memory and avoid object overhead
        batch_results = {
            'timestamp': [],
            'asset': [],
            'direction': [],
            'entry_price': [],
            'atr': [],
            'features': [],
            'max_profit_pct': [],
            'max_loss_pct': [],
            'final_pnl_pct': [],
            'bars_to_exit': [],
            'close_30_price': []
        }
        
        has_data = False
        
        for asset in assets:
            asset_actions = asset_actions_all[asset]
            
            # Vectorized finding of indices
            buy_mask = asset_actions > 0.33
            sell_mask = asset_actions < -0.33
            
            # Combine
            mask = buy_mask | sell_mask
            if not np.any(mask):
                continue
                
            active_local_indices = np.where(mask)[0]
            
            # Global indices in the original dataframe
            # start_idx + b_start + local_index
            global_indices = start_idx + b_start + active_local_indices
            
            for local_idx, global_idx in zip(active_local_indices, global_indices):
                direction = 1 if asset_actions[local_idx] > 0.33 else -1
                
                entry_price = close_arrays[asset][global_idx]
                atr = atr_arrays[asset][global_idx]
                
                # Use 1000-step lookahead for V3 labels
                f_end_step = min(global_idx + LOOKAHEAD_STEPS, len(high_arrays[asset]))
                future_highs = high_arrays[asset][global_idx+1 : f_end_step]
                future_lows = low_arrays[asset][global_idx+1 : f_end_step]
                future_closes = close_arrays[asset][global_idx+1 : f_end_step]
                
                if len(future_highs) == 0: continue
                
                # --- V3 Labels Calculation ---
                # Default Simulation: 2x ATR SL / 4x ATR TP
                sl_dist = 2.0 * atr
                tp_dist = 4.0 * atr
                
                if direction == 1:
                    sl_level = entry_price - sl_dist
                    tp_level = entry_price + tp_dist
                    
                    sl_hits = np.where(future_lows <= sl_level)[0]
                    tp_hits = np.where(future_highs >= tp_level)[0]
                else:
                    sl_level = entry_price + sl_dist
                    tp_level = entry_price - tp_dist
                    
                    sl_hits = np.where(future_highs >= sl_level)[0]
                    tp_hits = np.where(future_lows <= tp_level)[0]
                
                # Find first hit
                first_sl = sl_hits[0] if len(sl_hits) > 0 else 9999
                first_tp = tp_hits[0] if len(tp_hits) > 0 else 9999
                
                if first_sl < first_tp and first_sl != 9999:
                    bars_to_exit = first_sl + 1
                    exit_price = sl_level
                elif first_tp <= first_sl and first_tp != 9999:
                    bars_to_exit = first_tp + 1
                    exit_price = tp_level
                else:
                    # Time exit
                    bars_to_exit = len(future_highs)
                    exit_price = future_closes[-1]
                
                final_pnl_pct = (exit_price - entry_price) * direction / entry_price if entry_price != 0 else 0
                
                # Basic PnL for shadow simulation (still needed by Env)
                max_profit_pct = (np.max(future_highs) - entry_price) / entry_price if direction == 1 else (entry_price - np.min(future_lows)) / entry_price
                max_loss_pct = (np.min(future_lows) - entry_price) / entry_price if direction == 1 else (entry_price - np.max(future_highs)) / entry_price

                # --- V3 Feature Slicing (40-dim) ---
                # Slicing from 140-dim observation: [25 asset features] + [15 global features]
                asset_start_idx = i * 25
                asset_slice = batch_obs[local_idx][asset_start_idx : asset_start_idx + 25]
                global_slice = batch_obs[local_idx][125:140]
                v3_features = np.concatenate([asset_slice, global_slice])

                # Append to lists
                batch_results['timestamp'].append(timestamps[b_start + local_idx])
                batch_results['asset'].append(asset)
                batch_results['direction'].append(direction)
                batch_results['entry_price'].append(entry_price)
                batch_results['atr'].append(atr)
                batch_results['features'].append(v3_features.tolist())
                batch_results['max_profit_pct'].append(max_profit_pct)
                batch_results['max_loss_pct'].append(max_loss_pct)
                batch_results['final_pnl_pct'].append(final_pnl_pct)
                batch_results['bars_to_exit'].append(bars_to_exit)
                batch_results['close_30_price'].append(future_closes[-1])
                
                has_data = True

        if has_data:
            batch_df = pd.DataFrame(batch_results)
            total_signals += len(batch_df)
            
            if os.path.exists(output_file):
                # Append to existing
                existing_df = pd.read_parquet(output_file)
                combined = pd.concat([existing_df, batch_df], ignore_index=True)
                combined.to_parquet(output_file, index=False)
                del existing_df, combined
            else:
                batch_df.to_parquet(output_file, index=False)
            
            del batch_df
        
        del batch_results
        gc.collect()

    logger.info(f"Processing complete. Total signals: {total_signals}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Risk Dataset (Optimized)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of steps to process")
    parser.add_argument("--log_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = args.log_dir if args.log_dir else os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"generate_risk_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_file)
    
    try:
        BATCH_SIZE = args.batch
        logger.info(f"Starting optimized generation (Limit: {args.limit})...")
        generate_dataset_batched(args.model, args.data, args.output, limit=args.limit)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"CRITICAL ERROR: {e}")
    finally:
        tee.close()