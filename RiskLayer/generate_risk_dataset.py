import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints", "alpha")

def get_latest_model_files(models_dir):
    """Finds the latest zip model and its corresponding vecnormalize file."""
    if not os.path.exists(models_dir):
        return None, None
        
    # Check for specific ppo_final_model first
    preferred_model = os.path.join(models_dir, "ppo_final_model.zip")
    if os.path.exists(preferred_model):
        latest_zip = "ppo_final_model.zip"
    else:
        files = [f for f in os.listdir(models_dir) if f.endswith(".zip")]
        if not files:
            return None, None
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
        latest_zip = files[0]
    
    model_path = os.path.join(models_dir, latest_zip)
    
    # Try to find matching normalizer
    # Pattern 1: name_vecnormalize.pkl
    base_name = os.path.splitext(latest_zip)[0]
    norm_name = f"{base_name}_vecnormalize.pkl"
    norm_path = os.path.join(models_dir, norm_name)
    
    # Pattern 2: if name is ppo_final_model, check ppo_final_vecnormalize.pkl
    if not os.path.exists(norm_path):
        alt_base = base_name.replace("_model", "")
        norm_name = f"{alt_base}_vecnormalize.pkl"
        norm_path = os.path.join(models_dir, norm_name)
    
    if not os.path.exists(norm_path):
        norm_path = None
        
    return model_path, norm_path

DEFAULT_MODEL_PATH, DEFAULT_VEC_NORM_PATH = get_latest_model_files(MODELS_DIR)

# Fallback if no model found (prevents crash on import, fails on run)
if DEFAULT_MODEL_PATH is None:
    DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoints", "model_not_found.zip")
    DEFAULT_VEC_NORM_PATH = None

class SimpleDummyEnv(gym.Env):
    """Minimal env for VecNormalize loading."""
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    def reset(self, seed=None): return np.zeros((40,), dtype=np.float32), {}
    def step(self, action): return np.zeros((40,), dtype=np.float32), 0, False, False, {}

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_FILE = os.path.join(PROJECT_ROOT, "risk_dataset.parquet")
LOOKAHEAD_STEPS = 1000 # 1000 steps (5m candles) to match frozen env
BATCH_SIZE = 5000  # Further reduced batch size

def build_master_feature_cache(df, assets, start_idx, end_idx):
    """
    Constructs the master feature cache (N x 140) efficiently using vectorized operations.
    This cache contains features for ALL pairs (5 pairs * 25 features) + 15 Global Features.
    It is used to efficiently slice single-pair observations (40 features) during batch processing.
    
    Returns:
        np.ndarray: The master feature matrix of shape (end_idx - start_idx, 140)
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

def generate_dataset_batched(model_path, data_dir, output_file, vec_norm_path=None):
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

    # 1b. Load Normalizer (VecNormalize)
    norm_env = None
    if vec_norm_path and os.path.exists(vec_norm_path):
        logger.info(f"Loading VecNormalize stats from {vec_norm_path}...")
        try:
            # VecNormalize requires a venv to load
            dummy_venv = DummyVecEnv([lambda: SimpleDummyEnv()]) 
            norm_env = VecNormalize.load(vec_norm_path, dummy_venv)
            norm_env.training = False # Do not update stats during inference
            norm_env.norm_reward = False
        except Exception as e:
            logger.error(f"Failed to load VecNormalize: {e}")
            logger.warning("Proceeding WITHOUT normalization! (Results may be poor)")
    else:
        logger.warning(f"VecNormalize file not found at {vec_norm_path}. Proceeding unnormalized.")

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
    logger.info("Constructing MASTER feature cache (140 cols) for efficient slicing...")
    # This might take a moment but is much faster than doing it in a loop
    all_observations = build_master_feature_cache(df, assets, start_idx, end_idx)
    logger.info(f"Master cache built. Shape: {all_observations.shape}. Size: {all_observations.nbytes / 1024**2:.2f} MB")
    
    # 4. Batch Processing Loop
    logger.info(f"Starting batched inference (Batch Size: {BATCH_SIZE})...")
    logger.info("Generating samples with 40 features (25 Asset + 15 Global) per row.")
    
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
    
    # Validation Stats (1x SL / 5x TP)
    stats_wins = 0
    stats_losses = 0
    
    timestamps = df.index[start_idx:end_idx]
    
    # Store all batch dataframes to concat at the end
    all_batch_dfs = []
    
    for b_start in tqdm(batch_starts, desc="Processing Batches"):
        b_end = min(b_start + BATCH_SIZE, num_rows)
        batch_indices = indices[b_start:b_end]
        
        # Master observation (140 features)
        master_batch_obs = all_observations[b_start:b_end]
        
        # Prepare storage for results
        batch_results = {
            'timestamp': [],
            'asset': [],
            'direction': [],
            'entry_price': [],
            'atr': [],
            'features': [],
            'max_profit_pct': [],
            'max_loss_pct': [],
            'close_1000_price': []
        }
        
        has_data = False
        
        # Iterating per asset to construct single-pair observations
        for i, asset in enumerate(assets):
            # 1. Extract 40 features for this asset
            # [25 asset-specific]
            asset_col_start = i * 25
            asset_col_end = asset_col_start + 25
            asset_feats = master_batch_obs[:, asset_col_start:asset_col_end]
            
            # [15 global features] - they are at indices 125-140
            global_feats = master_batch_obs[:, 125:140]
            
            # Combine -> (Batch, 40)
            single_pair_obs = np.concatenate([asset_feats, global_feats], axis=1)
            
            # Apply Normalization if available
            if norm_env is not None:
                # normalize_obs expects shape (n_envs, n_features)
                # Our single_pair_obs is (Batch, 40), so it works like a vectorized env with 'Batch' environments
                single_pair_obs = norm_env.normalize_obs(single_pair_obs)

            # 2. Inference (Single Pair Model)
            # Output shape: (Batch, 1)
            actions, _ = model.predict(single_pair_obs, deterministic=True)
            
            # Flatten to (Batch,)
            asset_actions = actions.flatten()
            
            # 3. Signal Processing
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
                
                # Lookahead logic
                f_end_step = global_idx + LOOKAHEAD_STEPS
                # Safety check
                if f_end_step >= len(high_arrays[asset]):
                    continue
                    
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

                # --- Validation: Calculate Win Rate for Default 1x SL / 5x TP ---
                sl_pips = 1.0 * atr
                tp_pips = 5.0 * atr
                
                # Default Logic: Check what hit first
                # We need exact timing, so we scan the arrays
                # future_highs/lows are arrays of shape (LOOKAHEAD_STEPS-1,)
                
                did_win = False
                did_loss = False
                
                if direction == 1: # LONG
                    sl_price = entry_price - sl_pips
                    tp_price = entry_price + tp_pips
                    
                    # Boolean masks
                    sl_hits = future_lows <= sl_price
                    tp_hits = future_highs >= tp_price
                else: # SHORT
                    sl_price = entry_price + sl_pips
                    tp_price = entry_price - tp_pips
                    
                    # Boolean masks
                    sl_hits = future_highs >= sl_price
                    tp_hits = future_lows <= tp_price
                
                # Find first index
                first_sl = np.argmax(sl_hits) if np.any(sl_hits) else 99999
                first_tp = np.argmax(tp_hits) if np.any(tp_hits) else 99999
                
                if first_sl == 99999 and first_tp == 99999:
                    pass # Neither hit
                elif first_tp < first_sl:
                    stats_wins += 1
                else:
                    stats_losses += 1

                # --- Construct Reduced Feature Vector (40 dims) ---
                # We already built this for inference: single_pair_obs[local_idx]
                reduced_features = single_pair_obs[local_idx]

                # Append to lists
                batch_results['timestamp'].append(timestamps[b_start + local_idx])
                batch_results['asset'].append(asset)
                batch_results['direction'].append(direction)
                batch_results['entry_price'].append(entry_price)
                batch_results['atr'].append(atr)
                batch_results['features'].append(reduced_features.tolist()) # Save 40 dims
                batch_results['max_profit_pct'].append(max_profit_pct)
                batch_results['max_loss_pct'].append(max_loss_pct)
                batch_results['close_1000_price'].append(future_close)
                
                has_data = True

        if has_data:
            batch_df = pd.DataFrame(batch_results)
            all_batch_dfs.append(batch_df)
            total_signals += len(batch_df)
            del batch_df
        
        del batch_results
        gc.collect()

    logger.info(f"Processing complete. Total signals: {total_signals}")
    
    # Log Win Rate
    total_closed = stats_wins + stats_losses
    if total_closed > 0:
        win_rate = (stats_wins / total_closed) * 100.0
        logger.info(f"--- DATASET QUALITY CHECK ---")
        logger.info(f"Default Strategy (1x ATR SL / 5x ATR TP):")
        logger.info(f"Wins: {stats_wins}")
        logger.info(f"Losses: {stats_losses}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"-----------------------------")
    else:
        logger.warning("No trades closed within lookahead window for validation.")

    if all_batch_dfs:
        logger.info(f"Concatenating {len(all_batch_dfs)} batches and saving to {output_file}...")
        final_df = pd.concat(all_batch_dfs, ignore_index=True)
        final_df.to_parquet(output_file, index=False)
        logger.info(f"Saved dataset to {output_file} ({len(final_df)} rows).")
    else:
        logger.warning("No signals generated. No output file created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Risk Dataset (Optimized)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vec_norm", type=str, default=DEFAULT_VEC_NORM_PATH)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--log_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup logging
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = args.log_dir if args.log_dir else os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"generate_risk_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_file)
    
    try:
        BATCH_SIZE = args.batch
        logger.info("Starting optimized generation...")
        generate_dataset_batched(args.model, args.data, args.output, args.vec_norm)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"CRITICAL ERROR: {e}")
    finally:
        tee.close()