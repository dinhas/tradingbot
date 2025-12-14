import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from tqdm import tqdm
import logging
from src.frozen_alpha_env import TradingEnv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Defaults
# Defaults
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "8.03.zip") # Relative path from RiskLayer to checkpoints
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DEFAULT_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "risk_dataset.parquet")
LOOKAHEAD_STEPS = 6 # 30 mins (5m candles)
BATCH_SIZE = 50000  # Process 50k steps at a time to manage memory

def generate_dataset_batched(model_path, data_dir, output_file):
    """Generates the risk dataset using batched vectorized inference."""
    
    # 1. Load Model
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading Alpha Model from {model_path}...")
    try:
        # Load model on CPU
        model = PPO.load(model_path, device='cpu')
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # 2. Load Data
    logger.info(f"Loading and preprocessing data from {data_dir}...")
    
    # Check if data exists, if not try to download
    required_assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    data_missing = False
    for asset in required_assets:
        p1 = os.path.join(data_dir, f"{asset}_5m.parquet")
        p2 = os.path.join(data_dir, f"{asset}_5m_2025.parquet")
        if not os.path.exists(p1) and not os.path.exists(p2):
            data_missing = True
            break
            
    if data_missing:
        logger.warning(f"Data missing in {data_dir}. Attempting to download...")
        try:
            import subprocess
            # Assume download_training_data.py is in same dir as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            downloader = os.path.join(script_dir, "download_training_data.py")
            
            if os.path.exists(downloader):
                logger.info(f"Running {downloader}...")
                subprocess.check_call([sys.executable, downloader, "--output", data_dir])
            else:
                logger.error(f"Downloader script not found at {downloader}")
        except Exception as e:
            logger.error(f"Failed to auto-download data: {e}")
            logger.info("Please run download_training_data.py manually.")

    try:
        env = TradingEnv(data_dir=data_dir, stage=1, is_training=False)
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return
        
    df = env.processed_data
    total_rows = len(df)
    logger.info(f"Total data points: {total_rows}")
    
    # Range of valid data
    start_idx = 500
    end_idx = total_rows - LOOKAHEAD_STEPS
    
    if start_idx >= end_idx:
        logger.error("Data too short for lookahead window.")
        return

    all_signals = []
    
    # Cache raw arrays for fast lookups (Global scope for processing)
    assets = env.assets
    close_arrays = {a: env.close_arrays[a] for a in assets}
    high_arrays = {a: env.high_arrays[a] for a in assets}
    low_arrays = {a: env.low_arrays[a] for a in assets}
    atr_arrays = {a: env.atr_arrays[a] for a in assets}

    # 3. Batch Processing Loop
    logger.info(f"Starting batch processing (Batch Size: {BATCH_SIZE})...")
    
    # Create valid index chunks
    full_indices = range(start_idx, end_idx)
    chunks = [full_indices[i:i + BATCH_SIZE] for i in range(0, len(full_indices), BATCH_SIZE)]
    
    for chunk_indices in tqdm(chunks, desc="Processing Batches"):
        # Slice DataFrame for this chunk
        # Note: We need continuous slice from df, but features are already computed row-wise.
        # So df.iloc[chunk_indices] is valid.
        df_chunk = df.iloc[chunk_indices].copy()
        current_chunk_size = len(df_chunk)
        
        # --- Construct Feature Matrix for Chunk ---
        feature_cols = []
        
        # Pre-calculate chunk metrics
        gbp_ret = df_chunk.get(f"GBPUSD_return_1", 0)
        xau_ret = df_chunk.get(f"XAUUSD_return_1", 0)
        risk_on_score = (gbp_ret + xau_ret) / 2
        
        ret_cols = [f"{a}_return_1" for a in assets]
        asset_dispersion = df_chunk[ret_cols].std(axis=1)
        
        atr_ratio_cols = [f"{a}_atr_ratio" for a in assets]
        market_volatility = df_chunk[atr_ratio_cols].mean(axis=1)
        
        # Per-Asset Features
        zeros = np.zeros(current_chunk_size, dtype=np.float32)
        
        for asset in assets:
            # Data Features
            cols = [
                f"{asset}_close", f"{asset}_return_1", f"{asset}_return_12",
                f"{asset}_atr_14", f"{asset}_atr_ratio", f"{asset}_bb_position",
                f"{asset}_ema_9", f"{asset}_ema_21", f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21",
                f"{asset}_rsi_14", f"{asset}_macd_hist", f"{asset}_volume_ratio"
            ]
            for c in cols:
                feature_cols.append(df_chunk[c].values.astype(np.float32))
            
            # Portfolio Features (Zero State)
            for _ in range(7):
                feature_cols.append(zeros)
                
            # Cross-Asset
            cross_cols = [
                 f"{asset}_corr_basket", f"{asset}_rel_strength",
                 f"{asset}_corr_xauusd", f"{asset}_corr_eurusd", f"{asset}_rank"
            ]
            for c in cross_cols:
                feature_cols.append(df_chunk[c].values.astype(np.float32))
                
        # Global Features
        feature_cols.append(np.full(current_chunk_size, 10000.0, dtype=np.float32)) # equity
        feature_cols.append(zeros) # margin
        feature_cols.append(zeros) # drawdown
        feature_cols.append(zeros) # num_pos
        
        feature_cols.append(risk_on_score.values.astype(np.float32))
        feature_cols.append(asset_dispersion.values.astype(np.float32))
        feature_cols.append(market_volatility.values.astype(np.float32))
        
        sess_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'session_asian', 'session_london', 'session_ny', 'session_overlap']
        for c in sess_cols:
            feature_cols.append(df_chunk[c].values.astype(np.float32))
            
        # Stack
        observations = np.column_stack(feature_cols).astype(np.float32)
        
        # Inference
        actions, _ = model.predict(observations, deterministic=True)
        
        # Process Signals
        chunk_timestamps = df_chunk.index
        
        for i, asset in enumerate(assets):
            asset_actions = actions[:, i]
            
            # Find indices relative to CHUNK
            buy_indices = np.where(asset_actions > 0.33)[0]
            sell_indices = np.where(asset_actions < -0.33)[0]
            
            # Helper to process hits


        # ACTUAL PROCESSING WITH OBSERVATIONS ALIVE
        for i, asset in enumerate(assets):
            asset_actions = actions[:, i]
            buy_indices = np.where(asset_actions > 0.33)[0]
            sell_indices = np.where(asset_actions < -0.33)[0]
            
            for direction, indices in [(1, buy_indices), (-1, sell_indices)]:
                for idx in indices:
                    global_step = chunk_indices[idx]
                    
                    entry_price = close_arrays[asset][global_step]
                    atr = atr_arrays[asset][global_step]
                    
                    end_step = global_step + LOOKAHEAD_STEPS
                    future_highs = high_arrays[asset][global_step+1 : end_step]
                    future_lows = low_arrays[asset][global_step+1 : end_step]
                    future_close = close_arrays[asset][end_step-1]
                    
                    if len(future_highs) == 0: continue
                    max_h = np.max(future_highs)
                    min_l = np.min(future_lows)
                    
                    if direction == 1:
                        max_profit_pct = (max_h - entry_price) / entry_price
                        max_loss_pct = (min_l - entry_price) / entry_price
                    else:
                        max_profit_pct = (entry_price - min_l) / entry_price
                        max_loss_pct = (entry_price - max_h) / entry_price

                    all_signals.append({
                        'timestamp': chunk_timestamps[idx],
                        'asset': asset,
                        'direction': direction,
                        'entry_price': entry_price,
                        'atr': atr,
                        'features': observations[idx].tolist(),
                        'max_profit_pct': max_profit_pct,
                        'max_loss_pct': max_loss_pct,
                        'close_1000_price': future_close
                    })

        # Clean up memory after processing for this batch
        del feature_cols
        del observations

    # 4. Save
    if all_signals:
        result_df = pd.DataFrame(all_signals)
        result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
        
        # Create output dir if needed
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        result_df.to_parquet(output_file)
        logger.info(f"Saved {len(result_df)} rows to {output_file}")
        
    else:
        logger.warning("No signals generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Risk Dataset (Batched)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to model .zip")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="Path to data directory")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE, help="Path to output .parquet")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    
    args = parser.parse_args()
    
    BATCH_SIZE = args.batch
    
    try:
        generate_dataset_batched(args.model, args.data, args.output)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"CRITICAL ERROR: {e}")
