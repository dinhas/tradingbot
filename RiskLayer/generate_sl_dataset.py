import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
import logging
import shutil
import gc

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)

from RiskLayer.src.feature_engine import FeatureEngine as RiskFeatureEngine
from Alpha.src.model import AlphaSLModel
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Alpha", "models", "alpha_model.pth")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_FILE = os.path.join(BASE_DIR, "data", "sl_risk_dataset.parquet")

# Configuration
MAX_LOOKAHEAD = 500  # Max bars to look forward
BATCH_SIZE = 5000

def simulate_slippage(df, asset, atr_col):
    """Slippage estimate = rolling mean of (|open - prev_close|) in ATR units."""
    open_ = df[f"{asset}_open"]
    prev_close = df[f"{asset}_close"].shift(1)
    atr = df[atr_col]
    slippage = (open_ - prev_close).abs() / (atr + 1e-9)
    return slippage.rolling(100).mean().fillna(0.1) # Default 0.1 ATR slippage

def generate_sl_dataset(model_path, data_dir, output_file, limit=None, max_samples=None):
    # 1. Load Alpha Model (V2: 3 outputs)
    if not os.path.exists(model_path):
        logger.error(f"Alpha model not found at {model_path}")
        return

    logger.info(f"Loading Alpha Model from {model_path}...")
    alpha_model = AlphaSLModel(input_dim=40)
    alpha_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    alpha_model.eval()

    # 2. Load and Preprocess Data
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    data_dict = {}
    for asset in assets:
        path = os.path.join(data_dir, f"{asset}_5m.parquet")
        if not os.path.exists(path):
            path = os.path.join(data_dir, f"{asset}_5m_2025.parquet")
        data_dict[asset] = pd.read_parquet(path)
    
    logger.info("Preprocessing data with RiskFeatureEngine...")
    risk_engine = RiskFeatureEngine()
    raw_df, proc_df = risk_engine.preprocess_data(data_dict)
    
    logger.info("Preprocessing data with AlphaFeatureEngine for model inputs...")
    alpha_engine = AlphaFeatureEngine()
    _, alpha_proc_df = alpha_engine.preprocess_data(data_dict)
    
    # 3. Setup for Labeling
    total_rows = len(proc_df)
    start_idx = 500
    end_idx = total_rows - 50 # Leave room for lookahead
    if limit: end_idx = min(start_idx + limit, end_idx)
    
    # Pre-calculate slippage for all assets
    slippage_dict = {}
    for asset in assets:
        slippage_dict[asset] = simulate_slippage(raw_df, asset, f"{asset}_atr_14").values
    
    # 4. Process in Batches
    # Determine parquet engine and strategy
    try:
        import fastparquet
        use_fastparquet = True
        logger.info("Using fastparquet for efficient appending.")
    except ImportError:
        use_fastparquet = False
        logger.info("fastparquet not found. Switching to directory-based partitioned dataset (pyarrow compatible).")

    # If using directory mode, output_file will be treated as the directory name
    if not use_fastparquet:
        # If it was a file, remove it
        if os.path.exists(output_file) and os.path.isfile(output_file):
            os.remove(output_file)
        os.makedirs(output_file, exist_ok=True)
        # Clean existing chunks
        for f in os.listdir(output_file):
            if f.endswith('.parquet'):
                os.remove(os.path.join(output_file, f))
    else:
        # File mode
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        if os.path.exists(output_file): 
            if os.path.isdir(output_file):
                shutil.rmtree(output_file)
            else:
                os.remove(output_file)
    
    batch_starts = np.arange(start_idx, end_idx, BATCH_SIZE)
    total_signals = 0
    batch_count = 0
    
    for b_start in tqdm(batch_starts, desc="Generating Dataset"):
        if max_samples and total_signals >= max_samples: break
        
        b_end = min(b_start + BATCH_SIZE, end_idx)
        batch_proc = proc_df.iloc[b_start:b_end]
        batch_alpha = alpha_proc_df.iloc[b_start:b_end]
        
        results = []
        
        for asset in assets:
            # Get Alpha Inputs (40 features)
            alpha_obs = alpha_engine.get_observation_vectorized(batch_alpha, asset)
            
            # Get Alpha Predictions
            with torch.no_grad():
                dir_logits, quality, meta_logits = alpha_model(torch.from_numpy(alpha_obs))
                
            # Convert to probabilities / classes
            dir_probs = torch.softmax(dir_logits, dim=1).numpy()
            dir_pred = np.argmax(dir_probs, axis=1) - 1 # {-1, 0, 1}
            qual_val = torch.sigmoid(quality).numpy().flatten()
            meta_val = torch.sigmoid(meta_logits).numpy().flatten()
            
            # Filter Signals
            # - meta >= 0.30
            # - quality >= 0.30
            # - abs(direction) > 0.20 (weighted direction)
            weighted_dir = (dir_probs[:, 2] - dir_probs[:, 0])
            
            mask = (meta_val >= 0.30) & (qual_val >= 0.30) & (np.abs(weighted_dir) > 0.20)
            valid_indices = np.where(mask)[0]
            
            for local_idx in valid_indices:
                global_idx = b_start + local_idx
                if global_idx >= end_idx: continue
                
                direction = 1 if weighted_dir[local_idx] > 0 else -1
                atr = raw_df[f"{asset}_atr_14"].iloc[global_idx]
                entry_price = raw_df[f"{asset}_close"].iloc[global_idx]
                spread = proc_df[f"{asset}_spread"].iloc[global_idx] # Raw spread value
                
                # EXECUTION AWARE LABELING
                # Simulation: Bid/Ask
                # LONG Entry: Buy at Ask. Exit at Bid.
                # SHORT Entry: Sell at Bid. Exit at Ask.
                # For simplicity in this generator, entry is at 'entry_price' (close).
                # We adjust the TP/SL levels to be bid/ask aware.
                
                # Future Window
                f_start = global_idx + 1
                f_end = min(global_idx + MAX_LOOKAHEAD, total_rows)
                future_highs = raw_df[f"{asset}_high"].values[f_start:f_end]
                future_lows = raw_df[f"{asset}_low"].values[f_start:f_end]
                
                if len(future_highs) < 5: continue
                
                # ATR-normalized MFE/MAE calculation
                if direction == 1: # LONG
                    # TP hit if Bid_high >= TP. Bid_high = High - spread
                    # SL hit if Bid_low <= SL. Bid_low = Low - spread
                    bid_highs = future_highs - spread
                    bid_lows = future_lows - spread
                    
                    mfe_idx = np.argmax(bid_highs)
                    mfe_dist = bid_highs[mfe_idx] - entry_price
                    mae_dist = entry_price - np.min(bid_lows[:mfe_idx+1])
                else: # SHORT
                    # TP hit if Ask_low <= TP. Ask_low = Low + spread
                    # SL hit if Ask_high >= SL. Ask_high = High + spread
                    ask_highs = future_highs + spread
                    ask_lows = future_lows + spread
                    
                    mfe_idx = np.argmin(ask_lows)
                    mfe_dist = entry_price - ask_lows[mfe_idx]
                    mae_dist = np.max(ask_highs[:mfe_idx+1]) - entry_price
                
                mfe_atr = mfe_dist / (atr + 1e-9)
                mae_atr = mae_dist / (atr + 1e-9)
                
                # --- TARGETS ---
                target_sl_mult = np.clip(mae_atr + 0.2, 0.5, 4.0)
                target_tp_mult = np.clip(mfe_atr * 0.8, 0.5, 8.0) # Conservative TP
                
                # Execution Buffer
                # execution_buffer = (spread / ATR) + slippage
                slippage = slippage_dict[asset][global_idx]
                target_exec_buffer = (spread / (atr + 1e-9)) + slippage
                
                # Expected Value Logic
                # transaction_cost = spread + slippage (in ATR units)
                tx_cost_atr = (spread / (atr + 1e-9)) + slippage
                
                # If we assume P_win = 0.5 for neutral sizing, or we can use the "best" outcome
                # Here we label "tp_first" as 1 if MFE reached 2*MAE before MAE hit
                target_tp_first = 1 if mfe_atr > (2.0 * mae_atr) else 0
                
                # EV = (P_win * TP) - (P_loss * SL) - Costs
                # For labeling, we use the actual realized outcomes
                realized_ev = mfe_atr - mae_atr - tx_cost_atr
                target_expected_value = realized_ev
                
                # Position Size
                # target_size = max(EV / SL_mult, 0)
                target_size = np.clip(realized_ev / (target_sl_mult + 1e-9), 0, 1)
                
                # Update Risk Features in observation
                obs_vector = risk_engine.get_observation(proc_df.iloc[global_idx], {}, asset)
                # Feature indices for alpha outputs (from feature_engine.py):
                # 40: alpha_direction, 41: alpha_meta, 42: alpha_quality
                obs_vector[40] = weighted_dir[local_idx]
                obs_vector[41] = meta_val[local_idx]
                obs_vector[42] = qual_val[local_idx]
                
                results.append({
                    'timestamp': proc_df.index[global_idx],
                    'asset': asset,
                    'features': obs_vector.tolist(),
                    'target_sl_mult': float(target_sl_mult),
                    'target_tp_mult': float(target_tp_mult),
                    'target_size': float(target_size),
                    'target_execution_buffer_mult': float(target_exec_buffer),
                    'target_expected_value': float(target_expected_value),
                    'target_tp_first': int(target_tp_first),
                    'alpha_direction': float(weighted_dir[local_idx]),
                    'alpha_meta': float(meta_val[local_idx]),
                    'alpha_quality': float(qual_val[local_idx])
                })
                total_signals += 1
                if max_samples and total_signals >= max_samples: break
            if max_samples and total_signals >= max_samples: break
            
        if results:
            batch_df = pd.DataFrame(results)
            if use_fastparquet:
                if os.path.exists(output_file):
                    batch_df.to_parquet(output_file, engine='fastparquet', append=True)
                else:
                    batch_df.to_parquet(output_file, engine='fastparquet')
            else:
                # Directory mode: save chunk
                chunk_path = os.path.join(output_file, f"part_{batch_count:05d}.parquet")
                batch_df.to_parquet(chunk_path) # Uses default engine (pyarrow usually)
                batch_count += 1
        
        gc.collect()

    logger.info(f"Dataset Generation Complete. Total Samples: {total_signals}")
    
    # Validation Print
    if total_signals > 0:
        final_df = pd.read_parquet(output_file)
        logger.info("\n--- Dataset Distribution Summary ---")
        logger.info(f"Mean SL Mult: {final_df['target_sl_mult'].mean():.2f}")
        logger.info(f"Mean TP Mult: {final_df['target_tp_mult'].mean():.2f}")
        logger.info(f"Mean EV: {final_df['target_expected_value'].mean():.2f}")
        logger.info(f"Mean Size: {final_df['target_size'].mean():.2f}")
        logger.info(f"TP First Ratio: {final_df['target_tp_first'].mean():.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    generate_sl_dataset(args.model, args.data, args.output, args.limit, args.max_samples)
