import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm
import logging
import gc

# Add project root to sys.path to import Alpha models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.append(PROJECT_ROOT)

from RiskLayer.src.frozen_alpha_env import TradingEnv
from Alpha.src.model import AlphaSLModel
from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine

# Add numpy 1.x/2.x compatibility shim
if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "Alpha", "models", "alpha_model.pth")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_OUTPUT_FILE = os.path.join(BASE_DIR, "data", "sl_risk_dataset.parquet")

# Configuration
LOOKAHEAD_BARS = 300  # Increased for deeper search
BATCH_SIZE = 10000

def get_atr_percentile(env, asset, global_idx):
    """Calculates ATR percentile relative to recent history."""
    lookback = 1000
    if global_idx < lookback: return 0.5
    recent_atrs = env.atr_arrays[asset][global_idx-lookback:global_idx]
    current_atr = env.atr_arrays[asset][global_idx]
    return (recent_atrs < current_atr).mean()

def get_wick_body_ratio(env, asset, global_idx):
    """Calculates recent average wick to body ratio."""
    lookback = 10
    if global_idx < lookback: return 1.0
    
    highs = env.high_arrays[asset][global_idx-lookback:global_idx+1]
    lows = env.low_arrays[asset][global_idx-lookback:global_idx+1]
    closes = env.close_arrays[asset][global_idx-lookback:global_idx+1]
    opens = env.raw_data[f"{asset}_open"].values[global_idx-lookback:global_idx+1]
    
    bodies = np.abs(closes - opens)
    wicks = (highs - np.maximum(opens, closes)) + (np.minimum(opens, closes) - lows)
    
    return np.mean(wicks / (bodies + 1e-9))

def generate_sl_dataset(model_path, data_dir, output_file, limit=None, max_samples=None):
    # 1. Load Alpha Model
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading Alpha Model from {model_path}...")
    model = AlphaSLModel(input_dim=40)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. Load Data & Env
    logger.info(f"Loading data from {data_dir}...")
    env = TradingEnv(data_dir=data_dir, stage=3, is_training=False, skip_preprocess=True)
    env.feature_engine = AlphaFeatureEngine()
    
    if limit:
        logger.info(f"Slicing data to limit: {limit} samples")
        buffer = 300 
        for asset in env.assets:
            env.data[asset] = env.data[asset].iloc[:limit + buffer]
            
    env.raw_data, env.processed_data = env.feature_engine.preprocess_data(env.data)
    env._cache_data_arrays()
    
    df = env.processed_data
    total_rows = len(df)
    
    start_idx = 200 # Buffer for indicators
    end_idx = total_rows - 100 # End buffer
    if limit: end_idx = min(start_idx + limit, end_idx)
    
    assets = env.assets
    close_arrays = env.close_arrays
    high_arrays = env.high_arrays
    low_arrays = env.low_arrays
    atr_arrays = env.atr_arrays
    
    # 3. Process Batches
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    if os.path.exists(output_file): os.remove(output_file)
    
    num_rows = end_idx - start_idx
    batch_starts = np.arange(0, num_rows, BATCH_SIZE)
    total_signals = 0
    timestamps = df.index[start_idx:end_idx]
    
    for b_start in tqdm(batch_starts, desc="Processing Batches"):
        if max_samples and total_signals >= max_samples:
            break

        b_end = min(b_start + BATCH_SIZE, num_rows)
        batch_df_slice = df.iloc[start_idx + b_start : start_idx + b_end]
        
        # 1. Alpha Model Predictions for the batch
        asset_preds = {}
        asset_obs = {}
        for asset in assets:
            obs = env.feature_engine.get_observation_vectorized(batch_df_slice, asset)
            asset_obs[asset] = obs
            
            with torch.no_grad():
                dir_logits, quality, meta_logits = model(torch.from_numpy(obs))
            
            probs = torch.softmax(dir_logits, dim=1)
            asset_preds[asset] = {
                'dir': (probs[:, 2] - probs[:, 0]).numpy(),
                'qual': quality.numpy().flatten(),
                'meta': torch.sigmoid(meta_logits).numpy().flatten()
            }
        
        batch_results = {
            'timestamp': [], 'asset': [], 'direction': [], 'entry_price': [], 'atr': [],
            'alpha_direction': [], 'alpha_quality': [], 'alpha_meta': [],
            'features': [], 
            'target_sl_mult': [], 'target_tp_mult': [], 'target_size_factor': [], 'target_prob_tp_first': []
        }
        
        has_data = False
        for asset in assets:
            if max_samples and total_signals >= max_samples:
                break

            preds = asset_preds[asset]
            obs_batch = asset_obs[asset]
            
            # MANDATORY Alpha v2 filtering
            mask = (preds['meta'] >= 0.30) & (preds['qual'] >= 0.30) & (np.abs(preds['dir']) > 0.20)
            if not np.any(mask): continue
                
            active_indices = np.where(mask)[0]
            
            # Fixed spread per asset
            spread_val = 0.3 if asset == 'XAUUSD' else 0.00002
            
            for local_idx in active_indices:
                if max_samples and total_signals >= max_samples:
                    break
                
                global_idx = start_idx + b_start + local_idx
                direction = 1 if preds['dir'][local_idx] > 0 else -1
                atr = atr_arrays[asset][global_idx]
                if atr <= 0: continue
                
                # 1. SPREAD-AWARE EXECUTION
                # Entry: For LONG use Ask, for SHORT use Bid
                entry_price = close_arrays[asset][global_idx] + (direction * spread_val / 2.0)
                
                # 2. LOOKAHEAD & LABELING (MFE/MAE)
                lookahead = 300
                f_end = min(global_idx + lookahead, len(high_arrays[asset]))
                
                f_highs = high_arrays[asset][global_idx+1 : f_end]
                f_lows = low_arrays[asset][global_idx+1 : f_end]
                f_closes = close_arrays[asset][global_idx+1 : f_end]
                
                if len(f_highs) == 0: continue
                
                # Execution Realistic Prices
                if direction == 1: # LONG
                    bid_highs = f_highs - (spread_val / 2.0)
                    bid_lows = f_lows - (spread_val / 2.0)
                    
                    # MFE Calculation
                    mfe_idx = np.argmax(bid_highs)
                    mfe_val = bid_highs[mfe_idx] - entry_price
                    
                    # MAE Calculation (BEFORE MFE PEAK)
                    if mfe_idx > 0:
                        mae_val = entry_price - np.min(bid_lows[:mfe_idx+1])
                    else:
                        mae_val = entry_price - bid_lows[0]
                else: # SHORT
                    ask_lows = f_lows + (spread_val / 2.0)
                    ask_highs = f_highs + (spread_val / 2.0)
                    
                    # MFE Calculation
                    mfe_idx = np.argmin(ask_lows)
                    mfe_val = entry_price - ask_lows[mfe_idx]
                    
                    # MAE Calculation (BEFORE MFE PEAK)
                    if mfe_idx > 0:
                        mae_val = np.max(ask_highs[:mfe_idx+1]) - entry_price
                    else:
                        mae_val = ask_highs[0] - entry_price
                
                mfe_atr = max(mfe_val / atr, 0.0)
                mae_atr = max(mae_val / atr, 0.0)
                
                # 3. TARGET DEFINITIONS
                target_sl_mult = mae_atr + 0.15
                target_tp_mult = mfe_atr
                
                # 4. PROB TP FIRST SIMULATION
                tp_price = entry_price + (direction * target_tp_mult * atr)
                sl_price = entry_price - (direction * target_sl_mult * atr)
                
                target_prob_tp_first = 0
                for i in range(len(f_highs)):
                    if direction == 1: # LONG
                        if bid_lows[i] <= sl_price: break
                        if bid_highs[i] >= tp_price: 
                            target_prob_tp_first = 1
                            break
                    else: # SHORT
                        if ask_highs[i] >= sl_price: break
                        if ask_lows[i] <= tp_price: 
                            target_prob_tp_first = 1
                            break
                
                # 5. EV-BASED SIZE FACTOR
                # Slippage estimate
                lookback_slip = 20
                if global_idx > lookback_slip:
                    opens = env.raw_data[f"{asset}_open"].values[global_idx-lookback_slip:global_idx+1]
                    prev_closes = env.raw_data[f"{asset}_close"].values[global_idx-lookback_slip-1:global_idx]
                    slippage_estimate_atr = np.mean(np.abs(opens - prev_closes)) / atr
                else:
                    slippage_estimate_atr = 0.05
                
                execution_cost_atr = (spread_val / atr) + slippage_estimate_atr
                
                # Use binary outcome for labeling
                p_win = 1.0 if target_prob_tp_first == 1 else 0.0
                ev = (p_win * target_tp_mult) - ((1 - p_win) * target_sl_mult) - execution_cost_atr
                
                if ev <= 0:
                    target_size_factor = 0.0
                else:
                    target_size_factor = np.clip(ev / target_sl_mult, 0.0, 5.0)

                # 6. ENHANCED FEATURES (8 NEW)
                # alpha features (40)
                alpha_features = obs_batch[local_idx].tolist()
                
                # New features
                atr_p = get_atr_percentile(env, asset, global_idx)
                wb_ratio = get_wick_body_ratio(env, asset, global_idx)
                
                # Spread expansion (dummy/placeholder for now as we have fixed spread in env)
                spread_expansion = 1.0 
                
                extra_features = [
                    preds['dir'][local_idx],
                    preds['meta'][local_idx],
                    preds['qual'][local_idx],
                    spread_val,
                    spread_val / atr,
                    atr_p,
                    wb_ratio,
                    spread_expansion
                ]
                
                combined_features = alpha_features + extra_features
                
                # 7. STORE
                batch_results['timestamp'].append(timestamps[b_start + local_idx])
                batch_results['asset'].append(asset)
                batch_results['direction'].append(direction)
                batch_results['entry_price'].append(entry_price)
                batch_results['atr'].append(atr)
                batch_results['alpha_direction'].append(preds['dir'][local_idx])
                batch_results['alpha_quality'].append(preds['qual'][local_idx])
                batch_results['alpha_meta'].append(preds['meta'][local_idx])
                batch_results['features'].append(combined_features)
                batch_results['target_sl_mult'].append(target_sl_mult)
                batch_results['target_tp_mult'].append(target_tp_mult)
                batch_results['target_size_factor'].append(target_size_factor)
                batch_results['target_prob_tp_first'].append(target_prob_tp_first)
                
                total_signals += 1
                has_data = True

        if has_data:
            batch_df = pd.DataFrame(batch_results)
            if os.path.exists(output_file):
                existing_df = pd.read_parquet(output_file)
                pd.concat([existing_df, batch_df], ignore_index=True).to_parquet(output_file, index=False)
            else:
                batch_df.to_parquet(output_file, index=False)
        gc.collect()

    logger.info(f"Dataset Generation Complete. Saved {total_signals} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR, help="Directory for raw market data")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to Alpha model")
    args = parser.parse_args()
    
    generate_sl_dataset(args.model_path, args.data, args.output, args.limit, args.max_samples)
