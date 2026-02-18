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

from src.frozen_alpha_env import TradingEnv
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
LOOKAHEAD_BARS = 30  # 2.5 hours
BATCH_SIZE = 10000

def build_observation_matrix(df, assets, start_idx, end_idx):
    """Returns the processed data slice."""
    return df.iloc[start_idx:end_idx]

def generate_sl_dataset(model_path, data_dir, output_file, limit=None, max_samples=None):
    # 1. Load Model
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    logger.info(f"Loading Alpha Model from {model_path}...")
    model = AlphaSLModel(input_dim=40)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 2. Load Data & Env
    logger.info(f"Loading data from {data_dir}...")
    env = TradingEnv(data_dir=data_dir, stage=3, is_training=False)
    
    # Switch to Alpha Feature Engine for consistency with Alpha Model
    logger.info("Switching to Alpha Feature Engine...")
    env.feature_engine = AlphaFeatureEngine()
    env.raw_data, env.processed_data = env.feature_engine.preprocess_data(env.data)
    
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
        if max_samples and total_signals >= max_samples:
            break

        b_end = min(b_start + BATCH_SIZE, num_rows)
        batch_df_slice = all_observations_df.iloc[b_start:b_end]
        
        # Get Alpha Model Actions
        asset_predictions_all = {}
        asset_obs_all = {}
        for asset in assets:
            # Vectorized observation extraction
            obs_batch = env.feature_engine.get_observation_vectorized(batch_df_slice, asset)
            asset_obs_all[asset] = obs_batch
            
            # SL Model Prediction
            obs_tensor = torch.from_numpy(obs_batch)
            with torch.no_grad():
                dir_logits, quality, meta_logits = model(obs_tensor)
            
            # Alpha outputs
            probs = torch.softmax(dir_logits, dim=1)
            alpha_dir = (probs[:, 2] - probs[:, 0]).numpy() # Expected value
            alpha_qual = torch.sigmoid(quality).numpy().flatten() if quality.shape[1] == 1 else quality.numpy().flatten()
            alpha_meta = torch.sigmoid(meta_logits).numpy().flatten()
            
            asset_predictions_all[asset] = {
                'dir': alpha_dir,
                'qual': alpha_qual,
                'meta': alpha_meta
            }
        
        batch_results = {
            'timestamp': [], 'asset': [], 'direction': [], 'entry_price': [], 'atr': [],
            'alpha_direction': [], 'alpha_quality': [], 'alpha_meta': [],
            'features': [], 'target_sl_mult': [], 'target_tp_mult': [], 'target_size': [],
            'target_execution_buffer_mult': [], 'target_expected_value': [], 'target_tp_first': [],
            'mfe': [], 'mae': []
        }
        
        has_data = False
        for asset in assets:
            if max_samples and total_signals >= max_samples:
                break

            preds = asset_predictions_all[asset]
            obs_batch = asset_obs_all[asset]
            
            # Alpha v2 thresholds: meta >= 0.30, quality >= 0.30, abs(direction) > 0.20
            mask = (preds['meta'] >= 0.30) & (preds['qual'] >= 0.30) & (np.abs(preds['dir']) > 0.20)
            if not np.any(mask): continue
                
            active_local_indices = np.where(mask)[0]
            global_indices = start_idx + b_start + active_local_indices
            
            # Spread and Slippage estimates
            spread_val = 0.3 if asset == 'XAUUSD' else 0.00002
            
            for local_idx, global_idx in zip(active_local_indices, global_indices):
                if max_samples and total_signals >= max_samples:
                    break

                direction = 1 if preds['dir'][local_idx] > 0 else -1
                raw_entry = close_arrays[asset][global_idx]
                atr = atr_arrays[asset][global_idx]
                if atr == 0: continue

                # Bid/Ask aware entry
                entry_price = raw_entry + (direction * spread_val / 2.0)
                
                # Slippage estimate: rolling mean of (|open - prev_close|) in ATR units
                lookback_slippage = 20
                if global_idx > lookback_slippage:
                    opens = env.raw_data[f"{asset}_open"].values[global_idx-lookback_slippage:global_idx+1]
                    prev_closes = env.raw_data[f"{asset}_close"].values[global_idx-lookback_slippage-1:global_idx]
                    slippage_estimate = np.mean(np.abs(opens - prev_closes)) / atr
                else:
                    slippage_estimate = 0.05
                
                execution_buffer = (spread_val / atr) + slippage_estimate

                # Dynamic Lookahead search
                # Estimate average true range per bar (rolling 100) to gauge volatility
                if global_idx >= 100:
                    lookback_vol = 100
                    tr_window = np.abs(high_arrays[asset][global_idx-lookback_vol:global_idx] - low_arrays[asset][global_idx-lookback_vol:global_idx])
                    avg_tr_per_bar = np.mean(tr_window)
                else:
                    avg_tr_per_bar = atr / 5.0 # Crude fallback
                
                # Target at least 5 ATR potential movement coverage
                dynamic_lookahead = int(np.ceil((5.0 * atr) / (avg_tr_per_bar + 1e-9)))
                dynamic_lookahead = int(np.clip(dynamic_lookahead, 20, 300))
                
                f_end_step = min(global_idx + dynamic_lookahead, len(high_arrays[asset]))
                
                future_opens = env.raw_data[f"{asset}_open"].values[global_idx+1 : f_end_step]
                future_highs = high_arrays[asset][global_idx+1 : f_end_step]
                future_lows = low_arrays[asset][global_idx+1 : f_end_step]
                
                if len(future_highs) == 0: continue
                
                if direction == 1: # LONG
                    bid_highs = future_highs - (spread_val / 2.0)
                    bid_lows = future_lows - (spread_val / 2.0)
                    mae_dist = entry_price - np.min(bid_lows)
                    mfe_dist = np.max(bid_highs) - entry_price
                else: # SHORT
                    ask_lows = future_lows + (spread_val / 2.0)
                    ask_highs = future_highs + (spread_val / 2.0)
                    mae_dist = np.max(ask_highs) - entry_price
                    mfe_dist = entry_price - np.min(ask_lows)

                mfe_atr = max(mfe_dist / atr, 0.1)
                mae_atr = max(mae_dist / atr, 0.1)
                
                target_sl_mult = np.clip(mae_atr + 0.2, 0.2, 5.0)
                target_tp_mult = np.clip(mfe_atr * 0.8, 0.2, 10.0)

                # TP First check
                tp_price = entry_price + (direction * target_tp_mult * atr)
                sl_price = entry_price - (direction * target_sl_mult * atr)
                
                target_tp_first = 0
                for i in range(len(future_highs)):
                    if direction == 1: # LONG
                        if future_lows[i] - (spread_val / 2.0) <= sl_price:
                            break
                        if future_highs[i] - (spread_val / 2.0) >= tp_price:
                            target_tp_first = 1
                            break
                    else: # SHORT
                        if future_highs[i] + (spread_val / 2.0) >= sl_price:
                            break
                        if future_lows[i] + (spread_val / 2.0) <= tp_price:
                            target_tp_first = 1
                            break

                # Expected Value Logic
                transaction_cost = (spread_val / atr) + slippage_estimate
                p_win = 0.6 if target_tp_first == 1 else 0.4 
                ev = (p_win * target_tp_mult) - ((1 - p_win) * target_sl_mult) - transaction_cost
                
                target_size = np.clip(ev / target_sl_mult, 0.0, 1.0)

                batch_results['timestamp'].append(timestamps[b_start + local_idx])
                batch_results['asset'].append(asset)
                batch_results['direction'].append(direction)
                batch_results['entry_price'].append(entry_price)
                batch_results['atr'].append(atr)
                batch_results['alpha_direction'].append(preds['dir'][local_idx])
                batch_results['alpha_quality'].append(preds['qual'][local_idx])
                batch_results['alpha_meta'].append(preds['meta'][local_idx])
                batch_results['features'].append(obs_batch[local_idx].tolist())
                batch_results['target_sl_mult'].append(target_sl_mult)
                batch_results['target_tp_mult'].append(target_tp_mult)
                batch_results['target_size'].append(target_size)
                batch_results['target_execution_buffer_mult'].append(execution_buffer)
                batch_results['target_expected_value'].append(ev)
                batch_results['target_tp_first'].append(target_tp_first)
                batch_results['mfe'].append(mfe_atr)
                batch_results['mae'].append(mae_atr)
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

    # Print distribution summary
    if total_signals > 0:
        final_df = pd.read_parquet(output_file)
        logger.info("\n--- Dataset Distribution Summary ---")
        logger.info(f"Total Samples: {len(final_df)}")
        logger.info(f"SL Mult: Mean={final_df['target_sl_mult'].mean():.2f}, Std={final_df['target_sl_mult'].std():.2f}")
        logger.info(f"TP Mult: Mean={final_df['target_tp_mult'].mean():.2f}, Std={final_df['target_tp_mult'].std():.2f}")
        logger.info(f"EV: Mean={final_df['target_expected_value'].mean():.2f}, Std={final_df['target_expected_value'].std():.2f}")
        logger.info(f"Size: Mean={final_df['target_size'].mean():.2f}, Std={final_df['target_size'].std():.2f}")
        logger.info(f"TP First %: {final_df['target_tp_first'].mean():.1%}")

    logger.info(f"Dataset Generation Complete. Saved {total_signals} samples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    generate_sl_dataset(args.model, args.data, args.output, args.limit, args.max_samples)
