import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Alpha.src.model import AlphaSLModel
from Alpha.src.trading_env import TradingEnv
from backtest.rl_backtest import BacktestMetrics, generate_all_charts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_sl_backtest():
    parser = argparse.ArgumentParser(description="Alpha Supervised Model Backtester (Optimized)")
    parser.add_argument("--model-path", type=str, default="Alpha/models/alpha_model.pth")
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    parser.add_argument("--steps", type=int, default=100000, help="Number of steps to backtest")
    parser.add_argument("--meta-thresh", type=float, default=0.93)
    parser.add_argument("--qual-thresh", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=8192, help="Batch size for inference")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model_path
    data_dir = PROJECT_ROOT / args.data_dir
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    # 1. Initialize Environment
    logger.info(f"Initializing Environment with data from {data_dir}...")
    try:
        env = TradingEnv(data_dir=data_dir, is_training=False)
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        return
    
    # 2. Load Model
    logger.info(f"Loading SL Model from {model_path} to {DEVICE}...")
    model = AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model state: {e}")
        return

    # 3. Pre-calculate Signals (Vectorized Inference)
    logger.info("Pre-calculating signals for all assets...")
    num_assets = len(env.assets)
    master_obs = env.master_obs_matrix # Shape: (N, num_assets * 40)
    N = master_obs.shape[0]
    
    # Reshape to (N * num_assets, 40) for batch inference
    obs_flat = master_obs.reshape(-1, 40)
    
    dir_probs_list = []
    meta_probs_list = []
    qual_scores_list = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(obs_flat), args.batch_size), desc="Inference"):
            batch = torch.from_numpy(obs_flat[i : i + args.batch_size]).to(DEVICE)
            dir_logits, qual_pred, meta_logits = model(batch)
            
            dir_probs_list.append(torch.softmax(dir_logits, dim=1).cpu().numpy())
            meta_probs_list.append(torch.sigmoid(meta_logits).cpu().numpy())
            qual_scores_list.append(qual_pred.cpu().numpy())
            
    dir_probs_all = np.concatenate(dir_probs_list, axis=0).reshape(N, num_assets, 3)
    meta_probs_all = np.concatenate(meta_probs_list, axis=0).reshape(N, num_assets)
    qual_scores_all = np.concatenate(qual_scores_list, axis=0).reshape(N, num_assets)
    
    # 4. Metrics Tracker
    metrics = BacktestMetrics()
    
    # 5. Backtest Loop
    obs, _ = env.reset()
    # Ensure current_step is consistent with pre-calculated signals
    # reset() might set current_step to 500
    start_step = env.current_step
    end_step = min(start_step + args.steps, env.max_steps)
    
    logger.info(f"Starting Backtest from step {start_step} to {end_step} ({end_step - start_step} steps)...")
    logger.info(f"Thresholds: Meta > {args.meta_thresh}, Quality > {args.qual_thresh}")

    for step_idx in tqdm(range(start_step, end_step), desc="Backtesting"):
        env.current_step = step_idx
        actions = {}
        
        # Look up pre-calculated signals for this step
        for i, asset in enumerate(env.assets):
            dir_probs = dir_probs_all[step_idx, i]
            meta_prob = meta_probs_all[step_idx, i]
            qual_score = qual_scores_all[step_idx, i]
            
            # Get predicted direction
            pred_idx = np.argmax(dir_probs)
            direction = pred_idx - 1 # Map {0,1,2} -> {-1,0,1}
            
            # Apply thresholds
            if meta_prob >= args.meta_thresh and qual_score >= args.qual_thresh:
                actions[asset] = {
                    'direction': direction,
                    'size': 0.2, 
                    'sl_mult': 1.5,
                    'tp_mult': 4.0
                }
            else:
                actions[asset] = {'direction': 0}

        # 1. Execute trades at current price (Close of candle T)
        env.completed_trades = []
        env._execute_trades(actions)
        
        # 2. Advance to next candle (T+1)
        env.current_step += 1
        if env.current_step >= env.max_steps:
            break
            
        # 3. Check SL/TP using prices of candle T+1
        env._update_positions()
        
        # Track metrics
        if env.completed_trades:
            for trade in env.completed_trades:
                metrics.add_trade(trade)
        
        metrics.add_equity_point(env._get_current_timestamp(), env.equity)
        
        # Robustness: Max Drawdown Exit (Save time if strategy is failing)
        if env.equity < env.start_equity * 0.5: # 50% Drawdown
            logger.warning(f"Critical Drawdown (>50%) at step {step_idx}. Stopping backtest.")
            break

        if env.equity <= 200: # Margin call / account blown
            logger.warning(f"Account blown at step {step_idx}!")
            break

    # 6. Final Results
    logger.info("\n" + "="*40)
    logger.info("SL BACKTEST COMPLETE")
    logger.info("="*40)
    
    if not metrics.equity_curve:
        logger.error("No equity data collected. Backtest failed.")
        return

    results = metrics.calculate_metrics()
    for k, v in results.items():
        if isinstance(v, float):
            logger.info(f"{k:<25}: {v:.4f}")
        else:
            logger.info(f"{k:<25}: {v}")
    
    # Generate Charts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "backtest" / "results"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        per_asset = metrics.get_per_asset_metrics()
        generate_all_charts(metrics, per_asset, "SL_Final", output_dir, timestamp)
        logger.info(f"Charts saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to generate charts: {e}")

if __name__ == "__main__":
    run_sl_backtest()
