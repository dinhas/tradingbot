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
from RiskLayer.src.risk_model_sl import RiskModelSL
from RiskLayer.src.feature_engine import FeatureEngine as RiskFeatureEngine
from backtest.sl_backtest_utils import BacktestMetrics, generate_all_charts
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_sl_backtest():
    parser = argparse.ArgumentParser(description="Alpha Supervised Model Backtester (Optimized)")
    parser.add_argument("--model-path", type=str, default="Alpha/models/alpha_model.pth")
    parser.add_argument("--risk-model-path", type=str, default="RiskLayer/models/risk_model_sl_best.pth")
    parser.add_argument("--risk-scaler-path", type=str, default="RiskLayer/models/sl_risk_scaler.pkl")
    parser.add_argument("--use-risk-model", action="store_true", default=True)
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
    
    # 2. Load Alpha Model
    logger.info(f"Loading Alpha SL Model from {model_path} to {DEVICE}...")
    model = AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load alpha model state: {e}")
        return

    # 2.1 Load Risk Model if enabled
    risk_model = None
    risk_scaler = None
    if args.use_risk_model:
        risk_model_path = PROJECT_ROOT / args.risk_model_path
        risk_scaler_path = PROJECT_ROOT / args.risk_scaler_path

        logger.info(f"Loading Risk SL Model from {risk_model_path}...")
        risk_model = RiskModelSL(input_dim=48, hidden_dim=256, num_res_blocks=4).to(DEVICE)
        try:
            risk_model.load_state_dict(torch.load(risk_model_path, map_location=DEVICE))
            risk_model.eval()
            logger.info("Loading Risk Scaler...")
            risk_scaler = joblib.load(risk_scaler_path)
        except Exception as e:
            logger.error(f"Failed to load risk components: {e}")
            args.use_risk_model = False

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

    # 3.1 Pre-calculate Risk Signals
    sl_mults_all = None
    tp_mults_all = None
    sizes_all = None
    probs_tp_all = None

    if args.use_risk_model and risk_model is not None:
        logger.info("Calculating Risk Features with simulated spreads...")
        risk_fe = RiskFeatureEngine()
        risk_raw_df, _ = risk_fe.preprocess_data(env.data)

        # Inject Alpha signals into Risk features
        dir_all = np.argmax(dir_probs_all, axis=2) - 1 # (N, num_assets)
        for i, asset in enumerate(env.assets):
            risk_raw_df[f"{asset}_alpha_direction"] = dir_all[:, i]
            risk_raw_df[f"{asset}_alpha_meta"] = meta_probs_all[:, i]
            risk_raw_df[f"{asset}_alpha_quality"] = qual_scores_all[:, i]

        # Re-normalize Risk features with injected Alpha signals
        risk_norm_df = risk_fe._normalize_features(risk_raw_df.copy())

        # Vectorized Risk Inference
        risk_obs_all_assets = []
        for asset in env.assets:
            obs = risk_fe.get_observation_vectorized(risk_norm_df, asset)
            risk_obs_all_assets.append(obs)

        risk_obs_flat = np.concatenate(risk_obs_all_assets, axis=0) # (num_assets * N, 48)
        risk_obs_scaled = risk_scaler.transform(risk_obs_flat).astype(np.float32)

        sl_mults_list = []
        tp_mults_list = []
        sizes_list = []
        probs_tp_list = []

        with torch.no_grad():
            for i in tqdm(range(0, len(risk_obs_scaled), args.batch_size), desc="Risk Inference"):
                batch = torch.from_numpy(risk_obs_scaled[i : i + args.batch_size]).to(DEVICE)
                preds = risk_model(batch)

                sl_mults_list.append(preds['sl_mult'].cpu().numpy())
                tp_mults_list.append(preds['tp_mult'].cpu().numpy())
                sizes_list.append(preds['size'].cpu().numpy())
                # Using prob_head (which is prob_tp_first_logits in RiskModelSL)
                probs_tp_list.append(torch.sigmoid(preds['prob_tp_first_logits']).cpu().numpy())

        sl_mults_all = np.concatenate(sl_mults_list, axis=0).reshape(num_assets, N).T
        tp_mults_all = np.concatenate(tp_mults_list, axis=0).reshape(num_assets, N).T
        sizes_all = np.concatenate(sizes_list, axis=0).reshape(num_assets, N).T
        probs_tp_all = np.concatenate(probs_tp_list, axis=0).reshape(num_assets, N).T
    
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
                # Use Risk Model outputs if available
                if sl_mults_all is not None:
                    sl_mult = float(sl_mults_all[step_idx, i])
                    tp_mult = float(tp_mults_all[step_idx, i])
                    size = float(sizes_all[step_idx, i])
                    prob_tp = float(probs_tp_all[step_idx, i])

                    # Risk Model Filter: Only trade if TP probability > 0.5
                    if prob_tp >= 0.5:
                        actions[asset] = {
                            'direction': direction,
                            'size': size,
                            'sl_mult': sl_mult,
                            'tp_mult': tp_mult
                        }
                    else:
                        actions[asset] = {'direction': 0}
                else:
                    # Fallback to hardcoded values
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
