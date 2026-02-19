import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from Alpha.src.model import AlphaSLModel
from RiskLayer.src.feature_engine import FeatureEngine
from Alpha.src.labeling import Labeler
from backtest.data_fetcher_backtest import DataFetcherBacktest, SYMBOL_IDS
import joblib
from RiskLayer.src.risk_model_sl import RiskModelSL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_2025_data(data_dir):
    """Checks for 2025 data and downloads it if missing."""
    missing = False
    for asset in SYMBOL_IDS.keys():
        file_path = data_dir / f"{asset}_5m_2025.parquet"
        if not file_path.exists():
            logger.info(f"Missing 2025 data for {asset}")
            missing = True
            break
    
    if missing:
        logger.info("Starting DataFetcherBacktest to download 2025 data...")
        fetcher = DataFetcherBacktest()
        fetcher.start() # Note: This will block until complete
    else:
        logger.info("All 2025 backtest data found.")

def load_data(data_dir):
    """Loads 2025 data for all assets."""
    data_dict = {}
    for asset in SYMBOL_IDS.keys():
        file_path = data_dir / f"{asset}_5m_2025.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            data_dict[asset] = df
        else:
            logger.warning(f"Could not load data for {asset}")
    return data_dict

def optimize_thresholds():
    parser = argparse.ArgumentParser(description="Alpha & Risk Model Threshold Optimizer")
    parser.add_argument("--alpha-model-path", type=str, default="Alpha/models/alpha_model.pth", help="Path to trained Alpha model")
    parser.add_argument("--risk-model-path", type=str, default="RiskLayer/models/risk_model_sl_final.pth", help="Path to trained Risk model")
    parser.add_argument("--scaler-path", type=str, default="RiskLayer/models/sl_risk_scaler.pkl", help="Path to risk feature scaler")
    parser.add_argument("--data-dir", type=str, default="backtest/data", help="Directory for backtest data")
    args = parser.parse_args()

    alpha_model_path = PROJECT_ROOT / args.alpha_model_path
    risk_model_path = PROJECT_ROOT / args.risk_model_path
    scaler_path = PROJECT_ROOT / args.scaler_path
    data_dir = PROJECT_ROOT / args.data_dir
    
    # 1. Ensure Data
    ensure_2025_data(data_dir)
    
    # 2. Load Data and Model
    logger.info("Loading 2025 data...")
    data_dict = load_data(data_dir)
    if not data_dict:
        logger.error("No data loaded. Exiting.")
        return

    logger.info(f"Loading Alpha model from {alpha_model_path}...")
    alpha_model = AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4).to(DEVICE)
    alpha_model.load_state_dict(torch.load(alpha_model_path, map_location=DEVICE))
    alpha_model.eval()

    logger.info(f"Loading Risk model from {risk_model_path}...")
    risk_model = RiskModelSL(input_dim=48, hidden_dim=256, num_res_blocks=4).to(DEVICE)
    risk_model.load_state_dict(torch.load(risk_model_path, map_location=DEVICE))
    risk_model.eval()

    logger.info(f"Loading Risk scaler from {scaler_path}...")
    risk_scaler = joblib.load(scaler_path)

    # 3. Preprocess and Generate Predictions
    engine = FeatureEngine()
    labeler = Labeler(stride=5) 
    
    # We'll collect result tensors on GPU
    all_pred_dirs = []
    all_confidences = []
    all_meta_probs = []
    all_qual_preds = []
    all_risk_evs = []
    all_risk_sizes = []
    all_actual_dirs = []
    
    logger.info("Preprocessing data for all assets...")
    aligned_df, normalized_df = engine.preprocess_data(data_dict)
    
    for asset in data_dict.keys():
        logger.info(f"Generating predictions for {asset}...")
        labels_df = labeler.label_data(aligned_df, asset)
        common_indices = labels_df.index.intersection(normalized_df.index)
        filtered_norm_df = normalized_df.loc[common_indices]
        filtered_labels_df = labels_df.loc[common_indices]
        
        # 1. Get raw features
        X_raw = engine.get_observation_vectorized(filtered_norm_df, asset)
        
        # 2. Prepare Alpha Inputs (already normalized by RiskFeatureEngine for first 40 features)
        X_alpha = torch.from_numpy(X_raw).to(DEVICE).float()
        
        actual_dirs = torch.from_numpy(filtered_labels_df['direction'].values).to(DEVICE).long()
        
        batch_size = 8192
        with torch.no_grad():
            for i in range(0, len(X_alpha), batch_size):
                # 1. Alpha Batch - Only take first 40 features
                batch_X_alpha = X_alpha[i:i+batch_size, :40]
                dir_logits, qual_pred, meta_logits = alpha_model(batch_X_alpha)
                
                dir_probs = torch.softmax(dir_logits, dim=1)
                meta_probs = torch.sigmoid(meta_logits).squeeze()
                # quality in labeling is [0,1], generate_sl_dataset uses sigmoid on raw output
                qual_preds_batch = torch.sigmoid(qual_pred).squeeze()
                
                conf, pred_dir = torch.max(dir_probs, dim=1)
                pred_dir = pred_dir - 1 # Map {0,1,2} -> {-1,0,1}
                
                # 2. Risk Batch - Inject Alpha outputs into raw features before scaling
                X_raw_batch = X_raw[i:i+batch_size].copy()
                
                # Indices 40, 41, 42 correspond to alpha_direction, alpha_meta, alpha_quality
                # weighted_direction = P(Long) - P(Short)
                weighted_dir = (dir_probs[:, 2] - dir_probs[:, 0]).cpu().numpy()
                X_raw_batch[:, 40] = weighted_dir
                X_raw_batch[:, 41] = meta_probs.cpu().numpy()
                X_raw_batch[:, 42] = qual_preds_batch.cpu().numpy()
                
                # Scale the complete 48-feature vector
                X_risk_batch_scaled = risk_scaler.transform(X_raw_batch)
                X_risk_batch = torch.from_numpy(X_risk_batch_scaled).to(DEVICE).float()
                
                risk_outputs = risk_model(X_risk_batch)
                risk_ev = risk_outputs['expected_value'].squeeze()
                risk_size = risk_outputs['size'].squeeze()
                
                all_pred_dirs.append(pred_dir)
                all_confidences.append(conf)
                all_meta_probs.append(meta_probs)
                all_qual_preds.append(qual_preds_batch)
                all_risk_evs.append(risk_ev)
                all_risk_sizes.append(risk_size)
                all_actual_dirs.append(actual_dirs[i:i+batch_size])
        
    # Concatenate all results into single GPU tensors
    pred_dirs = torch.cat(all_pred_dirs)
    meta_probs = torch.cat(all_meta_probs)
    qual_preds = torch.cat(all_qual_preds)
    risk_evs = torch.cat(all_risk_evs)
    risk_sizes = torch.cat(all_risk_sizes)
    actual_dirs = torch.cat(all_actual_dirs)
    
    # Debug distributions
    logger.info(f"Alpha Meta Prob Dist: Mean={meta_probs.mean():.3f}, Min={meta_probs.min():.3f}, Max={meta_probs.max():.3f}")
    logger.info(f"Alpha Quality Dist:  Mean={qual_preds.mean():.3f}, Min={qual_preds.min():.3f}, Max={qual_preds.max():.3f}")
    logger.info(f"Risk EV Dist:        Mean={risk_evs.mean():.3f}, Min={risk_evs.min():.3f}, Max={risk_evs.max():.3f}")

    # 4. GPU-Accelerated Grid Search
    logger.info(f"Starting GPU-Accelerated Grid Search on {len(pred_dirs)} predictions...")
    
    meta_thresholds = torch.linspace(0.5, 0.95, 15).to(DEVICE)
    qual_thresholds = torch.linspace(0.3, 0.8, 10).to(DEVICE)
    risk_ev_thresholds = torch.linspace(0.0, 1.0, 10).to(DEVICE) # EVs are often low, let's try a range
    
    best_expectancy = -np.inf
    best_params = None
    all_results = []

    # Vectorized loop for thresholds
    for m_t in tqdm(meta_thresholds, desc="Optimizing"):
        for q_t in qual_thresholds:
            for r_t in risk_ev_thresholds:
                # Mask for trades that meet thresholds
                # meta_probs: Alpha confidence
                # qual_preds: Alpha quality score
                # risk_evs: Risk model expected value prediction
                mask = (meta_probs >= m_t) & (qual_preds >= q_t) & (risk_evs >= r_t) & (pred_dirs != 0)
                
                num_trades = mask.sum().item()
                if num_trades < 25: # Lowered threshold slightly for more combinations
                    continue
                    
                # Filtered actual and predicted
                t_actual = actual_dirs[mask]
                t_pred = pred_dirs[mask]
                
                # Calculate R-Returns on GPU
                is_win = (t_pred == t_actual)
                is_time_exit = (t_actual == 0)
                is_loss = (~is_win) & (~is_time_exit)
                
                r_returns = torch.zeros_like(t_actual, dtype=torch.float32)
                r_returns[is_win] = 4.0
                r_returns[is_time_exit] = -0.5
                r_returns[is_loss] = -1.5
                
                avg_r = r_returns.mean().item()
                win_rate = is_win.float().mean().item()
                
                # Max Loss Streak calculation (on CPU)
                is_loss_np = (r_returns < 0).cpu().numpy().astype(int)
                max_streak = 0
                current_streak = 0
                for val in is_loss_np:
                    if val == 1:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 0
                
                res = {
                    'meta_thresh': m_t.item(),
                    'qual_thresh': q_t.item(),
                    'risk_ev_thresh': r_t.item(),
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'avg_r': avg_r,
                    'max_loss_streak': max_streak,
                    'expectancy': avg_r
                }
                all_results.append(res)
                
                if avg_r > best_expectancy:
                    best_expectancy = avg_r
                    best_params = res

    # 5. Output Results
    if best_params:
        logger.info("\n" + "="*40)
        logger.info("OPTIMAL ALPHA & RISK THRESHOLDS FOUND")
        logger.info("="*40)
        logger.info(f"Alpha Meta Threshold:    {best_params['meta_thresh']:.2f}")
        logger.info(f"Alpha Quality Threshold: {best_params['qual_thresh']:.2f}")
        logger.info(f"Risk EV Threshold:       {best_params['risk_ev_thresh']:.2f}")
        logger.info(f"Expected Win Rate: {best_params['win_rate']:.1%}")
        logger.info(f"Expected Avg R:    {best_params['avg_r']:.3f}")
        logger.info(f"Total Trades (2025): {best_params['num_trades']}")
        logger.info("="*40)
        
        output_file = PROJECT_ROOT / "backtest" / "results" / "threshold_optimization.csv"
        os.makedirs(output_file.parent, exist_ok=True)
        pd.DataFrame(all_results).sort_values('expectancy', ascending=False).to_csv(output_file, index=False)
        logger.info(f"Full sweep results saved to {output_file}")
    else:
        logger.warning("No valid threshold combinations found.")

if __name__ == "__main__":
    optimize_thresholds()
