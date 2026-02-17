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
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.labeling import LabelingEngine
from backtest.data_fetcher_backtest import DataFetcherBacktest, SYMBOL_IDS

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
    parser = argparse.ArgumentParser(description="Alpha Model Threshold Optimizer")
    parser.add_argument("--model-path", type=str, default="Alpha/models/alpha_model.pth", help="Path to trained model")
    parser.add_argument("--data-dir", type=str, default="backtest/data", help="Directory for backtest data")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model_path
    data_dir = PROJECT_ROOT / args.data_dir
    
    # 1. Ensure Data
    ensure_2025_data(data_dir)
    
    # 2. Load Data and Model
    logger.info("Loading 2025 data...")
    data_dict = load_data(data_dir)
    if not data_dict:
        logger.error("No data loaded. Exiting.")
        return

    logger.info(f"Loading model from {model_path}...")
    model = AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 3. Preprocess and Generate Predictions
    engine = FeatureEngine()
    labeler = LabelingEngine(stride=5)
    
    # We'll collect result tensors on GPU
    all_pred_dirs = []
    all_confidences = []
    all_meta_probs = []
    all_qual_preds = []
    all_actual_dirs = []
    
    logger.info("Preprocessing data for all assets...")
    aligned_df, normalized_df = engine.preprocess_data(data_dict)
    
    for asset in data_dict.keys():
        logger.info(f"Generating predictions for {asset}...")
        labels_df = labeler.label_data(aligned_df, asset)
        common_indices = labels_df.index.intersection(normalized_df.index)
        filtered_norm_df = normalized_df.loc[common_indices]
        filtered_labels_df = labels_df.loc[common_indices]
        
        X = engine.get_observation_vectorized(filtered_norm_df, asset)
        X_tensor = torch.from_numpy(X).to(DEVICE)
        
        actual_dirs = torch.from_numpy(filtered_labels_df['direction'].values).to(DEVICE).long()
        
        batch_size = 8192
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                dir_logits, qual_pred, meta_logits = model(batch_X)
                
                dir_probs = torch.softmax(dir_logits, dim=1)
                meta_probs = torch.sigmoid(meta_logits)
                
                conf, pred_dir = torch.max(dir_probs, dim=1)
                pred_dir = pred_dir - 1 # Map {0,1,2} -> {-1,0,1}
                
                all_pred_dirs.append(pred_dir)
                all_confidences.append(conf)
                all_meta_probs.append(meta_probs.squeeze())
                all_qual_preds.append(qual_pred.squeeze())
                all_actual_dirs.append(actual_dirs[i:i+batch_size])
        
    # Concatenate all results into single GPU tensors
    pred_dirs = torch.cat(all_pred_dirs)
    meta_probs = torch.cat(all_meta_probs)
    qual_preds = torch.cat(all_qual_preds)
    actual_dirs = torch.cat(all_actual_dirs)
    
    # 4. GPU-Accelerated Grid Search
    logger.info(f"Starting GPU-Accelerated Grid Search on {len(pred_dirs)} predictions...")
    
    meta_thresholds = torch.linspace(0.5, 0.95, 20).to(DEVICE)
    qual_thresholds = torch.linspace(0.3, 0.8, 15).to(DEVICE)
    
    best_expectancy = -np.inf
    best_params = None
    all_results = []

    # Vectorized loop for thresholds
    for m_t in tqdm(meta_thresholds, desc="Optimizing"):
        for q_t in qual_thresholds:
            # Mask for trades that meet thresholds
            mask = (meta_probs >= m_t) & (qual_preds >= q_t) & (pred_dirs != 0)
            
            num_trades = mask.sum().item()
            if num_trades < 50:
                continue
                
            # Filtered actual and predicted
            t_actual = actual_dirs[mask]
            t_pred = pred_dirs[mask]
            
            # Calculate R-Returns on GPU
            # Win: pred == actual -> +4.0
            # Time Exit: actual == 0 -> -0.5
            # Loss: pred != actual AND actual != 0 -> -1.5
            
            is_win = (t_pred == t_actual)
            is_time_exit = (t_actual == 0)
            is_loss = (~is_win) & (~is_time_exit)
            
            r_returns = torch.zeros_like(t_actual, dtype=torch.float32)
            r_returns[is_win] = 4.0
            r_returns[is_time_exit] = -0.5
            r_returns[is_loss] = -1.5
            
            avg_r = r_returns.mean().item()
            win_rate = is_win.float().mean().item()
            
            # Max Loss Streak calculation (on CPU for streak logic)
            # A loss is any trade where r_return < 0
            is_loss_np = (r_returns < 0).cpu().numpy().astype(int)
            if len(is_loss_np) > 0:
                # Group consecutive values and count
                streak_groups = (is_loss_np != np.concatenate([[0], is_loss_np[:-1]])).cumsum()
                loss_streaks = np.where(is_loss_np == 1, 1, 0)
                # Sum within groups of 1s
                max_streak = 0
                current_streak = 0
                for val in is_loss_np:
                    if val == 1:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 0
            else:
                max_streak = 0
            
            res = {
                'meta_thresh': m_t.item(),
                'qual_thresh': q_t.item(),
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
        logger.info("OPTIMAL THRESHOLDS FOUND (GPU RUN)")
        logger.info("="*40)
        logger.info(f"Meta Threshold:    {best_params['meta_thresh']:.2f}")
        logger.info(f"Quality Threshold: {best_params['qual_thresh']:.2f}")
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
