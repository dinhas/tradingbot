import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
import argparse
import joblib
import json
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Add RiskLayer/src to path for custom policy loading
_risklayer_src = PROJECT_ROOT / "Risklayer" / "src"
if str(_risklayer_src) not in sys.path:
    sys.path.insert(0, str(_risklayer_src))

from Alpha.src.model import AlphaSLModel
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.labeling import Labeler
from backtest.data_fetcher_backtest import DataFetcherBacktest, SYMBOL_IDS

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_backtest_data(data_dir):
    """Checks for backtest data and downloads it if missing."""
    missing = False
    for asset in SYMBOL_IDS.keys():
        file_path = data_dir / f"{asset}_5m_backtest.parquet"
        if not file_path.exists():
            logger.info(f"Missing backtest data for {asset}")
            missing = True
            break

    if missing:
        logger.info("Starting DataFetcherBacktest to download data...")
        fetcher = DataFetcherBacktest()
        fetcher.start()  # Note: This will block until complete
    else:
        logger.info("All backtest data found.")


def load_data(data_dir):
    """Loads backtest data for all assets."""
    data_dict = {}
    for asset in SYMBOL_IDS.keys():
        file_path = data_dir / f"{asset}_5m_backtest.parquet"
        if not file_path.exists():
            # Fallback to 2025 legacy
            file_path = data_dir / f"{asset}_5m_2025.parquet"
            
        if file_path.exists():
            df = pd.read_parquet(file_path)
            # Ensure index is datetime
            df.index = pd.to_datetime(df.index)
            data_dict[asset] = df
            logger.info(f"Loaded {asset} from {file_path.name}")
        else:
            logger.warning(f"Could not load data for {asset}")
    return data_dict


def optimize_thresholds():
    parser = argparse.ArgumentParser(
        description="Alpha-Risk Multi-Model Threshold Optimizer"
    )
    parser.add_argument(
        "--alpha-model",
        type=str,
        default="Alpha/models/alpha_model.pth",
        help="Path to Alpha model",
    )
    parser.add_argument(
        "--risk-model",
        type=str,
        default="Risklayer/models/ppo_risk_model_final_v2_opt.zip",
        help="Path to RL Risk model (PPO)",
    )
    parser.add_argument(
        "--risk-scaler",
        type=str,
        default="Risklayer/models/rl_risk_scaler.pkl",
        help="Path to Risk scaler",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="backtest/data",
        help="Directory for backtest data",
    )
    args = parser.parse_args()

    alpha_path = PROJECT_ROOT / args.alpha_model
    risk_path = PROJECT_ROOT / args.risk_model
    scaler_path = PROJECT_ROOT / args.risk_scaler
    data_dir = PROJECT_ROOT / args.data_dir

    # 1. Ensure Data
    ensure_backtest_data(data_dir)

    # 2. Load Data and Models
    logger.info("Loading backtest data...")
    data_dict = load_data(data_dir)
    if not data_dict:
        logger.error("No data loaded. Exiting.")
        return

    logger.info(f"Loading Alpha model from {alpha_path}...")
    # Using parameters from Alpha/src/model.py
    alpha_model = AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4).to(
        DEVICE
    )
    alpha_model.load_state_dict(torch.load(alpha_path, map_location=DEVICE))
    alpha_model.eval()

    # Load RL Risk Model
    logger.info(f"Loading RL Risk model from {risk_path}...")
    if risk_path.exists() and risk_path.suffix == ".zip":
        risk_model = PPO.load(str(risk_path), device=DEVICE)
    else:
        logger.error(f"RL Risk model not found or invalid format: {risk_path}")
        return

    logger.info(f"Loading Risk Scaler from {scaler_path}...")
    risk_scaler = joblib.load(scaler_path)

    # 3. Preprocess and Generate Predictions
    engine = FeatureEngine()
    labeler = Labeler(stride=5)

    all_pred_dirs = []
    all_meta_probs = []
    all_qual_preds = []
    all_risk_sizes = []
    all_actual_dirs = []

    logger.info("Preprocessing data for all assets...")
    aligned_df, normalized_df = engine.preprocess_data(data_dict)

    for asset in data_dict.keys():
        logger.info(f"Generating signals for {asset}...")
        labels_df = labeler.label_data(aligned_df, asset)
        common_indices = labels_df.index.intersection(normalized_df.index)
        filtered_norm_df = normalized_df.loc[common_indices]
        filtered_labels_df = labels_df.loc[common_indices]

        # Get raw features for Alpha (using the vectorized engine)
        X = engine.get_observation_vectorized(filtered_norm_df, asset)

        # Risk Scaling
        X_risk = risk_scaler.transform(X).astype(np.float32)

        X_tensor = torch.from_numpy(X).to(DEVICE)
        X_risk_tensor = torch.from_numpy(X_risk).to(DEVICE)

        actual_dirs = (
            torch.from_numpy(filtered_labels_df["direction"].values).to(DEVICE).long()
        )

        batch_size = 8192
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                b_X = X_tensor[i : i + batch_size]
                b_X_risk = X_risk_tensor[i : i + batch_size]

                # Alpha Forward
                dir_logits, qual_pred, meta_logits = alpha_model(b_X)

                # RL Risk Prediction
                # PPO model predict returns (action, next_state)
                # We need to run it through the policy to get raw actions for sizing
                # or just use the sizing action from predict if we want deterministic
                actions, _ = risk_model.predict(b_X_risk.cpu().numpy(), deterministic=True)
                actions = torch.from_numpy(actions).to(DEVICE)
                
                # size_raw is index 2 in our action space [-1, 1]
                size_raw = actions[:, 2]
                # Convert from [-1, 1] to actual values [0.1, 0.3] matching risk_ppo_env
                risk_size = 0.1 + (size_raw + 1) / 2 * (0.3 - 0.1)

                dir_probs = torch.softmax(dir_logits, dim=1)
                meta_probs = torch.sigmoid(meta_logits)

                _, pred_dir = torch.max(dir_probs, dim=1)
                pred_dir = pred_dir - 1  # Map {0,1,2} -> {-1,0,1}

                all_pred_dirs.append(pred_dir)
                all_meta_probs.append(meta_probs.squeeze())
                all_qual_preds.append(qual_pred.squeeze())
                all_risk_sizes.append(risk_size.squeeze())
                all_actual_dirs.append(actual_dirs[i : i + batch_size])

    # Concatenate all results
    pred_dirs = torch.cat(all_pred_dirs)
    meta_probs = torch.cat(all_meta_probs)
    qual_preds = torch.cat(all_qual_preds)
    risk_sizes = torch.cat(all_risk_sizes)
    actual_dirs = torch.cat(all_actual_dirs)

    # 4. 3D GPU-Accelerated Grid Search
    logger.info(f"Starting 3D Grid Search on {len(pred_dirs)} predictions...")

    # Grid search ranges
    meta_thresholds = torch.linspace(0.7, 0.98, 12).to(DEVICE)
    qual_thresholds = torch.linspace(0.2, 0.6, 10).to(DEVICE)
    # Risk size thresholds (since sizes are 0.1 to 0.3, we look for 'confident' sizes)
    risk_thresholds = torch.linspace(0.1, 0.25, 8).to(DEVICE)

    best_expectancy = -np.inf
    best_params = None

    # Vectorized loop for thresholds
    for m_t in tqdm(meta_thresholds, desc="Optimizing"):
        for q_t in qual_thresholds:
            for r_t in risk_thresholds:
                # Mask: Alpha Meta AND Alpha Quality AND Risk Size
                mask = (
                    (meta_probs >= m_t)
                    & (qual_preds >= q_t)
                    & (risk_sizes >= r_t)
                    & (pred_dirs != 0)
                )

                num_trades = mask.sum().item()
                if num_trades < 100: # Increased min trades for statistical significance
                    continue

                t_actual = actual_dirs[mask]
                t_pred = pred_dirs[mask]

                # Calculate R-Returns
                is_win = t_pred == t_actual
                is_time_exit = t_actual == 0
                is_loss = (~is_win) & (~is_time_exit)

                r_returns = torch.zeros_like(t_actual, dtype=torch.float32)
                r_returns[is_win] = 3.5  # Conservative avg win R
                r_returns[is_time_exit] = -0.3 # Average timeout cost
                r_returns[is_loss] = -1.5 # Average loss cost

                avg_r = r_returns.mean().item()

                if avg_r > best_expectancy:
                    best_expectancy = avg_r
                    best_params = {
                        "meta_threshold": round(m_t.item(), 4),
                        "qual_threshold": round(q_t.item(), 4),
                        "risk_threshold": round(r_t.item(), 4),
                        "expected_win_rate": round(is_win.float().mean().item(), 4),
                        "expected_avg_r": round(avg_r, 4),
                        "num_trades_dataset": num_trades,
                        "optimized_at": datetime.now().isoformat(),
                    }

    # 5. Save Optimal Params to JSON
    if best_params:
        logger.info("\n" + "=" * 40)
        logger.info("OPTIMAL ALPHA-RISK THRESHOLDS")
        logger.info("=" * 40)
        logger.info(f"Alpha Meta Thresh:  {best_params['meta_threshold']:.4f}")
        logger.info(f"Alpha Qual Thresh:  {best_params['qual_threshold']:.4f}")
        logger.info(f"Risk Size Thresh:   {best_params['risk_threshold']:.4f}")
        logger.info(f"Expected Avg R:     {best_params['expected_avg_r']:.3f}")
        logger.info(f"Total Trades:       {best_params['num_trades_dataset']}")
        logger.info("=" * 40)

        output_file = PROJECT_ROOT / "backtest" / "results" / "optimal_thresholds.json"
        os.makedirs(output_file.parent, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(best_params, f, indent=4)

        logger.info(f"Optimal parameters saved to {output_file}")
    else:
        logger.warning("No valid threshold combinations found.")


if __name__ == "__main__":
    optimize_thresholds()
