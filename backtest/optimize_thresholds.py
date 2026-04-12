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
SEQ_LEN = 50


def _build_sequence_windows(features_2d: np.ndarray, seq_len: int = SEQ_LEN):
    """
    Build sequence windows for LSTM model.
    Returns:
      X_seq: (N-seq_len, seq_len, 40)
      valid_idx: indices in original array corresponding to sequence targets
    """
    n = len(features_2d)
    if n <= seq_len:
        return np.empty((0, seq_len, features_2d.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int64)

    valid_idx = np.arange(seq_len, n, dtype=np.int64)
    X_seq = np.stack([features_2d[i - seq_len : i] for i in valid_idx], axis=0).astype(np.float32)
    return X_seq, valid_idx


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


def optimize_thresholds_main(alpha_model, risk_model, risk_scaler, data_dir, alpha_only=False):
    alpha_path = PROJECT_ROOT / alpha_model
    risk_path = PROJECT_ROOT / risk_model
    scaler_path = PROJECT_ROOT / risk_scaler
    data_dir = PROJECT_ROOT / data_dir

    # 1. Ensure Data
    ensure_backtest_data(data_dir)

    # 2. Load Data and Models
    logger.info("Loading backtest data...")
    data_dict = load_data(data_dir)
    if not data_dict:
        logger.error("No data loaded. Exiting.")
        return

    logger.info(f"Loading Alpha model from {alpha_path}...")
    alpha_model = AlphaSLModel(input_dim=40, hidden_dim=256, num_layers=2).to(DEVICE)
    alpha_model.load_state_dict(torch.load(alpha_path, map_location=DEVICE))
    alpha_model.eval()

    # Load RL Risk Model if not alpha_only
    risk_m = None
    r_scaler = None
    if not alpha_only:
        logger.info(f"Loading RL Risk model from {risk_path}...")
        if risk_path.exists() and risk_path.suffix == ".zip":
            risk_m = PPO.load(str(risk_path), device=DEVICE)
        else:
            logger.error(f"RL Risk model not found or invalid format: {risk_path}")
            return

        logger.info(f"Loading Risk Scaler from {scaler_path}...")
        r_scaler = joblib.load(scaler_path)

    # 3. Preprocess and Generate Predictions
    engine = FeatureEngine()
    labeler = Labeler()

    all_pred_dirs = []
    all_meta_probs = []
    all_qual_preds = []
    all_risk_sizes = []
    all_actual_dirs = []

    logger.info("Preprocessing data for all assets...")
    aligned_df, normalized_df = engine.preprocess_data(data_dict)

    for asset in data_dict.keys():
        logger.info(f"Generating signals for {asset}...")
        # Get raw per-step features from the DENSE normalized_df
        X_dense = engine.get_observation_vectorized(normalized_df, asset).astype(np.float32)

        labels_df = labeler.label_data(aligned_df, asset)
        common_indices = labels_df.index.intersection(normalized_df.index)

        # Get integer position of each label in the dense matrix
        label_full_idx = normalized_df.index.get_indexer(common_indices)

        # Filter for valid lookback
        valid_mask = (label_full_idx >= SEQ_LEN)
        target_indices = label_full_idx[valid_mask]

        if len(target_indices) == 0:
            logger.warning(f"Skipping {asset}: not enough rows with SEQ_LEN={SEQ_LEN} lookback")
            continue

        # Build windows from DENSE features
        X_seq = np.stack([X_dense[i - SEQ_LEN : i] for i in target_indices], axis=0).astype(np.float32)

        X_tensor = torch.from_numpy(X_seq).to(DEVICE)
        
        # Risk model features if applicable
        X_risk_tensor = None
        if not alpha_only:
            X_risk = r_scaler.transform(X_dense[target_indices]).astype(np.float32)
            X_risk_tensor = torch.from_numpy(X_risk).to(DEVICE)

        # Align actual labels
        filtered_labels_df = labels_df.loc[common_indices].iloc[valid_mask]
        actual_dirs = torch.from_numpy(filtered_labels_df["direction"].values).to(DEVICE).long()

        batch_size = 2048 # Reduced from 8192 for memory stability
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                b_X = X_tensor[i : i + batch_size]
                
                # Alpha Forward
                dir_logits, qual_pred, meta_logits = alpha_model(b_X)

                # RL Risk Prediction if applicable
                if not alpha_only:
                    b_X_risk = X_risk_tensor[i : i + batch_size]
                    actions, _ = risk_m.predict(b_X_risk.cpu().numpy(), deterministic=True)
                    actions = torch.from_numpy(actions).to(DEVICE)
                    # size_raw is index 2 in our action space [-1, 1]
                    size_raw = actions[:, 2]
                    # Convert from [-1, 1] to actual values [0.1, 0.3] matching risk_ppo_env
                    risk_size = 0.1 + (size_raw + 1) / 2 * (0.3 - 0.1)
                    all_risk_sizes.append(risk_size.squeeze().cpu())
                else:
                    # Dummy risk size to keep tensors aligned
                    all_risk_sizes.append(torch.ones(len(b_X)))

                dir_probs = torch.softmax(dir_logits, dim=1)
                meta_probs = torch.sigmoid(meta_logits)

                _, pred_dir = torch.max(dir_probs, dim=1)
                pred_dir = pred_dir - 1  # Map {0,1,2} -> {-1,0,1}

                all_pred_dirs.append(pred_dir.cpu())
                all_meta_probs.append(meta_probs.squeeze().cpu())
                all_qual_preds.append(qual_pred.squeeze().cpu())
                all_actual_dirs.append(actual_dirs[i : i + batch_size].cpu())
        
        # Clean up per-asset tensors
        del X_tensor, X_risk_tensor, actual_dirs
        torch.cuda.empty_cache()

    # Concatenate all results
    pred_dirs = torch.cat(all_pred_dirs)
    meta_probs = torch.cat(all_meta_probs)
    qual_preds = torch.cat(all_qual_preds)
    risk_sizes = torch.cat(all_risk_sizes)
    actual_dirs = torch.cat(all_actual_dirs)

    # --- DEBUG: Log and Save Prediction Distributions ---
    logger.info("\n" + "=" * 40)
    logger.info("PREDICTION DISTRIBUTION DEBUG")
    logger.info("=" * 40)
    logger.info(f"Total Prediction Samples: {len(pred_dirs)}")
    
    # Meta Debug
    m_min, m_max = meta_probs.min().item(), meta_probs.max().item()
    m_mean, m_std = meta_probs.mean().item(), meta_probs.std().item()
    logger.info(f"Meta Probs  | Min: {m_min:.4f} | Max: {m_max:.4f} | Mean: {m_mean:.4f} | Std: {m_std:.4f}")
    
    # Quality Debug
    q_min, q_max = qual_preds.min().item(), qual_preds.max().item()
    q_mean, q_std = qual_preds.mean().item(), qual_preds.std().item()
    logger.info(f"Qual Preds  | Min: {q_min:.4f} | Max: {q_max:.4f} | Mean: {q_mean:.4f} | Std: {q_std:.4f}")
    
    # Risk Debug
    if not alpha_only:
        r_min, r_max = risk_sizes.min().item(), risk_sizes.max().item()
        r_mean, r_std = risk_sizes.mean().item(), risk_sizes.std().item()
        logger.info(f"Risk Sizes  | Min: {r_min:.4f} | Max: {r_max:.4f} | Mean: {r_mean:.4f} | Std: {r_std:.4f}")
    
    # Direction Distribution
    dir_counts = torch.bincount((pred_dirs + 1).long(), minlength=3)
    logger.info(f"Dir Preds   | Short: {dir_counts[0]} | Flat: {dir_counts[1]} | Long: {dir_counts[2]}")
    logger.info("=" * 40 + "\n")

    # Save raw outputs for external analysis
    debug_dir = PROJECT_ROOT / "backtest" / "results" / "debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_file = debug_dir / f"model_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(
        debug_file,
        pred_dirs=pred_dirs.numpy(),
        meta_probs=meta_probs.numpy(),
        qual_preds=qual_preds.numpy(),
        risk_sizes=risk_sizes.numpy(),
        actual_dirs=actual_dirs.numpy()
    )
    logger.info(f"Raw model outputs saved to {debug_file}")
    # ---------------------------------------------------

    # 4. GPU-Accelerated Grid Search
    logger.info(f"Starting Grid Search on {len(pred_dirs)} predictions...")

    # Grid search ranges (Updated to match actual model output distributions)
    meta_thresholds = torch.linspace(0.8, 0.99, 20).to(DEVICE)
    qual_thresholds = torch.linspace(0.6, 0.98, 20).to(DEVICE)
    # Risk size thresholds
    risk_thresholds = torch.linspace(0.1, 0.28, 10).to(DEVICE) if not alpha_only else torch.tensor([0.0]).to(DEVICE)

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
                if num_trades < 100: # Min trades for statistical significance
                    continue

                t_actual = actual_dirs[mask]
                t_pred = pred_dirs[mask]

                # Calculate R-Returns
                # logic: 4x TP / 2x SL => Reward/Risk = 2.0
                is_win = t_pred == t_actual
                is_time_exit = t_actual == 0
                is_loss = (~is_win) & (~is_time_exit)

                r_returns = torch.zeros_like(t_actual, dtype=torch.float32)
                # If TP hit, reward is 4.0 ATR. If SL hit, loss is 2.0 ATR.
                # In terms of 'R' where 1R = SL (2.0 ATR), win = 2.0R, loss = -1.0R.
                r_returns[is_win] = 2.0  
                r_returns[is_time_exit] = -0.2 # Small average timeout cost
                r_returns[is_loss] = -1.0 

                avg_r = r_returns.mean().item()

                if avg_r > best_expectancy:
                    best_expectancy = avg_r
                    best_params = {
                        "meta_threshold": round(m_t.item(), 4),
                        "qual_threshold": round(q_t.item(), 4),
                        "expected_win_rate": round(is_win.float().mean().item(), 4),
                        "expected_avg_r": round(avg_r, 4),
                        "num_trades_dataset": num_trades,
                        "optimized_at": datetime.now().isoformat(),
                        "alpha_only": alpha_only
                    }
                    if not alpha_only:
                        best_params["risk_threshold"] = round(r_t.item(), 4)

    # 5. Save Optimal Params to JSON
    if best_params:
        logger.info("\n" + "=" * 40)
        logger.info(f"OPTIMAL {'ALPHA-ONLY' if alpha_only else 'ALPHA-RISK'} THRESHOLDS")
        logger.info("=" * 40)
        logger.info(f"Alpha Meta Thresh:  {best_params['meta_threshold']:.4f}")
        logger.info(f"Alpha Qual Thresh:  {best_params['qual_threshold']:.4f}")
        if not alpha_only:
            logger.info(f"Risk Size Thresh:   {best_params['risk_threshold']:.4f}")
        logger.info(f"Expected Avg R:     {best_params['expected_avg_r']:.3f}")
        logger.info(f"Total Trades:       {best_params['num_trades_dataset']}")
        logger.info("=" * 40)

        filename = "optimal_thresholds_alpha_solo.json" if alpha_only else "optimal_thresholds.json"
        output_file = PROJECT_ROOT / "backtest" / "results" / filename
        os.makedirs(output_file.parent, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(best_params, f, indent=4)

        logger.info(f"Optimal parameters saved to {output_file}")
    else:
        logger.warning("No valid threshold combinations found.")


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
    parser.add_argument(
        "--alpha-only",
        action="store_true",
        help="Optimize for alpha-only backtest (fixed SL/TP)",
    )
    args = parser.parse_args()
    optimize_thresholds_main(
        alpha_model=args.alpha_model,
        risk_model=args.risk_model,
        risk_scaler=args.risk_scaler,
        data_dir=args.data_dir,
        alpha_only=args.alpha_only
    )


if __name__ == "__main__":
    optimize_thresholds()
