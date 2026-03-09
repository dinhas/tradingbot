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

# Standardized to RiskLayer (PascalCase) as per optimized backtest script
_risklayer_src = PROJECT_ROOT / "RiskLayer" / "src"
if str(_risklayer_src) not in sys.path:
    sys.path.insert(0, str(_risklayer_src))

from Alpha.src.model import AlphaSLModel
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.labeling import Labeler
from risk_model_sl import RiskModelSL
from backtest.data_fetcher_backtest import DataFetcherBacktest, SYMBOL_IDS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        fetcher.start()  
    else:
        logger.info("All 2025 backtest data found.")

def load_data(data_dir):
    """Loads 2025 data for all assets."""
    data_dict = {}
    for asset in SYMBOL_IDS.keys():
        file_path = data_dir / f"{asset}_5m_2025.parquet"
        if file_path.exists():
            df = pd.read_parquet(file_path)
            df.index = pd.to_datetime(df.index)
            data_dict[asset] = df
        else:
            logger.warning(f"Could not load data for {asset}")
    return data_dict

def optimize_thresholds():
    parser = argparse.ArgumentParser(description="Alpha-Risk Multi-Model Threshold Optimizer")
    parser.add_argument("--alpha-model", type=str, default="Alpha/models/alpha_model.pth")
    parser.add_argument("--risk-model", type=str, default="RiskLayer/models/ppo_risk_model_final_v2_opt.zip")
    parser.add_argument("--risk-scaler", type=str, default="RiskLayer/models/rl_risk_scaler.pkl")
    parser.add_argument("--use-rl", action="store_true", default=True)
    parser.add_argument("--data-dir", type=str, default="backtest/data")
    args = parser.parse_args()

    alpha_path = PROJECT_ROOT / args.alpha_model
    risk_path = PROJECT_ROOT / args.risk_model
    scaler_path = PROJECT_ROOT / args.risk_scaler
    data_dir = PROJECT_ROOT / args.data_dir
    use_rl = args.use_rl

    ensure_2025_data(data_dir)
    data_dict = load_data(data_dir)
    if not data_dict: return

    # Load Models
    alpha_model = AlphaSLModel(input_dim=40, hidden_dim=256, num_res_blocks=4).to(DEVICE)
    alpha_model.load_state_dict(torch.load(alpha_path, map_location=DEVICE))
    alpha_model.eval()

    if use_rl and risk_path.exists() and risk_path.suffix == ".zip":
        risk_model = PPO.load(str(risk_path), device=DEVICE)
    else:
        risk_model = RiskModelSL(input_dim=40).to(DEVICE)
        risk_model.load_state_dict(torch.load(risk_path, map_location=DEVICE))
        risk_model.eval()

    risk_scaler = joblib.load(scaler_path)

    # Signal Generation
    engine = FeatureEngine()
    labeler = Labeler(stride=5)
    
    results = {"pred_dirs": [], "meta_probs": [], "qual_preds": [], "risk_sizes": [], "actual_dirs": []}

    aligned_df, normalized_df = engine.preprocess_data(data_dict)

    for asset in data_dict.keys():
        labels_df = labeler.label_data(aligned_df, asset)
        idx = labels_df.index.intersection(normalized_df.index)
        X = engine.get_observation_vectorized(normalized_df.loc[idx], asset)
        X_risk = torch.from_numpy(risk_scaler.transform(X).astype(np.float32)).to(DEVICE)
        X_alpha = torch.from_numpy(X).to(DEVICE)
        y_true = torch.from_numpy(labels_df.loc[idx, "direction"].values).to(DEVICE).long()

        with torch.no_grad():
            for i in range(0, len(X_alpha), 8192):
                b_alpha = X_alpha[i:i+8192]
                b_risk = X_risk[i:i+8192]

                d_logits, q_pred, m_logits = alpha_model(b_alpha)
                
                if use_rl:
                    act, _ = risk_model.predict(b_risk.cpu().numpy(), deterministic=True)
                    r_size = 0.1 + (torch.from_numpy(act[:, 2]).to(DEVICE) + 1) / 2 * (0.3 - 0.1)
                else:
                    r_size = risk_model(b_risk)["size"]

                conf, p_dir = torch.max(torch.softmax(d_logits, dim=1), dim=1)
                
                results["pred_dirs"].append(p_dir - 1)
                results["meta_probs"].append(torch.sigmoid(m_logits).squeeze())
                results["qual_preds"].append(q_pred.squeeze())
                results["risk_sizes"].append(r_size.squeeze())
                results["actual_dirs"].append(y_true[i:i+8192])

    # 3D Grid Search (Vectorized)
    p_dirs = torch.cat(results["pred_dirs"])
    m_probs = torch.cat(results["meta_probs"])
    q_preds = torch.cat(results["qual_preds"])
    r_sizes = torch.cat(results["risk_sizes"])
    a_dirs = torch.cat(results["actual_dirs"])

    meta_ts = torch.linspace(0.6, 0.95, 10).to(DEVICE)
    qual_ts = torch.linspace(0.4, 0.8, 8).to(DEVICE)
    risk_ts = torch.linspace(0.3, 0.7, 8).to(DEVICE)

    best_expectancy, best_params = -np.inf, None

    for m_t in tqdm(meta_ts, desc="Optimizing"):
        for q_t in qual_ts:
            for r_t in risk_ts:
                mask = (m_probs >= m_t) & (q_preds >= q_t) & (r_sizes >= r_t) & (p_dirs != 0)
                if mask.sum().item() < 50: continue

                t_actual, t_pred = a_dirs[mask], p_dirs[mask]
                r_ret = torch.zeros_like(t_actual, dtype=torch.float32)
                r_ret[t_pred == t_actual] = 4.0        # Win
                r_ret[t_actual == 0] = -0.5            # Time-out
                r_ret[(t_pred != t_actual) & (t_actual != 0)] = -1.5 # Loss

                avg_r = r_ret.mean().item()
                if avg_r > best_expectancy:
                    best_expectancy = avg_r
                    best_params = {
                        "meta_threshold": round(m_t.item(), 4),
                        "qual_threshold": round(q_t.item(), 4),
                        "risk_threshold": round(r_t.item(), 4),
                        "expected_avg_r": round(avg_r, 4),
                        "num_trades_2025": mask.sum().item()
                    }

    if best_params:
        logger.info(f"Optimal Thresholds Found: Meta {best_params['meta_threshold']}, Qual {best_params['qual_threshold']}")
        with open(PROJECT_ROOT / "backtest" / "results" / "optimal_thresholds.json", "w") as f:
            json.dump(best_params, f, indent=4)

if __name__ == "__main__":
    optimize_thresholds()