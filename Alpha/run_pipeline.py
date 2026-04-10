import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from datetime import datetime
from tqdm import tqdm
import gc
import subprocess

# ... [Keep your imports and AlphaDataset class as they were] ...

def main():
    parser = argparse.ArgumentParser(description="Alpha Layer Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory containing OHLCV parquet files")
    parser.add_argument("--skip-gen", action="store_true", help="Skip dataset generation")
    parser.add_argument("--smoke-test", action="store_true", help="Run with limited samples")
    
    # Resolved Logic: Keep both to maintain backward compatibility but default to running
    parser.add_argument("--run-post-backtest", action="store_true", help="Run threshold optimizer and combined backtest after training")
    parser.add_argument("--skip-post-backtest", action="store_true", help="Skip threshold optimizer and combined backtest after training")
    
    parser.add_argument("--risk-model", type=str, default="Risklayer/models/ppo_risk_model_final_v2_opt.zip", help="Path to RL risk model")
    parser.add_argument("--risk-scaler", type=str, default="Risklayer/models/rl_risk_scaler.pkl", help="Path to RL risk scaler")
    parser.add_argument("--backtest-data-dir", type=str, default="backtest/data", help="Backtest data directory")
    parser.add_argument("--backtest-output-dir", type=str, default="backtest/results", help="Backtest results output directory")
    parser.add_argument("--backtest-max-steps", type=int, default=2000, help="Max steps for post-training backtest")
    
    args = parser.parse_args()
    
    # Logic: Run backtest UNLESS explicitly skipped, or if run-post-backtest is toggled
    run_post_backtest = (not args.skip_post_backtest) or args.run_post_backtest
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(os.path.join(base_dir, args.data_dir))

    dataset_dir = os.path.join(base_dir, "data", "training_set")
    model_path = os.path.join(base_dir, "models", "alpha_model.pth")
    
    if not args.skip_gen:
        if not os.path.exists(args.data_dir):
            logger.error(f"Data directory not found at {args.data_dir}")
            return
        generate_dataset(args.data_dir, dataset_dir, smoke_test=args.smoke_test)
    
    features_path = os.path.join(dataset_dir, "features.npy")
    labels_path = os.path.join(dataset_dir, "labels.npz")
    
    if os.path.exists(features_path) and os.path.exists(labels_path):
        train_model(features_path, labels_path, model_path)

        if run_post_backtest:
            alpha_model_relpath = os.path.relpath(model_path, os.path.dirname(base_dir))
            logger.info("Starting post-training threshold optimization...")
            optimizer_cmd = [
                sys.executable, "-m", "backtest.optimize_thresholds",
                "--alpha-model", alpha_model_relpath,
                "--risk-model", args.risk_model,
                "--risk-scaler", args.risk_scaler,
                "--data-dir", args.backtest_data_dir,
            ]
            logger.info(f"Running: {' '.join(optimizer_cmd)}")
            subprocess.run(optimizer_cmd, check=True)

            logger.info("Starting post-training combined backtest...")
            backtest_cmd = [
                sys.executable, "-m", "backtest.backtest_combined",
                "--alpha-model", alpha_model_relpath,
                "--risk-model", args.risk_model,
                "--risk-scaler", args.risk_scaler,
                "--data-dir", args.backtest_data_dir,
                "--output-dir", args.backtest_output_dir,
                "--max-steps", str(args.backtest_max_steps),
                "--episodes", "1",
            ]
            logger.info(f"Running: {' '.join(backtest_cmd)}")
            subprocess.run(backtest_cmd, check=True)
        else:
            logger.info("Skipping post-training optimizer/backtest.")
    else:
        logger.error(f"Dataset not found in {dataset_dir}. Cannot train.")

if __name__ == "__main__":
    main()