import subprocess
import sys
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command, description):
    logging.info(f"--- Starting: {description} ---")
    start_time = datetime.now()
    
    try:
        # Use shell=True for Windows compatibility with python command if needed, 
        # but list format is safer. Assumes 'python' is in PATH.
        # process = subprocess.run(command, shell=True, check=True) # If command is a string
        process = subprocess.run(command, check=True) # If command is a list
        
        duration = datetime.now() - start_time
        logging.info(f"--- Completed: {description} (Duration: {duration}) ---")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"!!! Failed: {description} (Exit Code: {e.returncode}) !!!")
        return False

def main():
    # 1. Download Training Data
    # Note: Using defaults (output to 'data' dir)
    cmd_download = [sys.executable, "RiskLayer/download_training_data.py"]
    if not run_command(cmd_download, "Download Raw Training Data"):
        return

    # 2. Generate Alpha Signals
    # This reads from 'data/' and writes to 'data/' with alpha labels
    cmd_signals = [sys.executable, "RiskLayer/src/generate_alpha_signals.py"]
    if not run_command(cmd_signals, "Generate Alpha Signals"):
        return

    # 3. Train Risk Model
    # Reads config from RiskLayer/config/ppo_config.yaml
    cmd_train = [sys.executable, "RiskLayer/train/train_risk_model.py"]
    if not run_command(cmd_train, "Train Risk Model (TradeGuard)"):
        return

    logging.info("=== Full Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
