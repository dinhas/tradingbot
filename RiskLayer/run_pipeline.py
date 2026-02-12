import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Utility to run shell commands and log output."""
    logger.info(f"--- Starting: {description} ---")
    logger.info(f"Running: {' '.join(command)}")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            sys.exit(process.returncode)
            
        logger.info(f"--- Completed: {description} ---")
    except Exception as e:
        logger.error(f"Error executing {description}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Unified RiskLayer SL Pipeline")
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick test with 5000 samples")
    parser.add_argument("--skip-gen", action="store_true", help="Skip data generation and go straight to training")
    parser.add_argument("--data-dir", type=str, default="../data", help="Directory for raw market data")
    
    args = parser.parse_args()
    
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_name = "sl_risk_dataset.parquet"
    if args.smoke_test:
        dataset_name = "smoke_test_sl_risk_dataset.parquet"
        
    dataset_path = os.path.join(data_dir, dataset_name)
    
    # 2. Data Generation Phase
    if not args.skip_gen:
        gen_cmd = [
            sys.executable, "generate_sl_dataset.py",
            "--data", args.data_dir,
            "--output", dataset_path
        ]
        
        if args.smoke_test:
            gen_cmd.extend(["--limit", "5000"])
            
        run_command(gen_cmd, "Data Generation (Oracle Labeling)")
    else:
        logger.info("Skipping Data Generation as requested.")

    # 3. Training Phase
    # We need to tell train_risk.py which dataset to use by modifying the environment or passing args
    # For simplicity, we'll set an environment variable that train_risk.py can check
    os.environ["SL_DATASET_PATH"] = dataset_path
    
    train_cmd = [sys.executable, "train_risk.py"]
    
    run_command(train_cmd, "Model Training (Multi-Task ResNet)")

    logger.info("==========================================")
    logger.info("Pipeline Execution Finished Successfully!")
    logger.info(f"Model and Scaler are in {os.path.join(base_dir, 'models')}")
    logger.info("==========================================")

if __name__ == "__main__":
    main()
