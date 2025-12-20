import os
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Runs a shell command and checks for errors."""
    logger.info(f"Starting: {description}")
    try:
        # Run command and stream output
        process = subprocess.Popen(
            command, 
            shell=True, 
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"Failed: {description} (Exit Code: {process.returncode})")
            sys.exit(process.returncode)
            
        logger.info(f"Completed: {description}")
        
    except Exception as e:
        logger.error(f"Error executing {description}: {e}")
        sys.exit(1)

def main():
    logger.info("Starting TradeGuard Training Pipeline...")
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # 1. Download Data
    # We rely on the script's internal logic to skip download if files exist.
    # We do NOT pass --force.
    logger.info(f"Checking data in {DATA_DIR}...")
    run_command(f"python {os.path.join(BASE_DIR, 'download_training_data.py')}", "Data Download Step")

    # 2. Generate Dataset
    # This script reads from Guard/data (as modified) and writes to Guard/data
    run_command(f"python {os.path.join(BASE_DIR, 'generate_dataset.py')}", "Dataset Generation Step")
    
    # 3. Train Model
    run_command(f"python {os.path.join(BASE_DIR, 'train_guard.py')}", "Model Training Step")
    
    logger.info("Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()
