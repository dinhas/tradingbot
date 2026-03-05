import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"pipeline_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)


def run_command(command, description, cwd=None, env=None):
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
            universal_newlines=True,
            cwd=cwd,
            env=env,
        )

        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            sys.exit(process.returncode)

        logger.info(f"--- Completed: {description} ---")
    except Exception as e:
        logger.error(f"Error executing {description}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Unified RiskLayer RL/PPO Pipeline")
    parser.add_argument(
        "--ppo", action="store_true", help="Run the PPO pipeline instead of the basic RL"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="Run a quick test with 5000 samples"
    )
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="Skip data generation and go straight to training",
    )
    parser.add_argument(
        "--data-dir", type=str, default="../data", help="Directory for raw market data"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate",
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = project_root

    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    dataset_name = "ppo_risk_dataset.parquet"
    if args.smoke_test:
        dataset_name = "smoke_test_ppo_risk_dataset.parquet"

    dataset_path = os.path.join(data_dir, dataset_name)

    # 1. Data Generation
    if not args.skip_gen:
        gen_script = os.path.join(base_dir, "generate_rl_risk_dataset.py")
        gen_cmd = [
            sys.executable,
            gen_script,
            "--data",
            args.data_dir,
            "--output",
            dataset_path,
        ]

        if args.smoke_test:
            gen_cmd.extend(["--max-samples", "5000"])
        elif args.max_samples:
            gen_cmd.extend(["--max-samples", str(args.max_samples)])

        run_command(gen_cmd, "PPO Data Generation (Alpha Filtered)", cwd=base_dir, env=env)
    else:
        logger.info("Skipping Data Generation as requested.")

    # 2. Training
    os.environ["PPO_DATASET_PATH"] = dataset_path # Pass to training script
    
    if args.ppo:
        train_script = os.path.join(base_dir, "train_risk_ppo.py")
        description = "PPO Model Training"
    else:
        os.environ["RL_DATASET_PATH"] = dataset_path
        train_script = os.path.join(base_dir, "train_risk_rl.py")
        description = "Basic RL Model Training"

    train_cmd = [sys.executable, train_script]
    run_command(train_cmd, description, cwd=base_dir, env=env)

    logger.info("==========================================")
    logger.info("Pipeline Execution Finished Successfully!")
    if args.ppo:
        logger.info(f"Model and Normalization stats are in {os.path.join(base_dir, 'models')}")
    else:
        logger.info(f"Model and Scaler are in {os.path.join(base_dir, 'models')}")
    logger.info("==========================================")


if __name__ == "__main__":
    main()
