import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

# --- Configuration ---
ASSETS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']

class Tee:
    """Redirect stdout/stderr to both console and file."""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = TeeStderr(self.file, self.stderr)
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        self.stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.stdout.flush()
    
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

class TeeStderr:
    """Handle stderr separately."""
    def __init__(self, file, stderr):
        self.file = file
        self.stderr = stderr
    
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stderr.write(text)
        self.stderr.flush()
    
    def flush(self):
        self.file.flush()
        self.stderr.flush()

def check_data_exists(data_dir):
    """Checks if all required asset data files exist."""
    missing = []
    for asset in ASSETS:
        file_path = data_dir / f"{asset}_5m.parquet"
        # Also check for the 2025 variant mentioned in the environment
        file_path_2025 = data_dir / f"{asset}_5m_2025.parquet"
        
        if not file_path.exists() and not file_path_2025.exists():
            missing.append(asset)
    return missing

def run_step(step_name, command, cwd):
    """Runs a pipeline step as a subprocess."""
    print(f"\n{'='*50}")
    print(f"STEP: {step_name}")
    print(f"Command: {' '.join(command)}")
    print(f"CWD: {cwd}")
    print(f"{'='*50}\n")

    start_time = time.time()
    
    try:
        # Run the command
        result = subprocess.run(
            command, 
            cwd=str(cwd),
            check=True,
            text=True
        )
        
        duration = time.time() - start_time
        print(f"\n✅ {step_name} COMPLETED successfully in {duration:.2f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {step_name} FAILED with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ {step_name} FAILED with error: {e}")
        return False

def main():
    alpha_dir = Path(__file__).resolve().parent
    project_root = alpha_dir.parent
    data_dir = project_root / "data"
    
    print("STARTING ALPHA MODEL PIPELINE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project Root: {project_root}")
    print(f"Alpha Directory: {alpha_dir}")
    print(f"Data Directory: {data_dir}")
    
    # Ensure data directory exists
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Check if data exists
    missing_assets = check_data_exists(data_dir)
    
    if missing_assets:
        print(f"⚠️ Missing data for: {', '.join(missing_assets)}")
        print("🚀 Starting Data Fetcher...")
        
        # Run Data Fetcher as a module
        fetch_cmd = [sys.executable, "-m", "src.data_fetcher"]
        if not run_step("Fetch Alpha Data", fetch_cmd, alpha_dir):
            print("❌ Data fetching failed. Aborting pipeline.")
            return False
    else:
        print("✅ All required data files found. Skipping data download.")
        
    # 2. Run Training
    print("🚀 Starting Alpha Model Training...")
    
    # Pass along any extra arguments from this script to the training script
    # e.g. python run_pipeline.py --dry-run
    train_args = sys.argv[1:]
    train_cmd = [sys.executable, "-m", "src.train"] + train_args
    
    if not run_step("Train Alpha Model", train_cmd, alpha_dir):
        print("❌ Training failed. Aborting pipeline.")
        return False

    print("\n" + "="*50)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
    print("="*50)
    return True

if __name__ == "__main__":
    # Setup logging
    ALPHA_DIR = Path(__file__).resolve().parent
    log_dir = ALPHA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"pipeline_alpha_{timestamp}.log"
    
    tee = Tee(str(log_file))
    
    try:
        print(f"Logging to: {log_file}")
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(130)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)
    finally:
        tee.close()
