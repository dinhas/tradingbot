import subprocess
import sys
import os
import time
from datetime import datetime

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

def run_step(step_name, script_name, cwd):
    print(f"\n{'='*50}")
    print(f"STEP: {step_name}")
    print(f"Script: {script_name}")
    print(f"{'='*50}\n")

    start_time = time.time()
    
    script_path = os.path.join(cwd, script_name)
    if not os.path.exists(script_path):
        print(f"❌ Error: Script not found -> {script_path}")
        return False

    try:
        # Run using the same python interpreter
        result = subprocess.run(
            [sys.executable, script_name], 
            cwd=cwd,
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
    # Use relative path if possible, or dot
    base_dir = os.path.dirname(__file__)
    if not base_dir:
        base_dir = "."
    
    print("STARTING RISK LAYER PIPELINE")
    print(f"Working Directory: {base_dir}")
    
    # 1. Fetch Data
    project_root = os.path.dirname(base_dir)
    data_dir = os.path.join(project_root, "data")
    required_assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    
    data_missing = False
    for asset in required_assets:
        # Check for standard naming or 2025 version
        if not os.path.exists(os.path.join(data_dir, f"{asset}_5m.parquet")) and \
           not os.path.exists(os.path.join(data_dir, f"{asset}_5m_2025.parquet")):
            data_missing = True
            break
    
    if data_missing:
        if not run_step("Fetch Training Data", "download_training_data.py", base_dir):
            sys.exit(1)
    else:
        print("\n✅ All required market data found. Skipping download step.")
        
    # 2. Generate Dataset
    if not run_step("Generate Risk Dataset", "generate_risk_dataset.py", base_dir):
        sys.exit(1)
        
    # 3. Train Model
    # Note: train_risk.py is configured for 'MAX SPEED'
    if not run_step("Train Risk Agent", "train_risk.py", base_dir):
        sys.exit(1)

    print("\n" + "="*50)
    print("ALL PIPELINE STEPS COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    # Setup logging to file - capture all terminal output
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    tee = Tee(log_file)
    
    try:
        print(f"All terminal output will be saved to: {log_file}")
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}")
    finally:
        tee.close()
        print(f"Log saved to: {log_file}")
