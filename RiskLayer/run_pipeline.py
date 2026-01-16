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

def run_step(step_name, script_name, cwd, extra_args=None):
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
        cmd = [sys.executable, script_name]
        if extra_args:
            cmd.extend(extra_args)
            
        result = subprocess.run(
            cmd, 
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
    # Get absolute paths to ensure consistency
    risk_layer_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(risk_layer_dir)
    
    # Consistent data directory
    data_dir = os.path.join(project_root, "Alpha", "data")
    risk_dataset_path = os.path.join(risk_layer_dir, "risk_dataset.parquet")
    
    print("STARTING RISK LAYER PIPELINE")
    print(f"Project Root: {project_root}")
    print(f"Risk Layer Dir: {risk_layer_dir}")
    print(f"Data Directory: {data_dir}")
    
    # 1. Fetch Data
    # download_training_data.py saves to 'Alpha/data' by default, 
    # but we'll be explicit using absolute path.
    if not run_step("Fetch Training Data", "download_training_data.py", risk_layer_dir, ["--output", data_dir]):
        print("CRITICAL: Fetching training data failed. Stopping pipeline.")
        sys.exit(1)
        
    # 2. Generate Dataset
    # We pass the same data_dir to generate_risk_dataset.py
    if not run_step("Generate Risk Dataset", "generate_risk_dataset.py", risk_layer_dir, ["--data", data_dir, "--output", risk_dataset_path]):
        print("CRITICAL: Generating risk dataset failed. Stopping pipeline.")
        sys.exit(1)
        
    # 3. Train Model
    # We pass the generated risk_dataset.parquet path to train_risk.py
    if not run_step("Train Risk Agent", "train_risk.py", risk_layer_dir, ["--dataset", risk_dataset_path]):
        print("CRITICAL: Training risk agent failed. Stopping pipeline.")
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
