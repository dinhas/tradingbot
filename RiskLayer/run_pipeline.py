import subprocess
import sys
import os
import time

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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("🚀 STARTING RISK LAYER PIPELINE")
    print(f"Working Directory: {base_dir}")
    
    # 1. Fetch Data
    if not run_step("Fetch Training Data", "download_training_data.py", base_dir):
        sys.exit(1)
        
    # 2. Generate Dataset
    if not run_step("Generate Risk Dataset", "generate_risk_dataset.py", base_dir):
        sys.exit(1)
        
    # 3. Train Model
    # Note: train_risk.py is configured for 'MAX SPEED'
    if not run_step("Train Risk Agent", "train_risk.py", base_dir):
        sys.exit(1)

    print("\n" + "="*50)
    print("✅✅ ALL PIPELINE STEPS COMPLETED SUCCESSFULLY ✅✅")
    print("="*50)

if __name__ == "__main__":
    main()
