import sys
import os
import pandas as pd
from pathlib import Path

# Add project root
project_root = str(Path(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

# Mock stable_baselines3 PPO load to avoid needing actual model files if they are missing or corrupt
# But the user said they are there. Let's try real import first.
try:
    from TradeGuard.src.generate_dataset import TrainingDatasetGenerator
    from Alpha.src.trading_env import TradingEnv
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_generation_setup():
    print("Initializing Generator...")
    try:
        # We need a valid model path or at least the file to exist? 
        # The generator __init__ doesn't load the model, it just stores the path.
        # But generate() does load it. Here we just test __init__ and manual slicing.
        generator = TrainingDatasetGenerator("Alpha/models/checkpoints/8.03.zip")
    except Exception as e:
        print(f"Generator init failed: {e}")
        return

    print("Checking data alignment...")
    # Check if indices match
    indices = [df.index for df in generator.full_data.values()]
    first_index = indices[0]
    for i, idx in enumerate(indices[1:]):
        if not first_index.equals(idx):
            print(f"MISMATCH found at index {i+1}")
            print(f"First index len: {len(first_index)}")
            print(f"This index len: {len(idx)}")
        else:
            print(f"Index {i+1} matches.")
            
    chunk_size = 1000
    warmup = 500
    start_idx = 0
    # Use the length of the first dataframe (they should all be equal now)
    total_rows = len(next(iter(generator.full_data.values())))
    end_idx = min(start_idx + chunk_size, total_rows)
    
    slice_start = max(0, start_idx - warmup)
    
    print(f"Slicing {slice_start} to {end_idx}...")
    sliced_data = {asset: generator.full_data[asset].iloc[slice_start:end_idx] for asset in generator.assets}
    
    print("Initializing TradingEnv with slice...")
    try:
        # This triggers feature_engine.preprocess_data -> _add_technical_indicators -> AverageTrueRange
        raw_env = TradingEnv(data=sliced_data, is_training=False, stage=1)
        print("✅ TradingEnv initialized successfully!")
    except Exception as e:
        print(f"❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generation_setup()
