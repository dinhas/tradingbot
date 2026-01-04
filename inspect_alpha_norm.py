
import os
import sys
from pathlib import Path
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the environment to create a dummy instance
from Alpha.src.trading_env import TradingEnv

def inspect_normalizer(norm_path, data_dir):
    print(f"--- Inspecting Normalizer: {norm_path} ---")
    
    if not os.path.exists(norm_path):
        print(f"Error: File not found at {norm_path}")
        return

    # Create a dummy environment required to load VecNormalize
    # We use a small dummy config or just the class defaults
    # We need to make sure data_dir is valid or Mock it if necessary
    # Assuming backtest/data exists as per previous context
    try:
        # Create a basic dummy env
        env = DummyVecEnv([lambda: TradingEnv(data_dir=data_dir, stage=1, is_training=False)])
        
        # Load the normalizer
        norm_env = VecNormalize.load(norm_path, env) 
        
        print("\nSuccessfully loaded VecNormalize object.")
        
        # Access internal running mean and variance
        # obs_rms is a RunningMeanStd object
        obs_mean = norm_env.obs_rms.mean
        obs_var = norm_env.obs_rms.var
        
        print(f"\nFeature Dimension: {obs_mean.shape[0]}")
        print("-" * 30)
        
        # Print stats for each feature
        # Assuming 40 features as per previous context (25 asset + 15 global)
        # We can try to map indices to feature names if we know them, but raw stats are fine for now.
        
        print(f"{ 'Index':<6} | { 'Mean':<12} | { 'Variance':<12} | { 'Std Dev':<12}")
        print("-" * 50)
        
        for i in range(len(obs_mean)):
            mean_val = obs_mean[i]
            var_val = obs_var[i]
            std_val = np.sqrt(var_val)
            print(f"{i:<6} | {mean_val:>12.4f} | {var_val:>12.4f} | {std_val:>12.4f}")
            
        print("-" * 50)
        
        # Check if reward normalization is enabled/tracked
        if hasattr(norm_env, 'ret_rms') and norm_env.ret_rms is not None:
            print(f"\nReward Running Mean: {norm_env.ret_rms.mean}")
            print(f"Reward Running Var:  {norm_env.ret_rms.var}")
        else:
            print("\nNo reward normalization stats found.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Define paths
    NORMALIZER_PATH = "models/checkpoints/alpha/ppo_final_vecnormalize.pkl"
    DATA_DIR = "backtest/data" # Needed to instantiate env
    
    inspect_normalizer(NORMALIZER_PATH, DATA_DIR)
