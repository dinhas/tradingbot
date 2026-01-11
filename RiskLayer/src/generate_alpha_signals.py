import sys
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import logging
from pathlib import Path

# Setup Paths
sys.path.append(os.getcwd())
# Add Alpha to path so we can import its modules
sys.path.append(os.path.join(os.getcwd(), 'Alpha', 'src'))

try:
    from Alpha.src.trading_env import TradingEnv
except ImportError:
    # Fallback if running from root
    from Alpha.src.trading_env import TradingEnv

def generate_signals():
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    model_path = "models/checkpoints/alpha/ppo_final_model.zip"
    norm_path = "models/checkpoints/alpha/ppo_final_vecnormalize.pkl"
    output_dir = "data"
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(model_path):
        logging.error(f"Model not found at {model_path}")
        return

    # Load Model
    logging.info("Loading Alpha Model...")
    # We need a dummy env to load VecNormalize correctly
    dummy_env = DummyVecEnv([lambda: TradingEnv(is_training=False)])
    try:
        # Load VecNormalize statistics
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, dummy_env)
            env.training = False # Don't update stats during inference
            env.norm_reward = False # We don't need reward normalization for inference
        else:
            logging.warning("VecNormalize file not found! Predictions might be garbage if model expects normalized data.")
            env = dummy_env
            
        model = PPO.load(model_path, env=env)
        logging.info("Model Loaded Successfully.")
        
    except Exception as e:
        logging.error(f"Failed to load model/env: {e}")
        return

    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    
    for asset in assets:
        logging.info(f"Generating signals for {asset}...")
        
        # Initialize specific env for this asset
        # We access the internal env of the VecEnv wrapper
        raw_env = env.envs[0] 
        raw_env.set_asset(asset)
        
        # We need to access the full dataset from the env
        # The env loads data in __init__. Let's get the processed dataframe.
        # Note: raw_env.data[asset] is the source. raw_env.processed_data contains features.
        # But raw_env.processed_data is a single DataFrame for *all* assets?
        # Looking at TradingEnv code: "self.raw_data, self.processed_data = self.feature_engine.preprocess_data(self.data)"
        # It seems processed_data has columns like 'EURUSD_close', etc.
        
        # We will iterate through the dataset from index 0 to max
        # NOTE: The env usually starts at a random point. We want SEQUENTIAL inference.
        
        # Extract the relevant timestamps and price data for saving
        # We'll rely on the env's internal pointers.
        
        obs = env.reset()
        # Force the underlying env to the specific asset again just in case reset shuffled it
        raw_env.current_asset = asset
        # Force start at specific step (e.g. 500 where data becomes valid) 
        start_step = 500 
        raw_env.current_step = start_step
        
        signals = []
        confidences = []
        timestamps = []
        closes = []
        
        max_steps = raw_env.max_steps
        
        # Loop through all available data
        # We manually step the environment's current_step pointer or rely on step()
        # step() increments current_step.
        
        # We need to loop until we run out of data
        logging.info(f"Processing {max_steps - start_step} bars...")
        
        # Get the DataFrame to align timestamps later
        # We can extract timestamps directly from the index if available
        df_index = raw_env.processed_data.index
        
        current_step = start_step
        
        while current_step < max_steps - 1:
            # 1. Get Observation
            # We use the raw_env method to get the observation for the specific step
            # But we must wrap it for VecNormalize if used
            raw_obs = raw_env._get_observation()
            
            # Normalize observation using the VecNormalize wrapper
            if isinstance(env, VecNormalize):
                norm_obs = env.normalize_obs(raw_obs)
            else:
                norm_obs = raw_obs
                
            # 2. Predict
            # deterministic=True ensures no random noise in action
            action, _states = model.predict(norm_obs, deterministic=True)
            
            # 3. Interpret Action
            # Alpha Env: > 0.33 BUY, < -0.33 SELL
            raw_val = action[0] if isinstance(action, np.ndarray) else action
            
            signal = 0.0
            if raw_val > 0.33:
                signal = 1.0
            elif raw_val < -0.33:
                signal = -1.0
                
            # 4. Store
            signals.append(signal)
            confidences.append(raw_val)
            timestamps.append(df_index[current_step])
            
            # Get close price for reference
            # Using cached array for speed
            closes.append(raw_env.close_arrays[asset][current_step])
            
            # 5. Advance
            current_step += 1
            raw_env.current_step = current_step
            
            if current_step % 10000 == 0:
                print(f"Processed {current_step}/{max_steps}", end='\r')
                
        # The env has the original raw data in raw_env.data[asset]
        original_df = raw_env.data[asset].iloc[start_step:current_step].copy()
        
        # Add new columns
        original_df['alpha_signal'] = signals
        original_df['alpha_confidence'] = confidences
        
        # Save
        out_path = f"{output_dir}/{asset}_alpha_labeled.parquet"
        original_df.to_parquet(out_path)
        logging.info(f"\nSaved {out_path} with {len(original_df)} rows. Signals: {original_df['alpha_signal'].abs().sum()}")

if __name__ == "__main__":
    generate_signals()
