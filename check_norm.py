import pickle
import numpy as np

# Mocking parts of SB3 if needed, but let's try direct pickle first
# VecNormalize might need the class definition if it's not standard pickle
try:
    from stable_baselines3.common.vec_env import VecNormalize
    with open('models/risk/vec_normalize.pkl', 'rb') as f:
        # VecNormalize.load handles the structure
        import torch # sometimes needed for SB3
        # We don't want to load the whole env, just the stats
        # But VecNormalize.load needs a venv
        print("Attempting to load with pickle...")
        data = pickle.load(f)
        if hasattr(data, 'obs_rms'):
             print(f"Obs RMS Mean Shape: {data.obs_rms.mean.shape}")
        else:
             print(f"Data keys: {data.keys() if isinstance(data, dict) else type(data)}")
             if isinstance(data, dict) and 'obs_rms' in data:
                 print(f"Obs RMS Mean Shape (from dict): {data['obs_rms'].mean.shape}")

except Exception as e:
    print(f"Error: {e}")
