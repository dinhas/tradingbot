import os
import pickle
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.running_mean_std import RunningMeanStd

def inspect_normalizer(path):
    print(f"--- Inspecting Normalizer: {path} ---")
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return

    try:
        # Load the pickled VecNormalize object
        with open(path, 'rb') as f:
            venv = pickle.load(f)
        
        # Access the RunningMeanStd object for observations
        # In SB3 VecNormalize, this is stored in self.obs_rms
        obs_rms = venv.obs_rms
        
        # Print stats
        mean = obs_rms.mean
        var = obs_rms.var
        count = obs_rms.count
        
        print(f"Count (steps updated): {count}")
        print(f"Feature Dimensions: {mean.shape[0]}")
        
        print("\nFeature Statistics (Mean / Variance):")
        print("Idx | Mean         | Variance     | Std Dev      | Status")
        print("-" * 65)
        
        has_issues = False
        for i in range(len(mean)):
            m = mean[i]
            v = var[i]
            std = np.sqrt(v)
            
            status = "OK"
            if v < 1e-4:
                status = "VARIANCE TOO LOW (Const?)"
                has_issues = True
            elif abs(m) > 100:
                status = "MEAN TOO HIGH (Unscaled?)"
                has_issues = True
            elif np.isnan(m) or np.isnan(v):
                status = "NAN DETECTED"
                has_issues = True
                
            print(f"{i:3d} | {m:12.4f} | {v:12.4f} | {std:12.4f} | {status}")

        if has_issues:
            print("\nWARNING: Some features have suspicious statistics.")
            print("- Low Variance: Feature might be constant (broken input).")
            print("- High Mean/Var: Feature might be unnormalized price data.")
        else:
            print("\nNormalizer looks healthy.")
            
    except Exception as e:
        print(f"Failed to inspect: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check both potentially relevant normalizers
    inspect_normalizer("models/checkpoints/tradegurd/tradeguard_ppotest_vecnormalize.pkl")
    # inspect_normalizer("models/checkpoints/alpha/ppo_final_vecnormalize.pkl") # Helper comparison
