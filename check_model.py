from stable_baselines3 import PPO
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Compatibility shim
if not hasattr(np, "_core"):
    import sys
    sys.modules["numpy._core"] = np.core

model_path = "models/checkpoints/ppo_final_model.zip"
try:
    model = PPO.load(model_path)
    print(f"Model loaded successfully.")
    print(f"Action Space: {model.action_space}")
    print(f"Observation Space: {model.observation_space}")
except Exception as e:
    print(f"Error loading model: {e}")
