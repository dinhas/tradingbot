
import time
import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Alpha.src.trading_env import TradingEnv

def benchmark_inference(model_path, data_dir, n_iterations=100):
    print(f"Loading model from {model_path}...")
    print(f"Loading data from {data_dir}...")
    
    # 1. Setup Environment and Model
    # We use Stage 1 as requested/implied by the model name, or default to 1 if not specified
    stage = 1
    if 'stage_2' in model_path: stage = 2
    if 'stage_3' in model_path: stage = 3
    
    try:
        env = TradingEnv(data_dir=data_dir, stage=stage, is_training=False)
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model or env: {e}")
        return

    print("Data and Model loaded successfully.")
    print("-" * 50)
    print(f"Running benchmark for {n_iterations} iterations...")
    
    feature_times = []
    inference_times = []
    total_times = []
    
    # Reset environment to get initial state
    env.reset()
    
    for i in range(n_iterations):
        # Pick a random step (ensure enough history for lookback if needed, though env handles this)
        # We simulate "getting a random data line" by setting the current step 
        rand_step = np.random.randint(50, len(env.processed_data) - 100)
        env.current_step = rand_step
        
        # --- START BENCHMARK ---
        start_time = time.perf_counter()
        
        # 1. Feature Extraction (Convert data line to 140 features)
        # We measure the time to construct the observation vector
        obs = env._get_observation()
        
        feature_time = time.perf_counter()
        
        # 2. Model Inference (Get output)
        action, _ = model.predict(obs, deterministic=True)
        
        inference_time = time.perf_counter()
        
        # End of inference cycle
        
        # Calculate durations
        t_feature = feature_time - start_time
        t_inference = inference_time - feature_time
        t_total = inference_time - start_time
        
        feature_times.append(t_feature)
        inference_times.append(t_inference)
        total_times.append(t_total)
        
    # Calculate statistics
    avg_feature = np.mean(feature_times) * 1000  # ms
    avg_inference = np.mean(inference_times) * 1000 # ms
    avg_total = np.mean(total_times) * 1000 # ms
    
    print("-" * 50)
    print(f"BENCHMARK RESULTS (averaged over {n_iterations} runs)")
    print("-" * 50)
    print(f"Input Processing (Feature Construction):  {avg_feature:.4f} ms")
    print(f"Model Inference (PPO Predict):            {avg_inference:.4f} ms")
    print(f"Total Prediction Latency:                 {avg_total:.4f} ms")
    print("-" * 50)
    print(f"Detailed Stats:")
    print(f"  Max Total Time: {np.max(total_times)*1000:.4f} ms")
    print(f"  Min Total Time: {np.min(total_times)*1000:.4f} ms")
    print("-" * 50)

if __name__ == "__main__":
    # Check for available models
    models_dir = "models/checkpoints"
    available_models = [f for f in os.listdir(models_dir) if f.endswith('.zip')]
    
    if not available_models:
        print("No models found in models/checkpoints/")
        sys.exit(1)
        
    # Pick the first available model or preferred one
    model_name = "stage_1_final(4).zip"
    if model_name not in available_models:
        model_name = available_models[0]
        
    model_path = os.path.join(models_dir, model_name)
    data_dir = "backtest/data"
    
    benchmark_inference(model_path, data_dir)
