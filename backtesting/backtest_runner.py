import gymnasium as gym
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from trading_env import TradingEnv
import os
import matplotlib.pyplot as plt
import json

def run_backtest(model_path="final_model.zip", stats_path="final_model_vecnormalize.pkl", data_dir="backtesting/"):
    """
    Runs the trained model on the TradingEnv using backtest data.
    """
    print(f"Loading model from {model_path}...")
    
    # 1. Load the Environment
    # We must wrap it exactly as we did during training (DummyVecEnv + VecNormalize)
    # However, for inference, we set training=False and norm_reward=False
    # We point data_dir to the backtesting directory where parquet files are
    env = DummyVecEnv([lambda: TradingEnv(data_dir=data_dir, volatility_file="backtesting/volatility_baseline.json")])
    
    # Load normalization statistics
    if os.path.exists(stats_path):
        print(f"Loading normalization stats from {stats_path}...")
        env = VecNormalize.load(stats_path, env)
        env.training = False      # Do not update stats during inference
        env.norm_reward = False   # We want to see real rewards/PnL
    else:
        print("⚠️ Warning: Normalization stats not found! Model performance may be degraded.")

    # 2. Load the Model
    model = RecurrentPPO.load(model_path, env=env)
    
    # 3. Run Inference Loop
    obs = env.reset()
    
    # LSTM States: (hidden_state, cell_state)
    lstm_states = None
    
    # Episode start mask
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    print("\nStarting Backtest Loop...")
    print("-" * 50)
    
    history = []
    
    try:
        while True:
            # Predict Action
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_starts, 
                deterministic=True
            )
            
            # Step Environment
            obs, rewards, dones, infos = env.step(action)
            
            # Update tracking
            episode_starts = dones
            info = infos[0]
            
            history.append({
                'portfolio_value': info['portfolio_value'],
                'return': info['return'],
                'drawdown': info['drawdown'],
                'fees': info['fees']
            })
            
            if len(history) % 100 == 0:
                print(f"Step {len(history)}: Portfolio=${info['portfolio_value']:.2f} | Return={info['return']:.4f}")
            
            if dones[0]:
                print("-" * 50)
                print(f"Backtest Finished!")
                print(f"Total Steps: {len(history)}")
                print(f"Final Portfolio Value: ${info['portfolio_value']:.2f}")
                break
                
    except KeyboardInterrupt:
        print("\nBacktest stopped by user.")
        
    # 4. Generate Report/Plots
    df_res = pd.DataFrame(history)
    
    # Plot Portfolio Value
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['portfolio_value'], label='Portfolio Value')
    plt.title('Backtest: Portfolio Value Over Time')
    plt.xlabel('Steps (15-min)')
    plt.ylabel('Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtesting/backtest_portfolio_value.png')
    print("Saved backtesting/backtest_portfolio_value.png")
    
    # Plot Drawdown
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['drawdown'] * 100, label='Drawdown %', color='red')
    plt.title('Backtest: Drawdown Over Time')
    plt.xlabel('Steps (15-min)')
    plt.ylabel('Drawdown (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('backtesting/backtest_drawdown.png')
    print("Saved backtesting/backtest_drawdown.png")
    
    # Save Metrics
    metrics = {
        'final_value': float(df_res['portfolio_value'].iloc[-1]),
        'total_return_pct': float((df_res['portfolio_value'].iloc[-1] - 10000) / 10000 * 100),
        'max_drawdown_pct': float(df_res['drawdown'].max() * 100),
        'total_fees': float(df_res['fees'].sum())
    }
    
    with open('backtesting/backtest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Saved backtesting/backtest_metrics.json")
    
    print("\nMetrics:")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    if not os.path.exists("final_model.zip"):
        print("❌ Model file not found in current directory.")
    else:
        run_backtest()
