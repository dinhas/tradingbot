import os
import sys
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Ensure project root is in path
sys.path.append(os.getcwd())

from RiskLayer.env.risk_env import RiskTradingEnv

def run_risk_backtest(model_path, data_dir="data", output_dir="RiskLayer/backtest/results"):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading Risk Model from {model_path}")
    # Load model with a dummy env
    dummy_env = DummyVecEnv([lambda: RiskTradingEnv(is_training=False, max_rows=1000)])
    model = PPO.load(model_path, env=dummy_env)
    
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    all_results = []
    
    for asset in assets:
        logger.info(f"Testing Risk Model on {asset}...")
        env = RiskTradingEnv(is_training=False, data_dir=data_dir)
        
        obs, _ = env.reset(options={'asset': asset})
        done = False
        truncated = False
        
        trades = []
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Record decision and outcome
            if info['action'] == 'OPEN':
                trades.append({
                    'asset': asset,
                    'timestamp': env.processed_data.index[env.current_step-1],
                    'direction': info['direction'],
                    'sl_mult': info['sl'],
                    'tp_mult': info['tp'],
                    'pnl': info['pnl'],
                    'outcome': info['outcome']
                })
            
            obs = next_obs
            
        if trades:
            df_trades = pd.DataFrame(trades)
            win_rate = (df_trades['pnl'] > 0).mean()
            total_pnl = df_trades['pnl'].sum()
            avg_pnl = df_trades['pnl'].mean()
            
            logger.info(f"DONE {asset}: Trades: {len(df_trades)} | WR: {win_rate:.1%} | Total PnL: {total_pnl:.2f}R")
            all_results.append(df_trades)
        else:
            logger.warning(f"No trades taken for {asset}")

    if all_results:
        final_df = pd.concat(all_results)
        final_df.to_csv(f"{output_dir}/risk_only_backtest.csv", index=False)
        
        overall_wr = (final_df['pnl'] > 0).mean()
        overall_pnl = final_df['pnl'].sum()
        
        logger.info("="*50)
        logger.info(f"OVERALL PERFORMANCE")
        logger.info(f"Total Trades: {len(final_df)}")
        logger.info(f"Win Rate: {overall_wr:.1%}")
        logger.info(f"Total PnL: {overall_pnl:.2f}R")
        logger.info("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    
    run_risk_backtest(args.model)
