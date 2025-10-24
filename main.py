import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from src.genetic_algorithm import optimize_method1, optimize_method2, optimize_method3
from src.backtest import backtest_method1_donchian, backtest_method2_atr, backtest_method3_volume
from src.config import TRAIN_TEST_SPLIT
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_data(df: pd.DataFrame, split_ratio: float = TRAIN_TEST_SPLIT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and testing sets."""
    split_idx = int(len(df) * split_ratio)
    train_data = df.iloc[:split_idx].copy()
    test_data = df.iloc[split_idx:].copy()
    return train_data, test_data

def print_results_summary(method_name: str, train_metrics: Dict, test_metrics: Dict, best_params: List):
    """Print formatted results summary."""
    print("\n" + "="*80)
    print(f"📊 {method_name} - RESULTS SUMMARY")
    print("="*80)
    print(f"Best Parameters: {best_params}")
    print("\n--- TRAINING SET PERFORMANCE ---")
    print(f"Win Rate: {train_metrics['win_rate']*100:.2f}%")
    print(f"Total Return: {train_metrics['total_return_pct']:.2f}%")
    print(f"Profit Factor: {train_metrics['profit_factor']:.2f}")
    print(f"Number of Trades: {train_metrics['num_trades']}")
    print(f"Risk-Reward Ratio: {train_metrics['risk_reward_ratio']:.2f}")
    print(f"Max Drawdown: {train_metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {train_metrics['sharpe_ratio']:.2f}")

    print("\n--- OUT-OF-SAMPLE TEST SET PERFORMANCE ---")
    print(f"Win Rate: {test_metrics['win_rate']*100:.2f}%")
    print(f"Total Return: {test_metrics['total_return_pct']:.2f}%")
    print(f"Profit Factor: {test_metrics['profit_factor']:.2f}")
    print(f"Number of Trades: {test_metrics['num_trades']}")
    print(f"Risk-Reward Ratio: {test_metrics['risk_reward_ratio']:.2f}")
    print(f"Max Drawdown: {test_metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {test_metrics['sharpe_ratio']:.2f}")
    print("="*80)

def main():
    """Main optimization pipeline."""
    print("\n" + "🚀"*40)
    print("FOREX BREAKOUT STRATEGY OPTIMIZER")
    print("EUR/USD 5-Minute Timeframe")
    print("🚀"*40 + "\n")

    # Load data
    logger.info("Loading EUR/USD data...")
    df = pd.read_csv('data/eurusd_5min_clean.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"Loaded {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Split data
    train_data, test_data = split_data(df)
    logger.info(f"Training set: {len(train_data)} bars ({df['timestamp'].min()} to {train_data['timestamp'].max()})")
    logger.info(f"Testing set: {len(test_data)} bars ({test_data['timestamp'].min()} to {df['timestamp'].max()})")

    # Store all results
    all_results = {}

    # ==================== METHOD 1: DONCHIAN CHANNEL ====================
    result1 = optimize_method1(train_data, generations=100)
    best_params1 = result1['best_params']

    # Test on out-of-sample data
    logger.info("Testing Method 1 on out-of-sample data...")
    test_metrics1 = backtest_method1_donchian(test_data, *best_params1)

    print_results_summary("Method 1: Donchian Channel", result1['best_metrics'], test_metrics1, best_params1)

    all_results['method1'] = {
        'name': 'Donchian Channel',
        'best_params': best_params1,
        'train_metrics': result1['best_metrics'],
        'test_metrics': test_metrics1,
        'fitness_history': result1['fitness_history']
    }

    # ==================== METHOD 2: ATR VOLATILITY ====================
    result2 = optimize_method2(train_data, generations=100)
    best_params2 = result2['best_params']

    # Test on out-of-sample data
    logger.info("Testing Method 2 on out-of-sample data...")
    test_metrics2 = backtest_method2_atr(test_data, *best_params2)

    print_results_summary("Method 2: ATR Volatility", result2['best_metrics'], test_metrics2, best_params2)

    all_results['method2'] = {
        'name': 'ATR Volatility',
        'best_params': best_params2,
        'train_metrics': result2['best_metrics'],
        'test_metrics': test_metrics2,
        'fitness_history': result2['fitness_history']
    }

    # ==================== METHOD 3: VOLUME-CONFIRMED ====================
    result3 = optimize_method3(train_data, generations=100)
    best_params3 = result3['best_params']

    # Test on out-of-sample data
    logger.info("Testing Method 3 on out-of-sample data...")
    test_metrics3 = backtest_method3_volume(test_data, *best_params3)

    print_results_summary("Method 3: Volume-Confirmed", result3['best_metrics'], test_metrics3, best_params3)

    all_results['method3'] = {
        'name': 'Volume-Confirmed',
        'best_params': best_params3,
        'train_metrics': result3['best_metrics'],
        'test_metrics': test_metrics3,
        'fitness_history': result3['fitness_history']
    }

    # ==================== FINAL COMPARISON ====================
    print("\n" + "🏆"*40)
    print("FINAL COMPARISON - OUT-OF-SAMPLE TEST RESULTS")
    print("🏆"*40 + "\n")

    comparison = []
    for key, result in all_results.items():
        comparison.append({
            'Method': result['name'],
            'Win Rate': f"{result['test_metrics']['win_rate']*100:.2f}%",
            'Return': f"{result['test_metrics']['total_return_pct']:.2f}%",
            'Profit Factor': f"{result['test_metrics']['profit_factor']:.2f}",
            'Trades': result['test_metrics']['num_trades'],
            'Max DD': f"{result['test_metrics']['max_drawdown_pct']:.2f}%",
            'Sharpe': f"{result['test_metrics']['sharpe_ratio']:.2f}"
        })

    comparison_df = pd.DataFrame(comparison)
    print(comparison_df.to_string(index=False))

    # Determine winner
    winner_idx = max(range(len(comparison)),
                     key=lambda i: all_results[f'method{i+1}']['test_metrics']['win_rate'])
    winner = all_results[f'method{winner_idx+1}']

    print("\n" + "🎯"*40)
    print(f"WINNER: {winner['name']}")
    print(f"Best Parameters: {winner['best_params']}")
    print(f"Test Win Rate: {winner['test_metrics']['win_rate']*100:.2f}%")
    print(f"Test Return: {winner['test_metrics']['total_return_pct']:.2f}%")
    print("🎯"*40 + "\n")

    # Save results
    logger.info("Saving results to results/optimization_results.json...")
    import os
    os.makedirs('results', exist_ok=True)

    # Convert non-serializable objects
    for method_key in all_results:
        for metrics_key in ['train_metrics', 'test_metrics']:
            if 'trades' in all_results[method_key][metrics_key]:
                del all_results[method_key][metrics_key]['trades']
            if 'equity_curve' in all_results[method_key][metrics_key]:
                del all_results[method_key][metrics_key]['equity_curve']

    with open('results/optimization_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info("✅ Optimization complete! Results saved.")

if __name__ == "__main__":
    main()
