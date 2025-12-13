"""
Stage Comparison Utility

Compares backtesting results across different curriculum stages.

Usage:
    python backtest/compare_stages.py --results-dir backtest/results
"""

import os
import sys
import argparse
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_stage_results(results_dir, stage):
    """Load the most recent results for a given stage"""
    metrics_files = list(Path(results_dir).glob(f"metrics_stage{stage}_*.json"))
    
    if not metrics_files:
        logger.warning(f"No results found for stage {stage}")
        return None
    
    # Get most recent file
    latest_file = max(metrics_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"Loaded stage {stage} results from {latest_file.name}")
    return metrics


def compare_stages(results_dir):
    """Compare performance across all stages"""
    logger.info("Loading results for all stages...")
    
    results = {}
    for stage in [1, 2, 3]:
        metrics = load_stage_results(results_dir, stage)
        if metrics:
            results[f"Stage {stage}"] = metrics
    
    if not results:
        logger.error("No results found to compare!")
        return
    
    # Create comparison DataFrame
    comparison_metrics = [
        'profit_factor',
        'total_return',
        'sharpe_ratio',
        'max_drawdown',
        'win_rate',
        'avg_rr_ratio',
        'trade_frequency',
        'total_trades'
    ]
    
    df = pd.DataFrame(results).T[comparison_metrics]
    
    # Print comparison table
    logger.info("\n" + "="*80)
    logger.info(f"{'STAGE COMPARISON':^80}")
    logger.info("="*80)
    print(df.to_string())
    logger.info("="*80)
    
    # Identify best stage for each metric
    logger.info(f"\n{'BEST PERFORMING STAGE PER METRIC':^80}")
    logger.info("="*80)
    
    for metric in comparison_metrics:
        if metric == 'max_drawdown':
            # For drawdown, higher (closer to 0) is better
            best_stage = df[metric].idxmax()
        else:
            best_stage = df[metric].idxmax()
        
        logger.info(f"{metric:<30} {best_stage}")
    
    logger.info("="*80)
    
    # Create comparison visualizations
    create_comparison_plots(df, results_dir)
    
    # Save comparison table
    output_file = os.path.join(results_dir, "stage_comparison.csv")
    df.to_csv(output_file)
    logger.info(f"\nSaved comparison table to {output_file}")
    
    return df


def create_comparison_plots(df, results_dir):
    """Create visualization comparing stages"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Curriculum Stage Performance Comparison', fontsize=16, fontweight='bold')
    
    stages = df.index.tolist()
    x = np.arange(len(stages))
    width = 0.6
    
    # Plot 1: Profit Factor
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, df['profit_factor'], width, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.axhline(y=1.3, color='green', linestyle='--', label='PRD Target (1.3)', linewidth=2)
    ax1.set_ylabel('Profit Factor', fontweight='bold')
    ax1.set_title('Profit Factor (PRIMARY METRIC)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Sharpe Ratio
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, df['sharpe_ratio'], width, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax2.axhline(y=1.0, color='green', linestyle='--', label='PRD Target (1.0)', linewidth=2)
    ax2.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax2.set_title('Sharpe Ratio', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Max Drawdown
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, df['max_drawdown'] * 100, width, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax3.axhline(y=-20, color='red', linestyle='--', label='PRD Limit (-20%)', linewidth=2)
    ax3.set_ylabel('Max Drawdown (%)', fontweight='bold')
    ax3.set_title('Maximum Drawdown', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stages)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='top' if height < 0 else 'bottom', fontweight='bold')
    
    # Plot 4: Win Rate
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, df['win_rate'] * 100, width, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax4.axhline(y=45, color='green', linestyle='--', label='PRD Target (45%)', linewidth=2)
    ax4.set_ylabel('Win Rate (%)', fontweight='bold')
    ax4.set_title('Win Rate', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(stages)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(results_dir, "stage_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare backtesting results across stages")
    parser.add_argument("--results-dir", type=str, default="backtest/results",
                       help="Directory containing backtest results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        logger.error(f"Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    compare_stages(args.results_dir)
