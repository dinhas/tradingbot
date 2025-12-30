"""
Backtesting script for RL Trading Bot with stage-aware functionality.

Usage:
    python backtest/backtest.py --model models/checkpoints/stage_1_final.zip --stage 1 --data-dir backtest/data
    python backtest/backtest.py --model models/checkpoints/stage_2_final.zip --stage 2 --episodes 100
    python backtest/backtest.py --model models/checkpoints/stage_3_final.zip --stage 3 --output-dir backtest/results
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path to allow absolute imports
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np

# Add numpy 1.x/2.x compatibility shim for SB3 model loading
if not hasattr(np, "_core"):
    import sys
    sys.modules["numpy._core"] = np.core

import argparse
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from Alpha.src.trading_env import TradingEnv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BacktestMetrics:
    """Calculate and store backtesting metrics per PRD Section 8.1"""
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.timestamps = []
        
    def add_trade(self, trade_info):
        """Add a completed trade"""
        self.trades.append(trade_info)
        
    def add_equity_point(self, timestamp, equity):
        """Add equity snapshot"""
        self.timestamps.append(timestamp)
        self.equity_curve.append(equity)
        
    def calculate_metrics(self):
        """Calculate all PRD metrics"""
        # Total Return (can be calculated without trades)
        initial_equity = self.equity_curve[0] if self.equity_curve else 1000
        final_equity = self.equity_curve[-1] if self.equity_curve else initial_equity
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Max Drawdown
        equity_array = np.array(self.equity_curve)
        if len(equity_array) > 0:
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        if not self.trades:
            logger.warning("No trades to analyze!")
            return {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'final_equity': final_equity,
                'initial_equity': initial_equity,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'avg_rr_ratio': 0,
                'trade_frequency': 0,
                'avg_hold_time_minutes': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'gross_profit': 0,
                'gross_loss': 0
            }
            
        df_trades = pd.DataFrame(self.trades)
        
        # Use net_pnl if available, otherwise calculate it
        if 'net_pnl' in df_trades.columns:
            df_trades['final_pnl'] = df_trades['net_pnl']
        else:
            df_trades['final_pnl'] = df_trades['pnl'] - df_trades['fees']
        
        # Separate winning and losing trades based on net P&L
        winning_trades = df_trades[df_trades['final_pnl'] > 0]
        losing_trades = df_trades[df_trades['final_pnl'] < 0]
        
        gross_profit = winning_trades['final_pnl'].sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades['final_pnl'].sum()) if len(losing_trades) > 0 else 0
        
        # PRIMARY METRIC: Profit Factor
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe Ratio (assuming 252 trading days, 5min candles)
        if len(self.equity_curve) > 1:
            returns = np.diff(equity_array) / equity_array[:-1]
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 288) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
            
        # Win Rate
        win_rate = len(winning_trades) / len(df_trades) if len(df_trades) > 0 else 0
        
        # Average RR Ratio
        if 'rr_ratio' in df_trades.columns:
            avg_rr_ratio = df_trades['rr_ratio'].mean()
        else:
            avg_rr_ratio = 0
            
        # Trade Frequency (trades per day)
        if self.timestamps:
            time_span_days = (self.timestamps[-1] - self.timestamps[0]).total_seconds() / 86400
            trade_frequency = len(df_trades) / time_span_days if time_span_days > 0 else 0
        else:
            trade_frequency = 0
            
        # Average Hold Time
        if 'hold_time' in df_trades.columns:
            avg_hold_time = df_trades['hold_time'].mean()
        else:
            avg_hold_time = 0
            
        metrics = {
            'profit_factor': profit_factor,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_rr_ratio': avg_rr_ratio,
            'trade_frequency': trade_frequency,
            'avg_hold_time_minutes': avg_hold_time,
            'total_trades': len(df_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'final_equity': final_equity,
            'initial_equity': initial_equity
        }
        
        return metrics
        
    def get_per_asset_metrics(self):
        """Calculate metrics per asset"""
        if not self.trades:
            return {}
            
        df_trades = pd.DataFrame(self.trades)
        
        # Use net_pnl if available
        if 'net_pnl' in df_trades.columns:
            df_trades['final_pnl'] = df_trades['net_pnl']
        else:
            df_trades['final_pnl'] = df_trades['pnl'] - df_trades['fees']
        
        per_asset = {}
        
        for asset in df_trades['asset'].unique():
            asset_trades = df_trades[df_trades['asset'] == asset]
            winning = asset_trades[asset_trades['final_pnl'] > 0]
            losing = asset_trades[asset_trades['final_pnl'] < 0]
            
            gross_profit = winning['final_pnl'].sum() if len(winning) > 0 else 0
            gross_loss = abs(losing['final_pnl'].sum()) if len(losing) > 0 else 0
            
            per_asset[asset] = {
                'num_trades': len(asset_trades),
                'win_rate': len(winning) / len(asset_trades) if len(asset_trades) > 0 else 0,
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
                'total_pnl': asset_trades['final_pnl'].sum()
            }
            
        return per_asset


class FullSystemMetrics(BacktestMetrics):
    """
    Extends BacktestMetrics to handle TradeGuard specifics and Shadow Portfolio.
    """
    def __init__(self):
        super().__init__()
        self.blocked_trades = []
        self.shadow_equity_curve = []
        self.shadow_timestamps = []

    def add_blocked_trade(self, trade_info):
        """Add a trade that was blocked by TradeGuard"""
        self.blocked_trades.append(trade_info)

    def add_shadow_equity_point(self, timestamp, equity):
        """Add equity snapshot for the Shadow Portfolio (Baseline)"""
        self.shadow_timestamps.append(timestamp)
        self.shadow_equity_curve.append(equity)

    def calculate_tradeguard_metrics(self):
        """Calculate TradeGuard-specific performance metrics"""
        if not self.blocked_trades and not self.trades:
            return {}

        total_signals = len(self.trades) + len(self.blocked_trades)
        approval_rate = len(self.trades) / total_signals if total_signals > 0 else 0

        # Block Accuracy: How many blocked trades would have been losses?
        good_blocks = sum(1 for t in self.blocked_trades if t['theoretical_pnl'] <= 0)
        bad_blocks = sum(1 for t in self.blocked_trades if t['theoretical_pnl'] > 0)
        block_accuracy = good_blocks / len(self.blocked_trades) if self.blocked_trades else 0

        # Net Value-Add: (Sum of avoided losses) - (Sum of missed profits)
        avoided_losses = sum(abs(t['theoretical_pnl']) for t in self.blocked_trades if t['theoretical_pnl'] <= 0)
        missed_profits = sum(t['theoretical_pnl'] for t in self.blocked_trades if t['theoretical_pnl'] > 0)
        net_value_add = avoided_losses - missed_profits

        return {
            'approval_rate': approval_rate,
            'block_accuracy': block_accuracy,
            'net_value_add_pct': net_value_add,
            'total_blocked': len(self.blocked_trades),
            'good_blocks': good_blocks,
            'bad_blocks': bad_blocks,
            'avoided_losses_pct': avoided_losses,
            'missed_profits_pct': missed_profits
        }

    def calculate_metrics(self):
        """Calculate base metrics plus TradeGuard and Shadow Portfolio metrics"""
        # Call the base class method to get standard metrics
        metrics = super().calculate_metrics()
        
        # TradeGuard metrics
        tg_metrics = self.calculate_tradeguard_metrics()
        metrics['tradeguard'] = tg_metrics
        
        # Shadow Portfolio (Baseline) Return
        if self.shadow_equity_curve:
            initial_shadow = self.shadow_equity_curve[0]
            final_shadow = self.shadow_equity_curve[-1]
            shadow_return = (final_shadow - initial_shadow) / initial_shadow
            metrics['baseline_return'] = shadow_return
            
            # Net Value-Add over Baseline
            metrics['net_value_add_vs_baseline'] = metrics.get('total_return', 0) - shadow_return
            
        return metrics


def make_backtest_env(data_dir, stage):
    """Create environment for backtesting"""
    def _init():
        env = TradingEnv(data_dir=data_dir, stage=stage, is_training=False)
        return env
    return _init


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def run_backtest(args):
    """Main backtesting function"""
    project_root = Path(__file__).resolve().parent.parent.parent
    model_path = project_root / args.model
    data_dir_path = project_root / args.data_dir
    output_dir_path = project_root / args.output_dir

    logger.info("Starting Alpha Model Backtest")
    logger.info(f"Model: {model_path}")
    logger.info(f"Data directory: {data_dir_path}")
    
    # Create output directory
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info("Loading model...")
    # Get temporary env to access metadata
    temp_env = TradingEnv(data_dir=data_dir_path, is_training=False)
    assets_to_test = temp_env.assets if args.asset == "all" else [args.asset]
    
    env = DummyVecEnv([lambda: TradingEnv(data_dir=data_dir_path, is_training=False)])
    
    # Load VecNormalize stats if available
    vecnorm_path = str(model_path).replace('.zip', '_vecnormalize.pkl')
    if os.path.exists(vecnorm_path):
        logger.info(f"Loading VecNormalize stats from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    
    model = PPO.load(model_path, env=env)
    logger.info("Model loaded successfully")
    
    # Initialize metrics tracker
    metrics_tracker = BacktestMetrics()
    
    # Run episodes for each asset
    for asset in assets_to_test:
        logger.info(f"\nTesting Asset: {asset}")
        logger.info("-" * 20)
        
        for episode in range(args.episodes):
            # Force environment to use the specific asset
            # Reach into the VecEnv to set the asset
            if hasattr(env, 'venv'):
                inner_env = env.venv.envs[0]
            else:
                inner_env = env.envs[0]
            
            inner_env.current_asset = asset
            obs = env.reset()
            done = False
            episode_reward = 0
            step_count = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
                step_count += 1
                
                if info and len(info) > 0:
                    env_info = info[0]
                    if 'trades' in env_info:
                        for trade in env_info['trades']:
                            metrics_tracker.add_trade(trade)
                    if 'equity' in env_info and 'timestamp' in env_info:
                        metrics_tracker.add_equity_point(env_info['timestamp'], env_info['equity'])
            
            final_equity = env_info.get('equity', 0) if info and len(info) > 0 else 0
            logger.info(f"[{asset}] Episode {episode + 1}/{args.episodes} complete. Reward: {episode_reward:.2f}, Final Equity: ${final_equity:.2f}")
    
    env.close()
    
    # Calculate metrics
    logger.info("\n" + "="*60)
    logger.info("CALCULATING METRICS")
    logger.info("="*60)
    
    metrics = metrics_tracker.calculate_metrics()
    
    # Print metrics
    logger.info(f"\n{'BACKTEST RESULTS':^60}")
    logger.info("="*60)
    logger.info(f"{'PRIMARY METRIC - Profit Factor:':<40} {metrics.get('profit_factor', 0):.3f}")
    logger.info(f"{'Total Return:':<40} {metrics.get('total_return', 0):.2%}")
    logger.info(f"{'Sharpe Ratio:':<40} {metrics.get('sharpe_ratio', 0):.3f}")
    logger.info(f"{'Max Drawdown:':<40} {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"{'Win Rate:':<40} {metrics.get('win_rate', 0):.2%}")
    logger.info(f"{'Average RR Ratio:':<40} {metrics.get('avg_rr_ratio', 0):.2f}")
    logger.info(f"{'Trade Frequency (per day):':<40} {metrics.get('trade_frequency', 0):.2f}")
    logger.info(f"{'Average Hold Time (minutes):':<40} {metrics.get('avg_hold_time_minutes', 0):.1f}")
    logger.info(f"{'Total Trades:':<40} {metrics.get('total_trades', 0)}")
    logger.info(f"{'Winning Trades:':<40} {metrics.get('winning_trades', 0)}")
    logger.info(f"{'Losing Trades:':<40} {metrics.get('losing_trades', 0)}")
    logger.info("="*60)
    
    # Check PRD success criteria (Section 8.1)
    logger.info(f"\n{'PRD SUCCESS CRITERIA CHECK':^60}")
    logger.info("="*60)
    pf_pass = metrics.get('profit_factor', 0) > 1.3
    dd_pass = metrics.get('max_drawdown', 0) > -0.20
    sr_pass = metrics.get('sharpe_ratio', 0) > 1.0
    wr_pass = metrics.get('win_rate', 0) > 0.45
    
    logger.info(f"{'Profit Factor > 1.3:':<40} {'✅ PASS' if pf_pass else '❌ FAIL'}")
    logger.info(f"{'Max Drawdown < 20%:':<40} {'✅ PASS' if dd_pass else '❌ FAIL'}")
    logger.info(f"{'Sharpe Ratio > 1.0:':<40} {'✅ PASS' if sr_pass else '❌ FAIL'}")
    logger.info(f"{'Win Rate > 45%:':<40} {'✅ PASS' if wr_pass else '❌ FAIL'}")
    logger.info("="*60)
    
    all_pass = pf_pass and dd_pass and sr_pass and wr_pass
    logger.info(f"\n{'OVERALL: ' + ('✅ ALL CRITERIA MET' if all_pass else '❌ SOME CRITERIA NOT MET'):^60}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save metrics JSON
    metrics_file = output_dir_path / f"metrics_alpha_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Saved metrics to {metrics_file}")
    
    # 2. Save trade log
    if metrics_tracker.trades:
        trades_file = output_dir_path / f"trades_alpha_{timestamp}.csv"
        pd.DataFrame(metrics_tracker.trades).to_csv(trades_file, index=False)
        logger.info(f"Saved trade log to {trades_file}")
    
    # 3. Save per-asset performance
    per_asset = metrics_tracker.get_per_asset_metrics()
    if per_asset:
        asset_file = output_dir_path / f"asset_breakdown_stage{args.stage}_{timestamp}.csv"
        pd.DataFrame(per_asset).T.to_csv(asset_file)
        logger.info(f"Saved per-asset breakdown to {asset_file}")
    
    # 4. Generate all visualizations
    if metrics_tracker.equity_curve and metrics_tracker.trades:
        logger.info("\nGenerating comprehensive charts...")
        generate_all_charts(metrics_tracker, per_asset, args.stage, output_dir_path, timestamp)
    
    logger.info("\nBacktest complete!")
    return metrics


def generate_all_charts(metrics_tracker, per_asset, stage, output_dir, timestamp):
    """Generate comprehensive visualization suite"""
    
    df_trades = pd.DataFrame(metrics_tracker.trades)
    equity = np.array(metrics_tracker.equity_curve)
    times = metrics_tracker.timestamps
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # ============ Chart 1: Equity Curve with Drawdown ============
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, equity, linewidth=2.5, color='#2E86AB', label='Equity', zorder=3)
    ax1.axhline(y=equity[0], color='gray', linestyle='--', alpha=0.5, label='Starting Capital', linewidth=1.5)
    ax1.fill_between(times, equity[0], equity, where=(equity >= equity[0]), alpha=0.2, color='green', label='Profit Zone')
    ax1.fill_between(times, equity[0], equity, where=(equity < equity[0]), alpha=0.2, color='red', label='Loss Zone')
    ax1.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
    # Determine year for title
    year = times[0].year if times else "2025"
    ax1.set_title(f'Stage {stage} - Equity Curve & Drawdown ({year})', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Drawdown subplot
    ax1_dd = fig.add_subplot(gs[1, :2], sharex=ax1)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    ax1_dd.fill_between(times, drawdown, 0, where=(drawdown < 0), color='#A23B72', alpha=0.6, label='Drawdown')
    ax1_dd.axhline(y=-20, color='red', linestyle='--', alpha=0.7, label='PRD Limit (-20%)', linewidth=2)
    ax1_dd.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
    ax1_dd.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1_dd.legend(loc='lower left', fontsize=9)
    ax1_dd.grid(True, alpha=0.3)
    ax1_dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1_dd.xaxis.get_majorticklabels(), rotation=45)
    
    # ============ Chart 2: P&L Distribution ============
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Use net_pnl if available
    if 'net_pnl' in df_trades.columns:
        pnl_values = df_trades['net_pnl'].values
    else:
        pnl_values = (df_trades['pnl'] - df_trades['fees']).values
    
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]
    
    ax2.hist(wins, bins=20, alpha=0.7, color='green', label=f'Wins ({len(wins)})', edgecolor='black')
    ax2.hist(losses, bins=20, alpha=0.7, color='red', label=f'Losses ({len(losses)})', edgecolor='black')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Net P&L ($)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax2.set_title('Net P&L Distribution (After Fees)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ============ Chart 3: Cumulative P&L by Asset ============
    ax3 = fig.add_subplot(gs[1, 2])
    if 'asset' in df_trades.columns:
        for asset in df_trades['asset'].unique():
            asset_trades = df_trades[df_trades['asset'] == asset].sort_values('timestamp' if 'timestamp' in df_trades.columns else df_trades.index)
            cumulative_pnl = asset_trades['pnl'].cumsum()
            ax3.plot(range(len(cumulative_pnl)), cumulative_pnl, marker='o', markersize=3, label=asset, linewidth=2)
        
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Trade Number', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Cumulative P&L ($)', fontsize=10, fontweight='bold')
        ax3.set_title('Cumulative P&L by Asset', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8, loc='best')
        ax3.grid(True, alpha=0.3)
    
    # ============ Chart 4: Per-Asset Performance Bars ============
    ax4 = fig.add_subplot(gs[2, :2])
    if per_asset:
        assets = list(per_asset.keys())
        pnls = [per_asset[a]['total_pnl'] for a in assets]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        
        bars = ax4.barh(assets, pnls, color=colors, alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax4.set_xlabel('Total P&L ($)', fontsize=11, fontweight='bold')
        ax4.set_title('Per-Asset Total P&L', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, pnl) in enumerate(zip(bars, pnls)):
            ax4.text(pnl, i, f' ${pnl:.2f}', va='center', ha='left' if pnl > 0 else 'right', fontweight='bold', fontsize=9)
    
    # ============ Chart 5: Win Rate by Asset ============
    ax5 = fig.add_subplot(gs[2, 2])
    if per_asset:
        assets = list(per_asset.keys())
        win_rates = [per_asset[a]['win_rate'] * 100 for a in assets]
        colors_wr = ['#2E86AB' if wr > 50 else '#F18F01' for wr in win_rates]
        
        bars_wr = ax5.bar(range(len(assets)), win_rates, color=colors_wr, alpha=0.7, edgecolor='black')
        ax5.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, label='50% Breakeven')
        ax5.axhline(y=45, color='green', linestyle='--', linewidth=1.5, label='PRD Target (45%)')
        ax5.set_xticks(range(len(assets)))
        ax5.set_xticklabels(assets, rotation=45, ha='right')
        ax5.set_ylabel('Win Rate (%)', fontsize=10, fontweight='bold')
        ax5.set_title('Win Rate by Asset', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, wr in zip(bars_wr, win_rates):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height, f'{wr:.1f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # ============ Chart 6: Trade Timeline (Entry/Exit Points) ============
    ax6 = fig.add_subplot(gs[3, :])
    if 'entry_price' in df_trades.columns and 'exit_price' in df_trades.columns:
        # Plot trades over time
        trade_times = range(len(df_trades))
        entry_prices = df_trades['entry_price'].values
        exit_prices = df_trades['exit_price'].values
        pnls = df_trades['pnl'].values
        
        # Color code by profit/loss
        colors_timeline = ['green' if p > 0 else 'red' for p in pnls]
        
        for i, (entry, exit, color) in enumerate(zip(entry_prices, exit_prices, colors_timeline)):
            ax6.plot([i, i], [entry, exit], color=color, alpha=0.6, linewidth=2)
            ax6.scatter(i, entry, color='blue', s=30, zorder=3, alpha=0.7)
            ax6.scatter(i, exit, color=color, s=30, zorder=3, alpha=0.7)
        
        ax6.set_xlabel('Trade Number', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Price', fontsize=11, fontweight='bold')
        ax6.set_title('Trade Timeline (Entry → Exit)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Entry'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Profitable Exit'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Loss Exit')
        ]
        ax6.legend(handles=legend_elements, fontsize=9, loc='best')
    
    plt.suptitle(f'Stage {stage} Backtest - Comprehensive Analysis ({year})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save comprehensive chart
    chart_file = output_dir / f"comprehensive_analysis_stage{stage}_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved comprehensive analysis chart to {chart_file}")
    
    # Also create individual detailed trade log chart
    create_detailed_trade_chart(df_trades, stage, output_dir, timestamp)


def create_detailed_trade_chart(df_trades, stage, output_dir, timestamp):
    """Create a detailed chart showing individual trade details"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Stage {stage} - Detailed Trade Analysis', fontsize=14, fontweight='bold')
    
    # Chart 1: Trade Size Distribution
    ax1 = axes[0, 0]
    if 'size' in df_trades.columns:
        ax1.hist(df_trades['size'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Position Size', fontsize=10, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax1.set_title('Position Size Distribution', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Chart 2: Hold Time Distribution
    ax2 = axes[0, 1]
    if 'hold_time' in df_trades.columns:
        ax2.hist(df_trades['hold_time'], bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Hold Time (minutes)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax2.set_title('Hold Time Distribution', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: Portfolio Impact (Equity Before vs After)
    ax3 = axes[1, 0]
    if 'equity_before' in df_trades.columns and 'equity_after' in df_trades.columns:
        trade_nums = range(len(df_trades))
        ax3.scatter(trade_nums, df_trades['equity_before'], alpha=0.6, s=40, color='blue', label='Before Trade')
        ax3.scatter(trade_nums, df_trades['equity_after'], alpha=0.6, s=40, color='green', label='After Trade')
        ax3.plot(trade_nums, df_trades['equity_after'], alpha=0.3, color='green', linewidth=1)
        ax3.set_xlabel('Trade Number', fontsize=10, fontweight='bold')
        ax3.set_ylabel('Portfolio Equity ($)', fontsize=10, fontweight='bold')
        ax3.set_title('Portfolio Effect per Trade', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    
    # Chart 4: Win/Loss Streaks
    ax4 = axes[1, 1]
    pnls = df_trades['pnl'].values
    win_loss = np.where(pnls > 0, 1, -1)
    
    # Calculate streaks
    streaks = []
    current_streak = 1
    for i in range(1, len(win_loss)):
        if win_loss[i] == win_loss[i-1]:
            current_streak += 1
        else:
            streaks.append(current_streak * win_loss[i-1])
            current_streak = 1
    streaks.append(current_streak * win_loss[-1])
    
    colors_streak = ['green' if s > 0 else 'red' for s in streaks]
    ax4.bar(range(len(streaks)), streaks, color=colors_streak, alpha=0.7, edgecolor='black')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax4.set_xlabel('Streak Number', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Streak Length', fontsize=10, fontweight='bold')
    ax4.set_title('Win/Loss Streaks', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    detail_file = output_dir / f"trade_details_stage{stage}_{timestamp}.png"
    plt.savefig(detail_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Saved detailed trade analysis to {detail_file}")


def generate_full_system_charts(metrics_tracker, per_asset, stage, output_dir, timestamp):
    """
    Generate comprehensive visualization suite for the Three-Layer System.
    Includes Shadow Portfolio comparison and TradeGuard analysis.
    """
    # 1. Main Comparative Analysis Chart
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    times = metrics_tracker.timestamps
    equity = np.array(metrics_tracker.equity_curve)
    shadow_equity = np.array(metrics_tracker.shadow_equity_curve)
    
    # --- Plot 1: Equity Curve Comparison (Full System vs. Baseline) ---
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(times, equity, linewidth=2.5, color='#2E86AB', label='Full System (With TradeGuard)', zorder=3)
    if len(shadow_equity) == len(times):
        ax1.plot(times, shadow_equity, linewidth=2.0, color='#A23B72', alpha=0.7, label='Baseline (Shadow Portfolio)', linestyle='--')
    
    ax1.set_ylabel('Equity ($)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Equity Curve Comparison: Full System vs. Baseline', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # --- Plot 2: TradeGuard Probability Distribution (Approved vs. Blocked) ---
    ax2 = fig.add_subplot(gs[0, 2])
    approved_probs = [t.get('prob', 0.5) for t in metrics_tracker.trades if 'prob' in t]
    blocked_probs = [t.get('prob', 0.5) for t in metrics_tracker.blocked_trades]
    
    if approved_probs:
        ax2.hist(approved_probs, bins=20, alpha=0.6, color='green', label='Approved', edgecolor='black')
    if blocked_probs:
        ax2.hist(blocked_probs, bins=20, alpha=0.6, color='red', label='Blocked', edgecolor='black')
    
    # Threshold line
    threshold = 0.5
    if metrics_tracker.blocked_trades:
        threshold = metrics_tracker.blocked_trades[0].get('threshold', 0.5)
    
    ax2.axvline(x=threshold, color='black', linestyle='--', label='Threshold')
    ax2.set_xlabel('TradeGuard Probability', fontsize=10, fontweight='bold')
    ax2.set_title('TradeGuard Probability Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Block Quality Timeline ---
    ax3 = fig.add_subplot(gs[1, :])
    if metrics_tracker.blocked_trades:
        bt_df = pd.DataFrame(metrics_tracker.blocked_trades)
        # Good blocks (avoided loss) = Green, Bad blocks (missed profit) = Red
        colors = ['#27AE60' if p <= 0 else '#E74C3C' for p in bt_df['theoretical_pnl']]
        sizes = [abs(p) * 5000 for p in bt_df['theoretical_pnl']]
        ax3.scatter(bt_df['timestamp'], bt_df['theoretical_pnl'], s=sizes, c=colors, alpha=0.6, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_ylabel('Avoided PnL (Ratio)', fontsize=10, fontweight='bold')
        ax3.set_title('TradeGuard Block Quality Timeline (Bubble size = magnitude)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # --- Plot 4: Per-Asset Approval Rates ---
    ax4 = fig.add_subplot(gs[2, 0])
    approved_asset_data = [{'asset': t['asset'], 'status': 'approved'} for t in metrics_tracker.trades]
    blocked_asset_data = [{'asset': t['asset'], 'status': 'blocked'} for t in metrics_tracker.blocked_trades]
    
    if approved_asset_data or blocked_asset_data:
        all_signals = pd.concat([pd.DataFrame(approved_asset_data), pd.DataFrame(blocked_asset_data)])
        if not all_signals.empty:
            counts = all_signals.groupby(['asset', 'status']).size().unstack(fill_value=0)
            if 'approved' in counts:
                total_sigs = counts.sum(axis=1)
                rates = counts['approved'] / total_sigs
                rates.plot(kind='bar', ax=ax4, color='#3498DB', edgecolor='black', alpha=0.7)
                ax4.set_title('Approval Rate by Asset', fontsize=11, fontweight='bold')
                ax4.set_ylim(0, 1.1)
                ax4.grid(True, alpha=0.3, axis='y')

    # --- Plot 5: Time-of-Day Decision Analysis ---
    ax5 = fig.add_subplot(gs[2, 1:])
    if approved_asset_data or blocked_asset_data:
        all_signals['hour'] = pd.to_datetime([t['timestamp'] for t in metrics_tracker.trades] + 
                                           [t['timestamp'] for t in metrics_tracker.blocked_trades]).hour
        hourly_stats = all_signals.groupby(['hour', 'status']).size().unstack(fill_value=0)
        hourly_stats.plot(kind='bar', stacked=True, ax=ax5, color=['#2ECC71', '#E74C3C'], alpha=0.7, edgecolor='black')
        ax5.set_title('Decision Volume by Hour (Approved vs. Blocked)', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Three-Layer System Analysis - Stage {stage}', fontsize=16, fontweight='bold', y=0.99)
    chart_file = output_dir / f"full_system_analysis_stage{stage}_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Detail Chart for TradeGuard Metrics
    create_tradeguard_performance_charts(metrics_tracker, stage, output_dir, timestamp)


def create_tradeguard_performance_charts(metrics_tracker, stage, output_dir, timestamp):
    """Detailed Performance metrics for TradeGuard layer"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Net Value Add vs. Probability
    ax1 = axes[0]
    if metrics_tracker.blocked_trades:
        bt_df = pd.DataFrame(metrics_tracker.blocked_trades)
        ax1.scatter(bt_df['prob'], bt_df['theoretical_pnl'], alpha=0.5)
        ax1.axhline(y=0, color='black', linestyle='--')
        ax1.set_xlabel('TradeGuard Probability')
        ax1.set_ylabel('Theoretical PnL')
        ax1.set_title('Block Quality vs. Confidence Score')
        ax1.grid(True, alpha=0.3)
    
    # Block Accuracy Pie
    ax2 = axes[1]
    tg_metrics = metrics_tracker.calculate_tradeguard_metrics()
    if tg_metrics and tg_metrics.get('total_blocked', 0) > 0:
        labels = [f"Good Blocks ({tg_metrics['good_blocks']})", f"Bad Blocks ({tg_metrics['bad_blocks']})"]
        sizes = [tg_metrics['good_blocks'], tg_metrics['bad_blocks']]
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#2ECC71', '#E74C3C'], startangle=90, wedgeprops={'edgecolor': 'white'})
        ax2.set_title(f"TradeGuard Filtering Accuracy (Total Blocked: {tg_metrics['total_blocked']})")
    
    plt.tight_layout()
    chart_file = output_dir / f"tradeguard_performance_stage3_{timestamp}.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest RL Trading Bot")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.zip file) relative to project root")
    parser.add_argument("--data-dir", type=str, default="Alpha/backtest/data",
                        help="Path to backtest data directory relative to project root")
    parser.add_argument("--output-dir", type=str, default="Alpha/backtest/results",
                        help="Path to save results relative to project root")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run per asset")
    parser.add_argument("--stage", type=int, default=1,
                        help="Training stage (1, 2, or 3) for reporting")
    parser.add_argument("--asset", type=str, default="all",
                        help="Specific asset to test (e.g., EURUSD) or 'all' to test the full basket")
    
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent.parent
    model_path = project_root / args.model
    data_dir_path = project_root / args.data_dir

    # Validate model exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Validate data directory exists and is not empty
    if not data_dir_path.exists():
        logger.error(f"Data directory not found: {data_dir_path}")
        logger.info("Please run: python -m Alpha.backtest.data_fetcher_backtest")
        sys.exit(1)
        
    # Check for parquet files
    parquet_files = list(data_dir_path.glob('*.parquet'))
    if not parquet_files:
        logger.error(f"No .parquet files found in {data_dir_path}")
        logger.info(f"Target directory '{data_dir_path}' is empty or contains no data.")
        logger.info("Please run: python -m Alpha.backtest.data_fetcher_backtest")
        sys.exit(1)
    
    run_backtest(args)
