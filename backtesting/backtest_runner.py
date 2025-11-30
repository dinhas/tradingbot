import gymnasium as gym
import numpy as np
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from trading_env import TradingEnv
import os
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Asset mapping for readable logs
ASSET_MAPPING = {
    0: 'BTC',
    1: 'ETH',
    2: 'SOL',
    3: 'EUR',
    4: 'GBP',
    5: 'JPY',
    6: 'CASH'
}

def run_backtest(model_path="best_model.zip", stats_path="final_model_vecnormalize.pkl", data_dir="data/"):
    """
    Runs the trained model on the TradingEnv using backtest data.
    Logs all trades and creates comprehensive visualizations.
    """
    print(f"Loading model from {model_path}...")
    
    # 1. Load the Environment
    env = DummyVecEnv([lambda: TradingEnv(data_dir=data_dir, volatility_file="data/volatility_baseline.json")])
    
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
    
    # Get access to the underlying environment for detailed logging
    base_env = env.envs[0]
    
    # LSTM States
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    
    print("\nStarting Backtest Loop...")
    print("=" * 80)
    
    # History tracking
    history = []
    trade_log = []
    
    # Previous positions for trade detection
    prev_holdings = {asset: 0.0 for asset in ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']}
    
    try:
        step_count = 0
        while True:
            # Get current timestamp from environment
            current_timestamp = base_env.timestamps[base_env.current_step]
            current_year = current_timestamp.year
            
            # Get current market prices
            current_data = base_env.data.iloc[base_env.current_step]
            prices = {asset: current_data[f"{asset}_close"] for asset in ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']}
            
            # Store pre-action state
            pre_portfolio_value = base_env.portfolio_value
            pre_cash = base_env.cash
            
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
            
            # Get post-action state
            post_holdings = base_env.holdings.copy()
            post_portfolio_value = info['portfolio_value']
            post_cash = base_env.cash
            
            # Detect and log trades
            trade_events = info.get('trade_events', {})
            
            for asset in ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']:
                prev_units = prev_holdings[asset]
                current_units = post_holdings[asset]
                
                if abs(current_units - prev_units) > 1e-6:  # Trade occurred
                    trade_type = "BUY" if current_units > prev_units else "SELL"
                    units_traded = abs(current_units - prev_units)
                    trade_value = units_traded * prices[asset]
                    
                    # Check for specific exit reason (SL/TP)
                    exit_reason = None
                    exit_pnl = 0.0
                    
                    if asset in trade_events:
                        event = trade_events[asset]
                        if event['reason'] == 'SL':
                            exit_reason = "Stop Loss Hit"
                        elif event['reason'] == 'TP':
                            exit_reason = "Take Profit Hit"
                        exit_pnl = event['pnl']
                    
                    trade_entry = {
                        'timestamp': current_timestamp,
                        'step': step_count,
                        'asset': asset,
                        'type': trade_type,
                        'units': units_traded,
                        'price': prices[asset],
                        'value_usd': trade_value,
                        'portfolio_value_before': pre_portfolio_value,
                        'portfolio_value_after': post_portfolio_value,
                        'cash_before': pre_cash,
                        'cash_after': post_cash,
                        'fees': info['fees'],
                        'sl_mult': info.get('sl_multiplier', 0.0),
                        'tp_mult': info.get('tp_multiplier', 0.0),
                        'year': current_year,
                        'exit_reason': exit_reason,
                        'exit_pnl': exit_pnl
                    }
                    trade_log.append(trade_entry)
            
            # Update previous holdings
            prev_holdings = post_holdings.copy()
            
            # Store history
            history.append({
                'timestamp': current_timestamp,
                'portfolio_value': post_portfolio_value,
                'return': info['return'],
                'drawdown': info['drawdown'],
                'fees': info['fees'],
                'cash': post_cash,
                **{f'{asset}_holdings': post_holdings[asset] for asset in ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']},
                **{f'{asset}_price': prices[asset] for asset in ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']}
            })
            
            step_count += 1
            
            # Progress logging
            if step_count % 500 == 0:
                print(f"Step {step_count}: {current_timestamp} | Portfolio=${post_portfolio_value:.2f} | "
                      f"Return={info['return']:.4f} | Trades={len(trade_log)}")
            
            if dones[0]:
                print("=" * 80)
                print(f"✅ Backtest Finished!")
                print(f"Total Steps: {step_count}")
                print(f"Total Trades: {len(trade_log)}")
                print(f"Final Portfolio Value: ${post_portfolio_value:.2f}")
                print(f"Total Return: {((post_portfolio_value - 10000) / 10000 * 100):.2f}%")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Backtest stopped by user.")
    
    # ========== SAVE TRADE LOG ==========
    print("\n" + "=" * 80)
    print("💾 Saving Trade Logs...")
    
    # Filter trades for 2025
    trades_2025 = [t for t in trade_log if t['year'] == 2025]
    
    # Save all 2025 trades to trades.txt
    trades_txt_path = os.path.join(data_dir, 'trades.txt')
    with open(trades_txt_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("TRADING BACKTEST - 2025 TRADE LOG\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Total Trades in 2025: {len(trades_2025)}\n")
        f.write("=" * 100 + "\n\n")
        
        if len(trades_2025) > 0:
            for idx, trade in enumerate(trades_2025, 1):
                f.write(f"Trade #{idx}\n")
                f.write(f"  Timestamp:    {trade['timestamp']}\n")
                f.write(f"  Asset:        {trade['asset']}\n")
                f.write(f"  Type:         {trade['type']}\n")
                f.write(f"  Units:        {trade['units']:.6f}\n")
                f.write(f"  Price:        ${trade['price']:.4f}\n")
                f.write(f"  Trade Value:  ${trade['value_usd']:.2f}\n")
                f.write(f"  Fees:         ${trade['fees']:.4f}\n")
                f.write(f"  SL Mult:      {trade['sl_mult']:.2f}x ATR\n")
                f.write(f"  TP Mult:      {trade['tp_mult']:.2f}x ATR\n")
                
                # Add Exit Reason if applicable
                if trade.get('exit_reason'):
                    pnl_str = f"${trade['exit_pnl']:.2f}"
                    if trade['exit_pnl'] > 0:
                        pnl_str = f"+{pnl_str}"
                    f.write(f"  Exit Result:  {trade['exit_reason']} - PnL: {pnl_str}\n")
                
                f.write(f"  Portfolio:    ${trade['portfolio_value_before']:.2f} → ${trade['portfolio_value_after']:.2f}\n")
                f.write(f"  Cash:         ${trade['cash_before']:.2f} → ${trade['cash_after']:.2f}\n")
                f.write("-" * 100 + "\n")
        else:
            f.write("No trades executed in 2025.\n")
    
    print(f"✅ Saved {len(trades_2025)} trades from 2025 to {trades_txt_path}")
    
    # Save all trades to CSV for analysis
    if len(trade_log) > 0:
        trades_df = pd.DataFrame(trade_log)
        trades_csv_path = os.path.join(data_dir, 'trades_all.csv')
        trades_df.to_csv(trades_csv_path, index=False)
        print(f"✅ Saved all {len(trade_log)} trades to {trades_csv_path}")
    
    # ========== GENERATE VISUALIZATIONS ==========
    print("\n" + "=" * 80)
    print("📊 Generating Visualizations...")
    
    df_res = pd.DataFrame(history)
    df_res.set_index('timestamp', inplace=True)
    
    # 1. Portfolio Value Chart
    plt.figure(figsize=(14, 6))
    plt.plot(df_res.index, df_res['portfolio_value'], label='Portfolio Value', linewidth=2, color='#2E86AB')
    plt.title('Backtest: Portfolio Value Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'backtest_portfolio_value.png'), dpi=150)
    plt.close()
    print("✅ Saved backtest_portfolio_value.png")
    
    # 2. Drawdown Chart
    plt.figure(figsize=(14, 6))
    plt.fill_between(df_res.index, df_res['drawdown'] * 100, 0, 
                     label='Drawdown %', color='#D62828', alpha=0.6)
    plt.title('Backtest: Drawdown Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'backtest_drawdown.png'), dpi=150)
    plt.close()
    print("✅ Saved backtest_drawdown.png")
    
    # 3. Price Charts with Buy/Sell Markers (one per asset)
    assets = ['BTC', 'ETH', 'SOL', 'EUR', 'GBP', 'JPY']
    
    for asset in assets:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot price line
        price_col = f'{asset}_price'
        if price_col in df_res.columns:
            ax.plot(df_res.index, df_res[price_col], label=f'{asset} Price', 
                   linewidth=2, color='#06A77D', alpha=0.8)
        
        # Plot buy/sell markers
        asset_trades = [t for t in trade_log if t['asset'] == asset]
        
        buy_trades = [t for t in asset_trades if t['type'] == 'BUY']
        sell_trades = [t for t in asset_trades if t['type'] == 'SELL']
        
        if buy_trades:
            buy_times = [t['timestamp'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            ax.scatter(buy_times, buy_prices, marker='^', s=120, c='#00CC66', 
                      label='Buy', edgecolors='black', linewidths=1.5, zorder=5)
        
        if sell_trades:
            sell_times = [t['timestamp'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            ax.scatter(sell_times, sell_prices, marker='v', s=120, c='#FF3366', 
                      label='Sell', edgecolors='black', linewidths=1.5, zorder=5)
        
        ax.set_title(f'{asset} Price Chart with Trading Activity', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{asset} Price ($)', fontsize=12)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        chart_path = os.path.join(data_dir, f'backtest_{asset}_trades.png')
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"✅ Saved backtest_{asset}_trades.png ({len(asset_trades)} trades)")
    
    # 4. Combined Multi-Asset Overview
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Multi-Asset Trading Overview', fontsize=18, fontweight='bold')
    
    for idx, asset in enumerate(assets):
        ax = axes[idx // 2, idx % 2]
        
        price_col = f'{asset}_price'
        if price_col in df_res.columns:
            ax.plot(df_res.index, df_res[price_col], linewidth=1.5, color='#06A77D', alpha=0.7)
        
        asset_trades = [t for t in trade_log if t['asset'] == asset]
        buy_trades = [t for t in asset_trades if t['type'] == 'BUY']
        sell_trades = [t for t in asset_trades if t['type'] == 'SELL']
        
        if buy_trades:
            ax.scatter([t['timestamp'] for t in buy_trades], [t['price'] for t in buy_trades],
                      marker='^', s=60, c='#00CC66', edgecolors='black', linewidths=0.8, zorder=5)
        if sell_trades:
            ax.scatter([t['timestamp'] for t in sell_trades], [t['price'] for t in sell_trades],
                      marker='v', s=60, c='#FF3366', edgecolors='black', linewidths=0.8, zorder=5)
        
        ax.set_title(f'{asset} ({len(asset_trades)} trades)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'backtest_all_assets_overview.png'), dpi=150)
    plt.close()
    print("✅ Saved backtest_all_assets_overview.png")
    
    # ========== SAVE METRICS ==========
    print("\n" + "=" * 80)
    print("📈 Calculating Final Metrics...")
    
    metrics = {
        'model_path': model_path,
        'normalizer_path': stats_path,
        'total_steps': len(df_res),
        'total_trades': len(trade_log),
        'trades_2025': len(trades_2025),
        'final_value': float(df_res['portfolio_value'].iloc[-1]),
        'initial_value': 10000.0,
        'total_return_pct': float((df_res['portfolio_value'].iloc[-1] - 10000) / 10000 * 100),
        'max_drawdown_pct': float(df_res['drawdown'].max() * 100),
        'total_fees': float(df_res['fees'].sum()),
        'avg_portfolio_value': float(df_res['portfolio_value'].mean()),
        'final_cash': float(df_res['cash'].iloc[-1]),
        'start_date': str(df_res.index[0]),
        'end_date': str(df_res.index[-1])
    }
    
    # Add per-asset trade counts
    for asset in assets:
        asset_trade_count = len([t for t in trade_log if t['asset'] == asset])
        metrics[f'trades_{asset}'] = asset_trade_count
    
    metrics_path = os.path.join(data_dir, 'backtest_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Saved metrics to {metrics_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 BACKTEST SUMMARY")
    print("=" * 80)
    print(json.dumps(metrics, indent=2))
    print("=" * 80)

if __name__ == "__main__":
    # Check for model files
    model_file = "best_model.zip"
    normalizer_file = "final_model_vecnormalize.pkl"
    
    if not os.path.exists(model_file):
        print(f"❌ Model file '{model_file}' not found in current directory.")
        print(f"   Please ensure {model_file} is in the same directory as this script.")
    elif not os.path.exists(normalizer_file):
        print(f"❌ Normalizer file '{normalizer_file}' not found in current directory.")
        print(f"   Please ensure {normalizer_file} is in the same directory as this script.")
    else:
        print("✅ Found model and normalizer files")
        run_backtest(model_path=model_file, stats_path=normalizer_file)
