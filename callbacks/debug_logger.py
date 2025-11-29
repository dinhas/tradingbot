from stable_baselines3.common.callbacks import BaseCallback
import os
from datetime import datetime


class DebugLoggingCallback(BaseCallback):
    """
    Custom callback for detailed debug logging every N steps.
    Saves human-readable snapshots of model behavior to debuglogs folder.
    """
    
    def __init__(self, log_freq=50000, log_dir="debuglogs", verbose=0):
        super(DebugLoggingCallback, self).__init__(verbose)
        self.log_freq = log_freq
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _on_step(self) -> bool:
        # Only log at specified frequency
        if self.num_timesteps % self.log_freq != 0:
            return True
        
        # Get environment (first env from vectorized env)
        env = self.training_env.envs[0]
        
        # Get current state from environment
        try:
            portfolio_value = env.portfolio_value
            cash = env.cash
            holdings = env.holdings
            peak_value = env.peak_value
            
            # Get current prices
            current_data = env.data.iloc[env.current_step]
            prices = {asset: current_data[f"{asset}_close"] for asset in env.assets}
            
            # Calculate portfolio weights
            weights = {}
            for asset in env.assets:
                value = holdings[asset] * prices[asset]
                weights[asset] = (value / portfolio_value * 100) if portfolio_value > 0 else 0
            weights['CASH'] = (cash / portfolio_value * 100) if portfolio_value > 0 else 0
            
            # Get risk parameters
            sl_mult = env.sl_multiplier
            tp_mult = env.tp_multiplier
            rr_ratio = tp_mult / (sl_mult + 1e-9)
            
            # Calculate drawdown
            drawdown = ((peak_value - portfolio_value) / peak_value * 100) if peak_value > 0 else 0
            
            # Get anti-cheat tracking
            cash_hoarding_steps = env.cash_hoarding_steps
            stale_steps = env.stale_steps
            
            # Format timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create log filename
            log_filename = os.path.join(self.log_dir, f"step_{self.num_timesteps:06d}.txt")
            
            # Write human-readable log
            with open(log_filename, 'w') as f:
                f.write("="*60 + "\n")
                f.write("              TRAINING DEBUG LOG\n")
                f.write("="*60 + "\n")
                f.write(f"Timestamp:      {timestamp}\n")
                f.write(f"Training Step:  {self.num_timesteps:,}\n")
                f.write(f"Episode:        {self.locals.get('n_calls', 'N/A')}\n")
                f.write("\n")
                
                f.write("PORTFOLIO ALLOCATION:\n")
                f.write("-" * 60 + "\n")
                for asset in env.assets:
                    value = holdings[asset] * prices[asset]
                    f.write(f"  {asset:4s}:  {weights[asset]:5.1f}%  (${value:,.2f})\n")
                f.write(f"  CASH:  {weights['CASH']:5.1f}%  (${cash:,.2f})\n")
                f.write("\n")
                
                f.write("RISK MANAGEMENT:\n")
                f.write("-" * 60 + "\n")
                f.write(f"  Stop Loss:    {sl_mult:.2f}x ATR\n")
                f.write(f"  Take Profit:  {tp_mult:.2f}x ATR\n")
                rr_status = "✅" if 1.5 <= rr_ratio <= 4.0 else "❌"
                f.write(f"  R:R Ratio:    1:{rr_ratio:.2f} {rr_status}\n")
                f.write("\n")
                
                f.write("PERFORMANCE:\n")
                f.write("-" * 60 + "\n")
                f.write(f"  Portfolio Value: ${portfolio_value:,.2f}\n")
                f.write(f"  Peak Value:      ${peak_value:,.2f}\n")
                f.write(f"  Drawdown:        {drawdown:.2f}%\n")
                
                # Calculate total fees from history if available
                total_fees = sum([step.get('fees', 0) for step in env.history]) if hasattr(env, 'history') else 0
                f.write(f"  Total Fees:      ${total_fees:,.2f}\n")
                f.write("\n")
                
                f.write("ANTI-CHEAT STATUS:\n")
                f.write("-" * 60 + "\n")
                cash_status = "⚠️" if cash_hoarding_steps > 10 else "✅"
                f.write(f"  Cash Hoarding Steps: {cash_hoarding_steps}/10 {cash_status}\n")
                stale_status = "⚠️" if stale_steps > 20 else "✅"
                f.write(f"  Stale Steps:         {stale_steps}/20 {stale_status}\n")
                f.write("\n")
                
                f.write("="*60 + "\n")
            
            if self.verbose > 0:
                print(f"[DEBUG LOG] Saved to {log_filename}")
        
        except Exception as e:
            print(f"[DEBUG LOG ERROR] Failed to write log: {e}")
        
        return True
