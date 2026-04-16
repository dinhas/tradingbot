import numpy as np
import os
from collections import Counter

def analyze_distribution():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    labels_path = os.path.join(base_dir, "data", "training_set", "labels.npz")
    
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found at {labels_path}")
        print("Please run the pipeline first to generate data.")
        return

    # Load labels
    data = np.load(labels_path)
    directions = data['direction']
    
    total_timesteps = len(directions)
    counts = Counter(directions)
    
    buys = counts.get(1.0, 0)
    sells = counts.get(-1.0, 0)
    neutrals = counts.get(0.0, 0)
    
    total_trades = buys + sells
    trade_percentage = (total_trades / total_timesteps) * 100 if total_timesteps > 0 else 0
    
    print("-" * 50)
    print("ALPHA MODEL DATA DISTRIBUTION ANALYSIS")
    print("-" * 50)
    print(f"Total Timesteps (Sequences):    {total_timesteps:,}")
    print(f"Timesteps with 'Trades' (TP):   {total_trades:,} ({trade_percentage:.2f}%)")
    print(f"Timesteps with 'No Trade' (SL/VB): {neutrals:,} ({100-trade_percentage:.2f}%)")
    print("-" * 50)
    print(f"Buy Trades (Label  1):         {buys:,} ({ (buys/total_timesteps*100) if total_timesteps > 0 else 0:.2f}%)")
    print(f"Sell Trades (Label -1):         {sells:,} ({ (sells/total_timesteps*100) if total_timesteps > 0 else 0:.2f}%)")
    print(f"Neutral/Hold (Label  0):        {neutrals:,} ({ (neutrals/total_timesteps*100) if total_timesteps > 0 else 0:.2f}%)")
    print("-" * 50)
    
    if total_trades > 0:
        buy_ratio = (buys / total_trades) * 100
        sell_ratio = (sells / total_trades) * 100
        print(f"Trade Bias (Buy/Sell):          {buy_ratio:.1f}% / {sell_ratio:.1f}%")
    
    print("-" * 50)
    print("Interpretation:")
    print(" - 'Trades' are sequences that hit the 4x ATR Profit target.")
    print(" - 'No Trade' (Neutral) represents hitting the 2x ATR Stop Loss")
    print("    OR failing to hit any target within 24 bars (Vertical Barrier).")
    print("-" * 50)

if __name__ == "__main__":
    analyze_distribution()
