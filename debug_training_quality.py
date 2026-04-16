import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# Add project root to sys.path
PROJECT_ROOT = Path(os.getcwd())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Alpha.src.data_loader import DataLoader as MyDataLoader
from Alpha.src.labeling import Labeler
from Alpha.src.feature_engine import FeatureEngine

def analyze_label_quality(df, asset, labels_df):
    """Deep analysis of label characteristics."""
    total = len(labels_df)
    if total == 0:
        return "No labels generated."

    counts = labels_df['direction'].value_counts()
    buys = counts.get(1, 0)
    sells = counts.get(-1, 0)
    neutrals = counts.get(0, 0)

    # 1. Label distribution
    stats = [
        f"\n--- {asset} Label Distribution ---",
        f"Total Samples: {total}",
        f"Buys (1)     : {buys} ({(buys/total)*100:.1f}%)",
        f"Sells (-1)   : {sells} ({(sells/total)*100:.1f}%)",
        f"Neutrals (0) : {neutrals} ({(neutrals/total)*100:.1f}%)",
        f"Buy/Sell Imbalance: {abs(buys-sells)/max(1, buys+sells)*100:.1f}%"
    ]

    # 2. Sequential Analysis (Autocorrelation/Stability)
    directions = labels_df['direction'].values
    changes = np.diff(directions) != 0
    stability = (1 - np.sum(changes) / len(directions)) * 100
    stats.append(f"Label Stability (Same as previous): {stability:.1f}%")
    
    # 3. Streak Analysis
    streaks = []
    current_val = directions[0]
    current_streak = 1
    for i in range(1, len(directions)):
        if directions[i] == current_val:
            current_streak += 1
        else:
            streaks.append((current_val, current_streak))
            current_val = directions[i]
            current_streak = 1
    streaks.append((current_val, current_streak))
    
    avg_streak = np.mean([s[1] for s in streaks])
    max_streak = np.max([s[1] for s in streaks])
    stats.append(f"Average Streak Length: {avg_streak:.1f} candles")
    stats.append(f"Max Streak Length    : {max_streak} candles")

    return "\n".join(stats)

def run_dataset_debug():
    print("Initializing Quality Debugger...")
    data_dir = os.path.join(os.getcwd(), "data")
    
    loader = MyDataLoader(data_dir=data_dir)
    labeler = Labeler(max_bars=24) # 2 Hours per user request
    engine = FeatureEngine()

    try:
        aligned_df, normalized_df = loader.get_features()
    except Exception as e:
        print(f"Error loading features: {e}")
        return

    all_results = []
    
    print(f"\nAnalyzing {len(loader.assets)} assets...")
    
    for asset in loader.assets:
        try:
            labels_df = labeler.label_data(aligned_df, asset)
            
            # Align with normalized features to check for NaNs or session overlaps
            common_indices = labels_df.index.intersection(normalized_df.index)
            labels_df = labels_df.loc[common_indices]
            
            report = analyze_label_quality(aligned_df, asset, labels_df)
            print(report)
            all_results.append(labels_df)
            
        except Exception as e:
            print(f"Error processing {asset}: {e}")

    if not all_results:
        print("No results to summarize.")
        return

    # Aggregate Summary
    full_labels = pd.concat(all_results)
    total_samples = len(full_labels)
    
    print("\n" + "="*40)
    print("GLOBAL DATASET SUMMARY")
    print("="*40)
    print(f"Total Combined Samples: {total_samples}")
    
    dir_counts = full_labels['direction'].value_counts(normalize=True) * 100
    print("\nGlobal Class Proportions:")
    for cls, pct in dir_counts.items():
        label_name = {1: "Buy", -1: "Sell", 0: "Neutral"}[cls]
        print(f"  {label_name:<8}: {pct:.1f}%")

    print("\nQuality Recommendation:")
    neutral_pct = dir_counts.get(0, 0)
    if neutral_pct > 80:
        print("  [!] WARNING: Neutral (0) dominates (>80%). The model might struggle to learn entries.")
        print("  [!] Consider increasing max_bars or adjusting TP/SL multipliers.")
    elif neutral_pct < 40:
        print("  [!] WARNING: Very few Neutrals. Labels might be too noisy or TP/SL too tight.")
    else:
        print("  [✓] Balance looks acceptable for training.")

    print("\nNext Steps:")
    print("1. If stability is > 95%, your labels are very redundant. Try CUSUM sampling.")
    print("2. If Buy/Sell imbalance is > 20%, verify your trend logic or data period.")
    print("="*40)

if __name__ == "__main__":
    # Note: MyDataLoader might need to be imported correctly based on your file structure
    # In run_pipeline.py it is: from Alpha.src.data_loader import DataLoader as MyDataLoader
    # Adjusting for that:
    from Alpha.src.data_loader import DataLoader as MyDataLoader
    run_dataset_debug()
