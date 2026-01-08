import pandas as pd
import os

def analyze_atr(file_path):
    print(f"Loading dataset: {file_path}")
    if not os.path.exists(file_path):
        print("Error: File not found.")
        return

    # Load only necessary columns to save memory
    df = pd.read_parquet(file_path, columns=['pair', 'atr'])
    
    print("\n--- ATR Averages by Pair ---")
    # Group by pair and calculate mean
    stats = df.groupby('pair')['atr'].mean().reset_index()
    stats.columns = ['Pair', 'Average ATR']
    
    # Sort by ATR value for better readability
    stats = stats.sort_values(by='Average ATR', ascending=False)
    
    print(stats.to_string(index=False))
    
    # Calculate global average
    global_avg = df['atr'].mean()
    print(f"\nGlobal Average ATR: {global_avg:.6f}")
    print(f"Total Samples: {len(df):,}")

if __name__ == "__main__":
    dataset_path = "RiskLayer/risk_dataset.parquet"
    analyze_atr(dataset_path)

