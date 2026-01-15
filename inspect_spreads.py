import pandas as pd
import numpy as np
from pathlib import Path

# Spread definitions from RiskTradingEnv
SPREADS = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.00015,
    "USDJPY": 0.01,
    "USDCHF": 0.00015,
    "XAUUSD": 0.20,
}

def inspect_spreads():
    assets = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD"]
    all_samples = []

    for asset in assets:
        file_path = Path(f"data/{asset}_5m.parquet")
        if not file_path.exists():
            print(f"Warning: {file_path} not found.")
            continue

        df = pd.read_parquet(file_path)
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Filter for 2025 data
        df_2025 = df[df.index.year == 2025]
        
        if len(df_2025) == 0:
            print(f"Warning: No 2025 data for {asset}, using latest available.")
            df_2025 = df.tail(100) # Fallback

        # Pick 20 samples
        if len(df_2025) > 20:
            indices = np.linspace(0, len(df_2025) - 1, 20, dtype=int)
            samples = df_2025.iloc[indices].copy()
        else:
            samples = df_2025.copy()

        spread = SPREADS[asset]
        samples['Asset'] = asset
        samples['Bid'] = samples['close']
        samples['Ask'] = samples['close'] + spread
        samples['Spread_Val'] = spread
        
        # Keep relevant columns
        result = samples[['Asset', 'Bid', 'Ask', 'Spread_Val']].reset_index()
        all_samples.append(result)

    if not all_samples:
        print("No data found.")
        return

    final_df = pd.concat(all_samples, ignore_index=True)
    
    # Save to CSV for inspection
    output_path = "spreads_inspection_2025.csv"
    final_df.to_csv(output_path, index=False)
    
    # Display summary
    print(f"\nCreated spreads inspector report: {output_path}")
    print("-" * 60)
    for asset in assets:
        asset_data = final_df[final_df['Asset'] == asset]
        if not asset_data.empty:
            print(f"{asset}: {len(asset_data)} samples | Spread: {SPREADS[asset]}")
    
    print("-" * 60)
    print("\nFirst 5 rows of the report:")
    print(final_df.head())

if __name__ == "__main__":
    inspect_spreads()
