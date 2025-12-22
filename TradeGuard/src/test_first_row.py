import pandas as pd
import numpy as np
import os
from feature_calculator import TradeGuardFeatureCalculator

def test_multi_asset_obs():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
    data_dict = {}

    for asset in assets:
        file_path = os.path.join(data_dir, f"{asset}_5m.parquet")
        if os.path.exists(file_path):
            data_dict[asset] = pd.read_parquet(file_path)

    calculator = TradeGuardFeatureCalculator(data_dict)

    # Multi-asset state
    portfolio_state = {asset: {'action_raw': 0.8, 'signal_persistence': 2, 'signal_reversal': 0} for asset in assets}
    portfolio_state.update({'total_drawdown': 0.02, 'total_exposure': 0.5})

    trade_infos = {asset: {'entry': data_dict[asset]['close'].iloc[0], 
                           'sl': data_dict[asset]['close'].iloc[0] * 0.99, 
                           'tp': data_dict[asset]['close'].iloc[0] * 1.02} for asset in assets}

    # Generate single observation for all 5 pairs
    obs = calculator.get_multi_asset_obs(0, trade_infos, portfolio_state)
    
    print(f"Multi-Asset Observation Shape: {obs.shape}")
    
    # Save to CSV for inspection
    df_out = pd.DataFrame([obs], columns=[f'f_{i}' for i in range(len(obs))])
    output_path = os.path.join(data_dir, 'multi_asset_obs.csv')
    df_out.to_csv(output_path, index=False)
    print(f"Saved 105 features to {output_path}")

if __name__ == "__main__":
    test_multi_asset_obs()