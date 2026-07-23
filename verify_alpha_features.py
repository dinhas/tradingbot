import pandas as pd
import numpy as np
import torch
from Alpha.src.feature_engine import FeatureEngine
from Alpha.src.model import AlphaSLModel

def verify():
    fe = FeatureEngine()
    dummy_data = {}
    for asset in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']:
        dummy_data[asset] = pd.DataFrame({
            'open': np.random.randn(1000),
            'high': np.random.randn(1000),
            'low': np.random.randn(1000),
            'close': np.random.randn(1000),
            'volume': np.random.randn(1000)
        }, index=pd.date_range('2025-01-01', periods=1000, freq='5min'))

    print("Preprocessing data...")
    raw, norm = fe.preprocess_data(dummy_data)

    print(f"Normalized features shape: {norm.shape}")
    print(f"Columns: {norm.columns.tolist()}")

    asset = 'EURUSD'
    obs = fe.get_observation_vectorized(norm, asset)
    print(f"Observation vector shape: {obs.shape}")

    model = AlphaSLModel(input_dim=obs.shape[1])
    # Dummy forward pass (batch=1, seq=50, dim=11)
    dummy_input = torch.randn(1, 50, obs.shape[1])
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")

    assert obs.shape[1] == 11, f"Expected 11 features, got {obs.shape[1]}"
    print("Verification SUCCESS: Feature Engine and Model are aligned.")

if __name__ == "__main__":
    verify()
