import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from src.indicators import calculate_donchian_channels, calculate_atr, calculate_volume_ma, add_all_indicators

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2025-01-01', periods=100, freq='5min')
    data = {
        'timestamp': dates,
        'open': 1.1 + np.random.randn(100) * 0.001,
        'high': 1.1 + np.random.randn(100) * 0.001 + 0.0005,
        'low': 1.1 + np.random.randn(100) * 0.001 - 0.0005,
        'close': 1.1 + np.random.randn(100) * 0.001,
        'volume': np.random.randint(100, 500, 100)
    }
    df = pd.DataFrame(data)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df

def test_donchian_channels(sample_data):
    """Test Donchian channel calculation."""
    df = calculate_donchian_channels(sample_data, period=20)

    # Check columns exist
    assert 'donchian_high' in df.columns
    assert 'donchian_low' in df.columns

    # Check NaN in first 19 rows
    assert df['donchian_high'].iloc[:19].isna().all()

    # Check valid values after period
    assert not df['donchian_high'].iloc[20:].isna().any()

    # Check donchian_high >= donchian_low
    valid_rows = df.iloc[20:]
    assert (valid_rows['donchian_high'] >= valid_rows['donchian_low']).all()

    print("✅ Donchian channels test passed")

def test_atr_calculation(sample_data):
    """Test ATR calculation."""
    df = calculate_atr(sample_data, period=14)

    # Check column exists
    assert 'atr' in df.columns

    # ATR should be positive
    valid_atr = df['atr'].dropna()
    assert (valid_atr > 0).all()

    # Check reasonable range for EUR/USD (should be small)
    assert valid_atr.max() < 0.01  # ATR shouldn't be more than 100 pips

    print("✅ ATR test passed")

def test_volume_ma(sample_data):
    """Test volume moving average."""
    df = calculate_volume_ma(sample_data, period=10)

    # Check column exists
    assert 'volume_ma' in df.columns

    # Volume MA should be positive
    valid_vol_ma = df['volume_ma'].dropna()
    assert (valid_vol_ma > 0).all()

    # Volume MA should be smoother than raw volume
    assert valid_vol_ma.std() < df['volume'].iloc[10:].std()

    print("✅ Volume MA test passed")

def test_add_all_indicators(sample_data):
    """Test adding all indicators at once."""
    df = add_all_indicators(sample_data, donchian_period=20, atr_period=14, volume_ma_period=10)

    # Check all columns exist
    required_cols = ['donchian_high', 'donchian_low', 'atr', 'volume_ma']
    for col in required_cols:
        assert col in df.columns

    # Original columns should still exist
    assert 'close' in df.columns
    assert 'volume' in df.columns

    print("✅ Add all indicators test passed")

def test_edge_cases():
    """Test edge cases."""
    # Single row
    single_row = pd.DataFrame({
        'timestamp': [pd.Timestamp('2025-01-01')],
        'open': [1.1], 'high': [1.101], 'low': [1.099], 'close': [1.1], 'volume': [100]
    })

    result = add_all_indicators(single_row, donchian_period=20, atr_period=14, volume_ma_period=10)
    assert len(result) == 1
    assert result['donchian_high'].isna().all()  # Not enough data

    print("✅ Edge cases test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
