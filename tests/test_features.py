import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from LiveExecution.src.features import FeatureManager

def test_feature_manager_init():
    # Mock DataLoader since it loads actual files during init
    with patch("LiveExecution.src.features.DataLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_macro_data.return_value = pd.DataFrame()
        mock_loader.load_cot_data.return_value = pd.DataFrame()
        mock_loader_class.return_value = mock_loader

        fm = FeatureManager()
        assert set(fm.assets) == set(["EURUSD", "GBPUSD", "USDCHF", "USDJPY"])
        for asset in fm.assets:
            assert asset in fm.history
            assert fm.history[asset].empty

def test_push_candle():
    with patch("LiveExecution.src.features.DataLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_macro_data.return_value = pd.DataFrame()
        mock_loader.load_cot_data.return_value = pd.DataFrame()
        mock_loader_class.return_value = mock_loader

        fm = FeatureManager()
    asset = "EURUSD"

    # 1. Push valid candle
    candle1 = {
        'timestamp': pd.Timestamp('2023-10-01 10:00:00'),
        'open': 1.0500,
        'high': 1.0550,
        'low': 1.0490,
        'close': 1.0520,
        'volume': 100
    }
    fm.push_candle(asset, candle1)
    assert len(fm.history[asset]) == 1
    assert fm.history[asset].index[0] == pd.Timestamp('2023-10-01 10:00:00')
    assert fm.history[asset].iloc[0]['close'] == 1.0520

    # 2. Push duplicate timestamp -> verify updated (not duplicated)
    candle1_updated = {
        'timestamp': pd.Timestamp('2023-10-01 10:00:00'),
        'open': 1.0500,
        'high': 1.0550,
        'low': 1.0490,
        'close': 1.0525,  # updated close
        'volume': 120
    }
    fm.push_candle(asset, candle1_updated)
    assert len(fm.history[asset]) == 1
    assert fm.history[asset].iloc[0]['close'] == 1.0525

    # 3. Push exceeding max_history -> verify trimmed
    fm.max_history = 5
    for i in range(10):
        c = {
            'timestamp': pd.Timestamp('2023-10-01 10:00:00') + pd.Timedelta(minutes=5*i),
            'open': 1.0500, 'high': 1.0550, 'low': 1.0490, 'close': 1.0500 + i*0.0001,
            'volume': 100
        }
        fm.push_candle(asset, c)
    assert len(fm.history[asset]) == 5
    assert fm.history[asset].index[-1] == pd.Timestamp('2023-10-01 10:45:00')

def test_update_data():
    with patch("LiveExecution.src.features.DataLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_macro_data.return_value = pd.DataFrame()
        mock_loader.load_cot_data.return_value = pd.DataFrame()
        mock_loader_class.return_value = mock_loader
        fm = FeatureManager()
    # Mock cTrader response
    mock_bar = MagicMock()
    mock_bar.low = 10490000
    mock_bar.deltaOpen = 10000
    mock_bar.deltaHigh = 60000
    mock_bar.deltaClose = 30000
    mock_bar.volume = 150
    mock_bar.utcTimestampInMinutes = int(pd.Timestamp('2023-10-01 10:00:00').timestamp() / 60)

    mock_res = MagicMock()
    mock_res.trendbar = [mock_bar]

    # symbol_id 1 is EURUSD
    fm.update_data(1, mock_res)
    assert len(fm.history['EURUSD']) == 1
    row = fm.history['EURUSD'].iloc[0]
    assert np.allclose(row['low'], 104.9) # divisor is 100000.0 -> 10490000 / 100000 = 104.9
    assert np.allclose(row['open'], 105.0) # low + deltaOpen / divisor -> 104.9 + 0.1 = 105.0
    assert np.allclose(row['high'], 105.5)
    assert np.allclose(row['close'], 105.2)
    assert row['volume'] == 150

def test_update_from_trendbar():
    with patch("LiveExecution.src.features.DataLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_macro_data.return_value = pd.DataFrame()
        mock_loader.load_cot_data.return_value = pd.DataFrame()
        mock_loader_class.return_value = mock_loader
        fm = FeatureManager()
    mock_bar = MagicMock()
    mock_bar.low = 10490000
    mock_bar.deltaOpen = 10000
    mock_bar.deltaHigh = 60000
    mock_bar.deltaClose = 30000
    mock_bar.volume = 150
    mock_bar.utcTimestampInMinutes = int(pd.Timestamp('2023-10-01 10:00:00').timestamp() / 60)

    fm.update_from_trendbar('EURUSD', mock_bar)
    assert len(fm.history['EURUSD']) == 1
    row = fm.history['EURUSD'].iloc[0]
    assert np.allclose(row['close'], 105.2)

def test_get_atr():
    with patch("LiveExecution.src.features.DataLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_macro_data.return_value = pd.DataFrame()
        mock_loader.load_cot_data.return_value = pd.DataFrame()
        mock_loader_class.return_value = mock_loader
        fm = FeatureManager()
    asset = "EURUSD"
    # Less than 15 bars -> returns 0.0
    assert fm.get_atr(asset) == 0.0

    # Create 20 bars
    for i in range(20):
        c = {
            'timestamp': pd.Timestamp('2023-10-01 10:00:00') + pd.Timedelta(minutes=5*i),
            'open': 1.0500, 'high': 1.0550, 'low': 1.0450, 'close': 1.0510,
            'volume': 100
        }
        fm.push_candle(asset, c)
    atr = fm.get_atr(asset)
    assert atr > 0.0

def test_is_ready():
    with patch("LiveExecution.src.features.DataLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_macro_data.return_value = pd.DataFrame()
        mock_loader.load_cot_data.return_value = pd.DataFrame()
        mock_loader_class.return_value = mock_loader
        fm = FeatureManager()
    assert fm.is_ready() is False

    # populate all 4 assets with 199 bars
    for asset in fm.assets:
        for i in range(199):
            c = {
                'timestamp': pd.Timestamp('2023-10-01 10:00:00') + pd.Timedelta(minutes=5*i),
                'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 100
            }
            fm.push_candle(asset, c)
    assert fm.is_ready() is False

    # Add 1 more to EURUSD -> still False (one asset USDJPY/GBPUSD/USDCHF has 199)
    c = {
        'timestamp': pd.Timestamp('2023-10-01 10:00:00') + pd.Timedelta(minutes=5*200),
        'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 100
    }
    fm.push_candle("EURUSD", c)
    assert fm.is_ready() is False

    # Bring all to 200+
    for asset in fm.assets:
        if len(fm.history[asset]) < 200:
            c = {
                'timestamp': pd.Timestamp('2023-10-01 10:00:00') + pd.Timedelta(minutes=5*200),
                'open': 1.0, 'high': 1.1, 'low': 0.9, 'close': 1.0, 'volume': 100
            }
            fm.push_candle(asset, c)
    assert fm.is_ready() is True

def test_get_asset_name_from_id():
    with patch("LiveExecution.src.features.DataLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_macro_data.return_value = pd.DataFrame()
        mock_loader.load_cot_data.return_value = pd.DataFrame()
        mock_loader_class.return_value = mock_loader
        fm = FeatureManager()
    assert fm._get_asset_name_from_id(1) == 'EURUSD'
    assert fm._get_asset_name_from_id(2) == 'GBPUSD'
    assert fm._get_asset_name_from_id(6) == 'USDCHF'
    assert fm._get_asset_name_from_id(4) == 'USDJPY'
    assert fm._get_asset_name_from_id(999) is None

def test_get_alpha_sequence_and_filter_features():
    with patch("LiveExecution.src.features.DataLoader") as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.load_macro_data.return_value = pd.DataFrame()
        mock_loader.load_cot_data.return_value = pd.DataFrame()
        mock_loader_class.return_value = mock_loader
        fm = FeatureManager()

    # Empty history - when any asset is empty, is_ready is False, and get_alpha_sequence checks if not history empty.
    # But get_alpha_sequence/get_filter_features tries to process even if some are empty but returns None/shapes as built.
    # Actually, let's verify empty case behavior:
    # if not any(not df.empty ...) in get_filter_features returns None.
    assert fm.get_filter_features() is None

    # Populate 250 bars for all assets with correct columns
    # We must use at least 200 bars so alignment doesn't get messed up and features can be computed.
    for asset in fm.assets:
        idx = pd.date_range(start='2023-10-01 10:00:00', periods=250, freq='5min')
        df = pd.DataFrame({
            'open': [1.0500] * 250,
            'high': [1.0550] * 250,
            'low': [1.0450] * 250,
            'close': [1.0510] * 250,
            'volume': [100] * 250
        }, index=idx)
        fm.history[asset] = df

    # Since FeatureEngine might require columns like '_close' and other structures, let's make sure history dataframe has correct index and columns
    seq = fm.get_alpha_sequence("EURUSD", 25)
    assert seq is not None
    assert seq.shape == (25, len(fm.alpha_fe.feature_names))

    filt = fm.get_filter_features()
    assert filt is not None
    # FilterFeatureEngine returns (N, 26) vector
    assert filt.shape[1] == 26
