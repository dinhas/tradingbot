import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

class FeatureEngine:
    def __init__(self):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.feature_names = []
        self._define_feature_names()

    def _define_feature_names(self):
        """Defines the list of 48 features for the Risk Model."""
        # 1. Base Alpha-style Features (40)
        self.feature_names = [
            "close", "return_1", "return_12",
            "atr_14", "atr_ratio", "bb_position",
            "ema_9", "ema_21", "price_vs_ema9", "ema9_vs_ema21",
            "rsi_14", "macd_hist", "volume_ratio",
            "htf_ema_alignment", "htf_rsi_divergence", "swing_structure_proximity",
            "vwap_deviation", "delta_pressure", "volume_shock",
            "volatility_squeeze", "wick_rejection_strength", "breakout_velocity",
            "rsi_slope_divergence", "macd_momentum_quality",
            "corr_basket", "rel_strength", "corr_xauusd", "corr_eurusd", "rank",
            "risk_on_score", "asset_dispersion", "market_volatility",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "session_asian", "session_london", "session_ny", "session_overlap"
        ]
        
        # 2. Risk Specific Enhanced Features (8)
        self.feature_names.extend([
            "alpha_direction", "alpha_meta", "alpha_quality",
            "spread", "spread_atr_ratio", "atr_percentile",
            "wick_body_ratio", "spread_expansion"
        ])

    def preprocess_data(self, data_dict):
        # Implementation assumed as standard indicator calculation
        pass

    def get_observation_vectorized(self, df, asset):
        """Vectorized extraction of 48 features."""
        obs_cols = []
        for name in self.feature_names:
            # Check if it's a global feature (no asset prefix)
            global_features = [
                "risk_on_score", "asset_dispersion", "market_volatility",
                "hour_sin", "hour_cos", "day_sin", "day_cos",
                "session_asian", "session_london", "session_ny", "session_overlap"
            ]
            if name in global_features:
                obs_cols.append(name)
            else:
                obs_cols.append(f"{asset}_{name}")
        
        return df.reindex(columns=obs_cols, fill_value=0).values.astype(np.float32)

    def get_observation(self, current_step_data, portfolio_state, asset):
        """Single-step extraction of 48 features."""
        obs = []
        for name in self.feature_names:
            global_features = [
                "risk_on_score", "asset_dispersion", "market_volatility",
                "hour_sin", "hour_cos", "day_sin", "day_cos",
                "session_asian", "session_london", "session_ny", "session_overlap"
            ]
            if name in global_features:
                val = current_step_data.get(name, 0)
            else:
                val = current_step_data.get(f"{asset}_{name}", 0)
            obs.append(val)
        return np.array(obs, dtype=np.float32)
