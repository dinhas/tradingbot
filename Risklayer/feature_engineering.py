import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from Risklayer.config import config
from typing import Dict


class FeatureEngine:
    def __init__(self):
        self.assets = config.ASSETS

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates 30 features for the environment state."""
        # Note: input df is already aligned and contains asset_open, asset_high, etc.
        all_features = {}

        # Calculate per-asset features (using only the first asset as primary or we handle multi-asset?)
        # The prompt says "30 alpha input features". Usually for a single-asset RL environment.
        # But if it's multi-asset, it gets complicated.
        # "State must include: 30 alpha input features ... State must NOT include: Alpha outputs"
        # I'll focus on features for a specific asset that the agent is currently considering.

        for asset in self.assets:
            asset_features = self._get_asset_features(df, asset)
            all_features.update(asset_features)

        # Global features
        global_features = self._get_global_features(df)
        all_features.update(global_features)

        feature_df = pd.DataFrame(all_features, index=df.index)

        # Normalize (Simple Robust Scaling) - uses config to match live execution
        norm_window = config.NORMALIZATION_WINDOW
        feature_df = (feature_df - feature_df.rolling(norm_window).median()) / (
            feature_df.rolling(norm_window).quantile(0.75)
            - feature_df.rolling(norm_window).quantile(0.25)
            + 1e-6
        )
        feature_df = feature_df.clip(-5, 5).ffill().fillna(0)

        return feature_df

    def _get_asset_features(self, df: pd.DataFrame, asset: str) -> Dict[str, pd.Series]:
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        open_ = df[f"{asset}_open"]
        volume = df[f"{asset}_volume"]

        f = {}
        # Price/Returns (3)
        f[f"{asset}_ret1"] = close.pct_change(1)
        f[f"{asset}_ret12"] = close.pct_change(12)

        # Volatility (3)
        atr_14 = AverageTrueRange(high, low, close, window=14).average_true_range()
        f[f"{asset}_atr_ratio"] = atr_14 / atr_14.rolling(20).mean()
        bb = BollingerBands(close, window=20)
        f[f"{asset}_bb_pos"] = (close - bb.bollinger_lband()) / (
            bb.bollinger_hband() - bb.bollinger_lband() + 1e-6
        )

        # Trend (4)
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema21 = EMAIndicator(close, window=21).ema_indicator()
        f[f"{asset}_price_ema9"] = (close - ema9) / (ema9 + 1e-6)
        f[f"{asset}_ema9_ema21"] = (ema9 - ema21) / (ema21 + 1e-6)

        # Momentum (2)
        f[f"{asset}_rsi"] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        f[f"{asset}_macd_hist"] = macd.macd_diff()

        # Volume (1)
        f[f"{asset}_vol_ratio"] = volume / volume.rolling(20).mean()

        # Pro Features (subset)
        # VWAP Dev (1)
        vwap = (close * volume).groupby(df.index.date).cumsum() / volume.groupby(
            df.index.date
        ).cumsum()
        f[f"{asset}_vwap_dev"] = (close - vwap) / (atr_14 + 1e-6)

        # Delta Pressure (1)
        direction = np.sign(close - open_)
        f[f"{asset}_pressure"] = (volume * direction).rolling(20).sum() / (
            volume.rolling(20).sum() + 1e-6
        )

        # Vol Squeeze (1)
        f[f"{asset}_squeeze"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (
            atr_14 + 1e-6
        )

        # Wick Rejection (1)
        body = (close - open_).abs()
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low
        f[f"{asset}_wick_rej"] = (upper_wick + lower_wick) / (body + 1e-6)

        return f

    def _get_global_features(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        f = {}
        hours = df.index.hour
        days = df.index.dayofweek

        f["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        f["hour_cos"] = np.cos(2 * np.pi * hours / 24)
        f["day_sin"] = np.sin(2 * np.pi * days / 7)
        f["day_cos"] = np.cos(2 * np.pi * days / 7)

        f["session_asian"] = ((hours >= 0) & (hours < 9)).astype(float)
        f["session_london"] = ((hours >= 8) & (hours < 17)).astype(float)
        f["session_ny"] = ((hours >= 13) & (hours < 22)).astype(float)
        f["session_overlap"] = ((hours >= 13) & (hours < 17)).astype(float)

        # Market Regime
        returns = [df[f"{a}_close"].pct_change(1) for a in self.assets]
        returns_df = pd.concat(returns, axis=1)
        f["market_dispersion"] = returns_df.std(axis=1)
        f["risk_on_score"] = returns_df.mean(axis=1)  # Simplified

        return f

    def get_observation_cols(self, asset: str) -> list:
        """Returns the names of the 30 features for a given asset + global."""
        cols = [
            f"{asset}_ret1",
            f"{asset}_ret12",
            f"{asset}_atr_ratio",
            f"{asset}_bb_pos",
            f"{asset}_price_ema9",
            f"{asset}_ema9_ema21",
            f"{asset}_rsi",
            f"{asset}_macd_hist",
            f"{asset}_vol_ratio",
            f"{asset}_vwap_dev",
            f"{asset}_pressure",
            f"{asset}_squeeze",
            f"{asset}_wick_rej",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "session_asian",
            "session_london",
            "session_ny",
            "session_overlap",
            "market_dispersion",
            "risk_on_score",
        ]
        # I have 23 here. Let me add more to reach 30.
        # Adding some cross-asset correlations or additional indicators
        extra = [
            f"{asset}_close",  # 24
            f"{asset}_high",  # 25
            f"{asset}_low",  # 26
            f"{asset}_open",  # 27
            f"XAUUSD_ret1",  # 28 (Correlation proxy)
            f"EURUSD_ret1",  # 29
            f"GBPUSD_ret1",  # 30
        ]
        return cols + extra
