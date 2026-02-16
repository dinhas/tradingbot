import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

class FeatureEngine:
    def __init__(self):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.feature_names = []
        self._define_feature_names()

    def _define_feature_names(self):
        """Defines the list of 40 features."""
        self.feature_names = [
            # 1. Asset Specific (13)
            "close", "return_1", "return_12", "atr_14", "atr_ratio", 
            "bb_position", "ema_9", "ema_21", "price_vs_ema9", "ema9_vs_ema21",
            "rsi_14", "macd_hist", "volume_ratio",
            
            # 2. Pro Features (11)
            "htf_ema_alignment", "htf_rsi_divergence", "swing_structure_proximity",
            "vwap_deviation", "delta_pressure", "volume_shock", 
            "volatility_squeeze", "wick_rejection_strength", "breakout_velocity",
            "rsi_slope_divergence", "macd_momentum_quality",
            
            # 3. Cross-Asset (5)
            "corr_basket", "rel_strength", "corr_xauusd", "corr_eurusd", "rank",
            
            # 4. Global & Macro (11)
            "market_volatility", "avg_atr_ratio", "asset_dispersion", 
            "adx_14", "return_48", "sma_200_dist", "stoch_rsi",
            "hour_sin", "hour_cos", "day_sin", "day_cos"
        ]

    def preprocess_data(self, data_dict):
        """Preprocesses raw OHLCV data for all assets."""
        aligned_df = self._align_data(data_dict)
        
        for asset in self.assets:
            aligned_df = self._add_technical_indicators(aligned_df, asset)
            
        aligned_df = self._add_cross_asset_features(aligned_df)
        aligned_df = self._add_global_features(aligned_df)
        
        normalized_df = aligned_df.copy()
        normalized_df = self._normalize_features(normalized_df)
        
        normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)
        aligned_df = aligned_df.ffill().fillna(0).astype(np.float32)
        
        return aligned_df, normalized_df

    def _align_data(self, data_dict):
        common_index = None
        for df in data_dict.values():
            if common_index is None: common_index = df.index
            else: common_index = common_index.intersection(df.index)
        
        aligned_df = pd.DataFrame(index=common_index)
        for asset, df in data_dict.items():
            df_subset = df.loc[common_index].copy()
            df_subset.columns = [f"{asset}_{col}" for col in df_subset.columns]
            aligned_df = pd.concat([aligned_df, df_subset], axis=1)
        return aligned_df

    def _add_technical_indicators(self, df, asset):
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        open_ = df[f"{asset}_open"]
        volume = df[f"{asset}_volume"]
        
        # Core Indicators
        df[f"{asset}_return_1"] = close.pct_change(1)
        df[f"{asset}_return_12"] = close.pct_change(12)
        
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_ma = atr.rolling(window=20).mean()
        df[f"{asset}_atr_14"] = atr
        df[f"{asset}_atr_ratio"] = atr / (atr_ma + 1e-9)
        
        bb = BollingerBands(close, window=20, window_dev=2)
        df[f"{asset}_bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
        
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema21 = EMAIndicator(close, window=21).ema_indicator()
        df[f"{asset}_ema_9"] = ema9
        df[f"{asset}_ema_21"] = ema21
        df[f"{asset}_price_vs_ema9"] = (close - ema9) / (ema9 + 1e-9)
        df[f"{asset}_ema9_vs_ema21"] = (ema9 - ema21) / (ema21 + 1e-9)
        
        df[f"{asset}_rsi_14"] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        df[f"{asset}_macd_hist"] = macd.macd_diff()
        
        vol_ma = volume.rolling(window=20).mean()
        df[f"{asset}_volume_ratio"] = volume / (vol_ma + 1e-9)
        
        # --- PRO FEATURES ---
        # HTF
        close_1h = close.resample('60min').last().reindex(close.index, method='ffill')
        ema21_1h = close_1h.ewm(span=21, adjust=False).mean()
        df[f"{asset}_htf_ema_alignment"] = (close - ema21_1h) / (atr + 1e-9)
        df[f"{asset}_htf_rsi_divergence"] = df[f"{asset}_rsi_14"] - RSIIndicator(close_1h, window=14).rsi().reindex(close.index, method='ffill')
        
        swing_high = high.rolling(40).max()
        swing_low = low.rolling(40).min()
        df[f"{asset}_swing_structure_proximity"] = np.minimum((swing_high - close), (close - swing_low)) / (atr + 1e-9)

        # Volume/PA
        vwap = (close * volume).groupby(df.index.date).cumsum() / (volume.groupby(df.index.date).cumsum() + 1e-9)
        df[f"{asset}_vwap_deviation"] = (close - vwap) / (atr + 1e-9)
        df[f"{asset}_delta_pressure"] = (volume * np.sign(close - open_)).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
        df[f"{asset}_volume_shock"] = np.log(volume / (vol_ma + 1e-9) + 1e-6)
        df[f"{asset}_volatility_squeeze"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (atr + 1e-9)
        
        body = (close - open_).abs()
        wicks = (high - df[[f"{asset}_open", f"{asset}_close"]].max(axis=1)) + (df[[f"{asset}_open", f"{asset}_close"]].min(axis=1) - low)
        df[f"{asset}_wick_rejection_strength"] = (wicks / (body + 1e-6)).rolling(3).mean()
        
        range_h, range_l = high.rolling(20).max().shift(1), low.rolling(20).min().shift(1)
        df[f"{asset}_breakout_velocity"] = np.where(close > range_h, (close - range_h)/atr, np.where(close < range_l, (close - range_l)/atr, 0))
        
        df[f"{asset}_rsi_slope_divergence"] = np.sign(close.diff(10)) - np.sign(df[f"{asset}_rsi_14"].diff(10))
        df[f"{asset}_macd_momentum_quality"] = df[f"{asset}_macd_hist"].diff() * np.sign(macd.macd_signal())
        
        # New Additions
        df[f"{asset}_adx_14"] = ADXIndicator(high, low, close).adx()
        df[f"{asset}_return_48"] = close.pct_change(48) # 4H
        sma200 = SMAIndicator(close, window=200).sma_indicator()
        df[f"{asset}_sma_200_dist"] = (close - sma200) / (atr + 1e-9)
        df[f"{asset}_stoch_rsi"] = StochRSIIndicator(close).stochrsi()

        return df

    def _add_cross_asset_features(self, df):
        returns = df[[f"{a}_return_1" for a in self.assets]]
        basket_return = returns.mean(axis=1)
        for asset in self.assets:
            asset_ret = df[f"{asset}_return_1"]
            df[f"{asset}_corr_basket"] = asset_ret.rolling(50).corr(basket_return)
            df[f"{asset}_rel_strength"] = asset_ret - basket_return
            df[f"{asset}_corr_xauusd"] = asset_ret.rolling(50).corr(df.get("XAUUSD_return_1", pd.Series(0, index=df.index)))
            df[f"{asset}_corr_eurusd"] = asset_ret.rolling(50).corr(df.get("EURUSD_return_1", pd.Series(0, index=df.index)))
            df[f"{asset}_rank"] = df[[f"{a}_return_12" for a in self.assets]].rank(axis=1, ascending=False)[f"{asset}_return_12"]
        return df

    def _add_global_features(self, df):
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        df['asset_dispersion'] = df[[f"{a}_return_1" for a in self.assets]].std(axis=1)
        df['market_volatility'] = df[[f"{a}_atr_ratio" for a in self.assets]].mean(axis=1)
        df['avg_atr_ratio'] = df['market_volatility'] # Alias for clarity if needed
        return df

    def _normalize_features(self, df):
        for col in df.columns:
            if any(x in col for x in ["close", "return", "atr", "ema", "rsi", "macd", "volume", "htf", "swing", "vwap", "delta", "volatility", "wick", "breakout", "dispersion", "adx", "sma", "stoch"]):
                rolling_median = df[col].rolling(window=500).median()
                rolling_iqr = df[col].rolling(window=500).quantile(0.75) - df[col].rolling(window=500).quantile(0.25)
                df[col] = (df[col] - rolling_median) / (rolling_iqr + 1e-6)
                df[col] = df[col].clip(-5, 5)
        return df

    def get_observation(self, current_step_data, portfolio_state=None, asset=None):
        """Constructs the 40-feature observation vector."""
        if asset is None: return np.zeros(40)
        
        obs = []
        # Match order in _define_feature_names
        for feature in self.feature_names:
            if feature in ["hour_sin", "hour_cos", "day_sin", "day_cos", "asset_dispersion", "market_volatility", "avg_atr_ratio"]:
                obs.append(current_step_data.get(feature, 0))
            else:
                obs.append(current_step_data.get(f"{asset}_{feature}", 0))
        
        return np.array(obs, dtype=np.float32)
