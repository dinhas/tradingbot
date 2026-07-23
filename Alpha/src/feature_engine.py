import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

class FeatureEngine:
    def __init__(self):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.feature_names = []
        self._define_feature_names()

    def _kalman_filter(self, data, Q_base=1e-4, R_base=1e-4):
        """
        Adaptive 1D Kalman Filter optimized for regime-aware denoising.
        Optimized Params: Q=1e-4, R=1e-4 (Balanced Configuration).
        """
        if len(data) == 0: return np.array([], dtype=np.float32)
        
        xhat = data[0]
        P = 1.0
        filtered = []
        var_innovation = 1e-5
        alpha = 0.1
        
        for z in data:
            P = P + Q_base
            innovation = z - xhat
            var_innovation = (1 - alpha) * var_innovation + alpha * (innovation ** 2)
            
            Q_adaptive = max(Q_base, 0.05 * var_innovation)
            P = P + Q_adaptive
            
            K = P / (P + R_base)
            xhat = xhat + K * innovation
            P = (1 - K) * P
            filtered.append(xhat)
            
        return np.array(filtered, dtype=np.float32)

    def _get_weights_ffd(self, d, size):
        w = [1.]
        for k in range(1, size):
            w.append(-w[-1] * (d - k + 1) / k)
        return np.array(w[::-1])

    def _frac_diff_fixed(self, series, d, window=20):
        weights = self._get_weights_ffd(d, window)
        prices = series.values
        fd = np.convolve(prices, weights, mode='full')[:len(prices)]
        # Valid output starts at index (window-1)
        if len(fd) >= window:
            fd[:window-1] = fd[window-1]
        return pd.Series(fd, index=series.index)

    def _define_feature_names(self):
        """Defines the list of 11 V3 features based on regime-aware research."""
        self.feature_names = [
            "bollinger_pB", "ema_diff", "macd_hist", "rsi_momentum", "rsi",
            "bb_width", "volatility", "atr_norm", "hour", "regime", "is_tradeable"
        ]

    def preprocess_data(self, data_dict):
        logger = logging.getLogger(__name__)
        aligned_df = self._align_data(data_dict)
        
        for col in aligned_df.columns:
            aligned_df[col] = aligned_df[col].astype(np.float32)
        
        all_new_cols = {}
        for asset in self.assets:
            logger.info(f"Calculating features for {asset} (with Kalman Filtering)...")
            asset_cols = self._get_asset_features(aligned_df, asset)
            all_new_cols.update(asset_cols)
            
        logger.info("Adding time features...")
        time_cols = self._get_time_features(aligned_df)
        all_new_cols.update(time_cols)

        new_features_df = pd.DataFrame(all_new_cols, index=aligned_df.index).astype(np.float32)
        aligned_df = pd.concat([aligned_df, new_features_df], axis=1)
        
        logger.info("Normalizing features...")
        normalized_df = aligned_df.copy()
        normalized_df = self._normalize_features(normalized_df)
        
        normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)
        aligned_df = aligned_df.ffill().fillna(0).astype(np.float32)
        
        return aligned_df, normalized_df

    def _align_data(self, data_dict):
        common_index = None
        for df in data_dict.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        aligned_parts = []
        for asset, df in data_dict.items():
            df_subset = df.loc[common_index].copy()
            df_subset.columns = [f"{asset}_{col}" for col in df_subset.columns]
            aligned_parts.append(df_subset)
            
        return pd.concat(aligned_parts, axis=1)

    def _get_asset_features(self, df, asset):
        raw_close = df[f"{asset}_close"]
        raw_high = df[f"{asset}_high"]
        raw_low = df[f"{asset}_low"]
        
        # 1. Apply Adaptive Kalman Denoising (Balanced config)
        close_filled = raw_close.ffill().bfill().to_numpy(copy=True)
        high_filled = raw_high.ffill().bfill().to_numpy(copy=True)
        low_filled = raw_low.ffill().bfill().to_numpy(copy=True)

        close = pd.Series(self._kalman_filter(close_filled, Q_base=1e-4, R_base=1e-4), index=raw_close.index)
        high = pd.Series(self._kalman_filter(high_filled, Q_base=1e-4, R_base=1e-4), index=raw_close.index)
        low = pd.Series(self._kalman_filter(low_filled, Q_base=1e-4, R_base=1e-4), index=raw_close.index)
        
        new_cols = {}
        
        # 2. Core V3 Technical Features (Calculated on denoised price)
        new_cols[f"{asset}_rsi"] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        new_cols[f"{asset}_macd_hist"] = macd.macd_diff()
        
        bb = BollingerBands(close, window=20, window_dev=2)
        new_cols[f"{asset}_bollinger_pB"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
        new_cols[f"{asset}_bb_width"] = bb.bollinger_wband()
        
        # 3. Distance and Interaction features
        new_cols[f"{asset}_ema_diff"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)
        new_cols[f"{asset}_rsi_momentum"] = new_cols[f"{asset}_rsi"] * (close / close.shift(5) - 1)

        # 4. Volatility Features
        new_cols[f"{asset}_volatility"] = close.pct_change().rolling(20).std()
        new_cols[f"{asset}_atr_norm"] = AverageTrueRange(high, low, close, window=14).average_true_range() / (close + 1e-8)

        # 5. Regime Features (Sophisticated Classification)
        atr_indicator = AverageTrueRange(raw_high, raw_low, raw_close, window=14)
        atr = atr_indicator.average_true_range().fillna(0)
        adx = ADXIndicator(raw_high, raw_low, raw_close, window=14).adx().fillna(0)

        atr_norm_raw = atr / (raw_close + 1e-8)
        atr_q75 = atr_norm_raw.rolling(500).quantile(0.75)

        # Trending = High ADX + Moderate Vol
        is_trending = (adx > 25) & (atr_norm_raw < atr_q75)
        new_cols[f"{asset}_regime"] = is_trending.astype(np.float32) # Trending (1) vs Other (0)
        new_cols[f"{asset}_is_tradeable"] = is_trending.astype(np.float32)

        # 6. Backward Compatibility for Backtester (Not in model features)
        new_cols[f"{asset}_atr"] = atr
        new_cols[f"{asset}_adx"] = adx
        
        return new_cols

    def _get_time_features(self, df):
        new_cols = {}
        hours = df.index.hour
        new_cols['hour_of_day'] = hours
        new_cols['is_late_session'] = ((hours >= 14) & (hours <= 20)).astype(int)
        new_cols['is_friday'] = (df.index.dayofweek == 4).astype(int)
        return new_cols

    def _normalize_features(self, df):
        """V3 Normalization: Rolling 200-window Z-Score."""
        for asset in self.assets:
            cols_to_scale = [
                f"{asset}_bollinger_pB", f"{asset}_ema_diff", f"{asset}_macd_hist",
                f"{asset}_rsi_momentum", f"{asset}_rsi", f"{asset}_bb_width",
                f"{asset}_volatility", f"{asset}_atr_norm"
            ]
            for col in cols_to_scale:
                if col in df.columns:
                    mean = df[col].rolling(200).mean()
                    std = df[col].rolling(200).std()
                    df[col] = (df[col] - mean) / (std + 1e-8)
                    df[col] = df[col].clip(-4.0, 4.0)
            
        if 'hour_of_day' in df.columns:
            df['hour_of_day'] = (df['hour_of_day'] - 12) / 12.0
            
        return df

    def get_observation_vectorized(self, df, asset):
        obs_cols = [
            f"{asset}_bollinger_pB", f"{asset}_ema_diff", f"{asset}_macd_hist",
            f"{asset}_rsi_momentum", f"{asset}_rsi", f"{asset}_bb_width",
            f"{asset}_volatility", f"{asset}_atr_norm", 'hour_of_day',
            f"{asset}_regime", f"{asset}_is_tradeable"
        ]

        return df.reindex(columns=obs_cols, fill_value=0).values.astype(np.float32)

    def get_observation(self, current_step_data, portfolio_state, asset):
        obs = [
            current_step_data.get(f"{asset}_bollinger_pB", 0),
            current_step_data.get(f"{asset}_ema_diff", 0),
            current_step_data.get(f"{asset}_macd_hist", 0),
            current_step_data.get(f"{asset}_rsi_momentum", 0),
            current_step_data.get(f"{asset}_rsi", 0),
            current_step_data.get(f"{asset}_bb_width", 0),
            current_step_data.get(f"{asset}_volatility", 0),
            current_step_data.get(f"{asset}_atr_norm", 0),
            current_step_data.get('hour_of_day', 0),
            current_step_data.get(f"{asset}_regime", 0),
            current_step_data.get(f"{asset}_is_tradeable", 0)
        ]
        return np.array(obs, dtype=np.float32)
