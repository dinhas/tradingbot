import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from shared_constants import FX_ALPHA_ASSETS

class FeatureEngine:
    def __init__(self):
        self.assets = FX_ALPHA_ASSETS
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
        """Defines causal multi-asset features, including side-selection information."""
        self.feature_names = [
            "bollinger_pB", "ema_diff", "rsi_momentum", "rsi",
            "volatility", "atr_norm", "hour", "regime",
            "htf_trend", "htf_ema_dist", "htf_rsi",
            "return_3_atr", "return_6_atr", "return_12_atr",
            "ema_slope_atr", "di_spread",
            "breakout_position",
            "momentum_3", "momentum_6",
            "bar_strength", "intraday_position",
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
        
        bb = BollingerBands(close, window=20, window_dev=2)
        new_cols[f"{asset}_bollinger_pB"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
        
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
        price_atr = atr.replace(0, np.nan)
        for bars in (3, 6, 12):
            new_cols[f"{asset}_return_{bars}_atr"] = (
                (raw_close - raw_close.shift(bars)) / price_atr
            ).clip(-10.0, 10.0)
        ema_20 = raw_close.ewm(span=20, adjust=False).mean()
        new_cols[f"{asset}_ema_slope_atr"] = (
            (ema_20 - ema_20.shift(3)) / price_atr
        ).clip(-5.0, 5.0)
        new_cols[f"{asset}_di_spread"] = (
            (ADXIndicator(raw_high, raw_low, raw_close, window=14).adx_pos()
             - ADXIndicator(raw_high, raw_low, raw_close, window=14).adx_neg()) / 100.0
        ).clip(-1.0, 1.0)
        rolling_high = raw_high.shift(1).rolling(20).max()
        rolling_low = raw_low.shift(1).rolling(20).min()
        new_cols[f"{asset}_breakout_position"] = (
            ((raw_close - rolling_low) / (rolling_high - rolling_low + 1e-8)) * 2.0 - 1.0
        ).clip(-2.0, 2.0)

        # 5b. Directional OHLCV features — pure price action for short vs long
        raw_open = df[f"{asset}_open"]
        new_cols[f"{asset}_momentum_3"] = (raw_close / raw_close.shift(3) - 1.0).clip(-0.05, 0.05)
        new_cols[f"{asset}_momentum_6"] = (raw_close / raw_close.shift(6) - 1.0).clip(-0.10, 0.10)
        bar_range = (raw_high - raw_low + 1e-8)
        new_cols[f"{asset}_bar_strength"] = ((raw_close - raw_open) / bar_range).clip(-1.0, 1.0)
        new_cols[f"{asset}_intraday_position"] = ((raw_close - raw_low) / bar_range).clip(0.0, 1.0)

        # 6. Higher-Timeframe (1H) Trend Features — CAUSAL (only fully completed hours).
        # These expose the same trend context used by the Labeler so the model can
        # actually learn the +1 vs -1 distinction. shift(1) guarantees no lookahead:
        # a 5M bar only ever sees the last fully closed 1H candle.
        close_1h = raw_close.ffill().resample('1h').last().ffill()
        ema_1h = EMAIndicator(close_1h, window=100).ema_indicator()

        trend_1h = pd.Series(np.where(close_1h > ema_1h, 1.0, -1.0), index=close_1h.index)
        trend_1h[ema_1h.isna()] = np.nan
        trend_1h_5m = trend_1h.shift(1).reindex(df.index, method='ffill')
        new_cols[f"{asset}_htf_trend"] = trend_1h_5m.fillna(0).astype(np.float32)

        # Normalized distance of price to the 1H EMA-100 (scale-free via ATR units).
        # NOT z-scored so the trend level information is preserved.
        ema_1h_5m = ema_1h.shift(1).reindex(df.index, method='ffill')
        htf_dist = (raw_close - ema_1h_5m) / (atr * 20.0 + 1e-8)
        new_cols[f"{asset}_htf_ema_dist"] = htf_dist.clip(-3.0, 3.0).fillna(0).astype(np.float32)

        # 1H RSI, centered to [-1, 1] to preserve absolute level (not z-scored).
        rsi_1h = RSIIndicator(close_1h, window=14).rsi()
        rsi_1h_5m = rsi_1h.shift(1).reindex(df.index, method='ffill')
        new_cols[f"{asset}_htf_rsi"] = ((rsi_1h_5m - 50.0) / 50.0).fillna(0).astype(np.float32)

        # 7. Backward Compatibility for Backtester (Not in model features)
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
                f"{asset}_bollinger_pB", f"{asset}_ema_diff",
                f"{asset}_rsi_momentum", f"{asset}_rsi",
                f"{asset}_volatility", f"{asset}_atr_norm",
                f"{asset}_momentum_3", f"{asset}_momentum_6",
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
            f"{asset}_bollinger_pB", f"{asset}_ema_diff",
            f"{asset}_rsi_momentum", f"{asset}_rsi",
            f"{asset}_volatility", f"{asset}_atr_norm", 'hour_of_day',
            f"{asset}_regime", f"{asset}_htf_trend", f"{asset}_htf_ema_dist", f"{asset}_htf_rsi",
            f"{asset}_return_3_atr", f"{asset}_return_6_atr",
            f"{asset}_return_12_atr",
            f"{asset}_ema_slope_atr", f"{asset}_di_spread", f"{asset}_breakout_position",
            f"{asset}_momentum_3", f"{asset}_momentum_6",
            f"{asset}_bar_strength", f"{asset}_intraday_position",
        ]

        return df.reindex(columns=obs_cols, fill_value=0).values.astype(np.float32)

    def get_observation(self, current_step_data, portfolio_state, asset):
        obs = [
            current_step_data.get(f"{asset}_bollinger_pB", 0),
            current_step_data.get(f"{asset}_ema_diff", 0),
            current_step_data.get(f"{asset}_rsi_momentum", 0),
            current_step_data.get(f"{asset}_rsi", 0),
            current_step_data.get(f"{asset}_volatility", 0),
            current_step_data.get(f"{asset}_atr_norm", 0),
            current_step_data.get('hour_of_day', 0),
            current_step_data.get(f"{asset}_regime", 0),
            current_step_data.get(f"{asset}_htf_trend", 0),
            current_step_data.get(f"{asset}_htf_ema_dist", 0),
            current_step_data.get(f"{asset}_htf_rsi", 0),
            current_step_data.get(f"{asset}_return_3_atr", 0),
            current_step_data.get(f"{asset}_return_6_atr", 0),
            current_step_data.get(f"{asset}_return_12_atr", 0),
            current_step_data.get(f"{asset}_ema_slope_atr", 0),
            current_step_data.get(f"{asset}_di_spread", 0),
            current_step_data.get(f"{asset}_breakout_position", 0),
            current_step_data.get(f"{asset}_momentum_3", 0),
            current_step_data.get(f"{asset}_momentum_6", 0),
            current_step_data.get(f"{asset}_bar_strength", 0),
            current_step_data.get(f"{asset}_intraday_position", 0),
        ]
        return np.array(obs, dtype=np.float32)
