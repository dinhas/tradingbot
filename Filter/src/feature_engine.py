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
            # --- Original 16 regime/momentum features ---
            "volatility", "atr_norm", "hour", "regime",
            "return_12_atr",
            "ema_slope_atr", "di_spread", "breakout_position",
            "momentum_6",
            "bar_strength", "intraday_position",
            "adx_momentum", "vol_percentile",
            "activity_ratio",
            "trend_momentum",
            "breakout_conviction",
            # --- Tier 1a: Cross-pair flow (directional USD signal) ---
            "usd_index_return",
            "pair_residual",
            "cross_pair_divergence",
            # --- Tier 1b: Session structure (mean-reversion anchors) ---
            "session_open_dist",
            "prev_session_range_pos",
            "asian_range_pos",
            # --- Tier 1c: Momentum exhaustion (reversal detection) ---
            "consec_dir_bars",
            "wick_ratio",
            "atr_contraction",
            "return_decel",
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

        # Add new cols so far so cross-pair/session features can reference atr etc
        new_features_df = pd.DataFrame(all_new_cols, index=aligned_df.index).astype(np.float32)
        aligned_df = pd.concat([aligned_df, new_features_df], axis=1)

        # Tier 1a: Cross-pair USD flow (needs all assets' closes)
        logger.info("Calculating cross-pair flow features...")
        cross_pair_cols = self._get_cross_pair_features(aligned_df)
        cross_pair_df = pd.DataFrame(cross_pair_cols, index=aligned_df.index).astype(np.float32)
        aligned_df = pd.concat([aligned_df, cross_pair_df], axis=1)

        # Tier 1b: Session structure features
        logger.info("Calculating session structure features...")
        session_cols = self._get_session_features(aligned_df)
        session_df = pd.DataFrame(session_cols, index=aligned_df.index).astype(np.float32)
        aligned_df = pd.concat([aligned_df, session_df], axis=1)

        # Tier 1c: Momentum exhaustion features
        logger.info("Calculating momentum exhaustion features...")
        exhaust_cols = self._get_exhaustion_features(aligned_df)
        exhaust_df = pd.DataFrame(exhaust_cols, index=aligned_df.index).astype(np.float32)
        aligned_df = pd.concat([aligned_df, exhaust_df], axis=1)

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

        # 5c. New V4 features — designed to boost mutual information with labels
        # ADX momentum: rate of change in trend strength (captures trend acceleration)
        new_cols[f"{asset}_adx_momentum"] = (adx - adx.shift(3)).clip(-20.0, 20.0)

        # Volatility regime percentile: where current vol sits in the 500-bar distribution
        # Normalized to [0, 1] — captures regime context
        vol_rolling = close.pct_change().rolling(20).std()
        new_cols[f"{asset}_vol_percentile"] = vol_rolling.rolling(500).rank(pct=True).clip(0.0, 1.0)

        # Return-volatility interaction: risk-adjusted momentum signal
        # High return + low vol = strong signal; high return + high vol = noise
        new_cols[f"{asset}_return_vol_interaction"] = (
            (raw_close / raw_close.shift(6) - 1.0) / (vol_rolling + 1e-8)
        ).clip(-5.0, 5.0)

        # Activity ratio: current bar range relative to ATR — captures intraday activity
        # High activity = potential breakout; low activity = consolidation
        new_cols[f"{asset}_activity_ratio"] = (bar_range / (atr + 1e-8)).clip(0.0, 3.0)

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

        # V4 interaction features — designed to boost MI with labels
        # Trend-momentum interaction: ADX strength × recent return direction
        # Captures "strong trend + momentum" setups that are most tradeable
        trend_momentum = adx / 25.0 * (raw_close / raw_close.shift(6) - 1.0)
        new_cols[f"{asset}_trend_momentum"] = trend_momentum.clip(-2.0, 2.0)

        # Volatility-adjusted RSI: RSI normalized by recent volatility regime
        # High RSI in low vol = overbought; high RSI in high vol = trend continuation
        rsi_val = new_cols.get(f"{asset}_rsi", pd.Series(50.0, index=df.index))
        vol_ratio = vol_rolling / (vol_rolling.rolling(200).mean() + 1e-8)
        new_cols[f"{asset}_rsi_vol_adj"] = ((rsi_val - 50.0) / 50.0 / (vol_ratio + 1e-8)).clip(-3.0, 3.0)

        # Breakout conviction: breakout_position × activity_ratio
        # Strong breakout in active market = high conviction signal
        new_cols[f"{asset}_breakout_conviction"] = (
            new_cols[f"{asset}_breakout_position"] * new_cols[f"{asset}_activity_ratio"]
        ).clip(-3.0, 3.0)

        # EMA slope normalized by volatility: directional conviction
        # Steep EMA slope in low vol = strong trend; steep slope in high vol = noise
        new_cols[f"{asset}_slope_vol_ratio"] = (
            new_cols[f"{asset}_ema_slope_atr"] / (vol_ratio + 1e-8)
        ).clip(-5.0, 5.0)

        # Return acceleration: second derivative of price movement
        # Captures momentum shifts that precede tradeable setups
        ret_3 = (raw_close / raw_close.shift(3) - 1.0)
        ret_6 = (raw_close / raw_close.shift(6) - 1.0)
        new_cols[f"{asset}_return_accel"] = ((ret_3 - ret_6) / (vol_rolling + 1e-8)).clip(-3.0, 3.0)

        # HTF trend strength: magnitude of distance from 1H EMA in ATR units
        # Large distance + trending = strong directional bias
        htf_dist_raw = (raw_close - ema_1h_5m.fillna(raw_close)) / (atr * 20.0 + 1e-8)
        new_cols[f"{asset}_htf_dist_strength"] = (htf_dist_raw * trend_1h_5m.fillna(0)).clip(-3.0, 3.0)

        # 7. Backward Compatibility for Backtester (Not in model features)
        new_cols[f"{asset}_atr"] = atr
        new_cols[f"{asset}_adx"] = adx
        
        return new_cols

    def _get_cross_pair_features(self, df):
        """Cross-pair USD flow features — directional signal from correlated pairs.

        All 4 pairs share USD. When dollar strengthens, EURUSD/GBPUSD fall and
        USDJPY/USDCHF rise. The lead/lag between pairs is 1-3 bars on 5M.
        """
        new_cols = {}

        # Per-bar returns for each asset (1-bar pct change)
        returns = {}
        for asset in self.assets:
            returns[asset] = df[f"{asset}_close"].pct_change().fillna(0)

        # USD index proxy: average signed return across all 4 pairs
        # EURUSD, GBPUSD have negative USD beta; USDJPY, USDCHF have positive
        usd_sign = {'EURUSD': -1.0, 'GBPUSD': -1.0, 'USDJPY': 1.0, 'USDCHF': 1.0}
        usd_index_return = sum(
            usd_sign[a] * returns[a] for a in self.assets
        ) / len(self.assets)

        for asset in self.assets:
            # Smoothed USD index (6-bar EMA for noise reduction)
            usd_smooth = usd_index_return.ewm(span=6, adjust=False).mean()
            new_cols[f"{asset}_usd_index_return"] = usd_smooth.clip(-0.01, 0.01)

            # Pair-specific residual: this pair's move minus what USD flow explains
            beta = usd_sign[asset]
            residual = returns[asset] - (beta * usd_index_return)
            residual_smooth = residual.ewm(span=6, adjust=False).mean()
            new_cols[f"{asset}_pair_residual"] = residual_smooth.clip(-0.005, 0.005)

            # Cross-pair divergence: how many other pairs agree on USD direction
            # If 3 pairs say USD up but this one disagrees → convergence signal
            other_assets = [a for a in self.assets if a != asset]
            other_usd_returns = sum(
                usd_sign[a] * returns[a] for a in other_assets
            ) / len(other_assets)
            other_smooth = other_usd_returns.ewm(span=6, adjust=False).mean()
            own_usd = usd_sign[asset] * returns[asset].ewm(span=6, adjust=False).mean()
            divergence = own_usd - other_smooth
            new_cols[f"{asset}_cross_pair_divergence"] = divergence.clip(-0.005, 0.005)

        return new_cols

    def _get_session_features(self, df):
        """Session structure features — exploit FX intraday mean-reversion patterns.

        London session open (08:00 UTC) and NY session open (13:00 UTC) act as
        directional anchors. Asian session range (00:00-08:00 UTC) defines the
        breakout zone for London.
        """
        new_cols = {}
        hours = df.index.hour

        for asset in self.assets:
            raw_close = df[f"{asset}_close"]
            raw_high = df[f"{asset}_high"]
            raw_low = df[f"{asset}_low"]
            atr = df.get(f"{asset}_atr")
            if atr is None:
                atr = AverageTrueRange(raw_high, raw_low, raw_close, window=14).average_true_range().fillna(0)

            # Session open price (London 08:00 UTC, most liquid session)
            # Use the first bar at or after 08:00 each day as session open
            session_open = raw_close.copy()
            session_open[:] = np.nan
            is_session_start = (hours == 8) & (df.index.minute == 0)
            session_open[is_session_start] = raw_close[is_session_start]
            session_open = session_open.ffill()

            # Distance from session open in ATR units — directional mean-reversion anchor
            dist_from_open = (raw_close - session_open) / (atr + 1e-8)
            new_cols[f"{asset}_session_open_dist"] = dist_from_open.clip(-5.0, 5.0).fillna(0)

            # Previous session range position
            # Where current price sits relative to previous full day's high/low
            daily_high = raw_high.resample('D').max().shift(1).reindex(df.index, method='ffill')
            daily_low = raw_low.resample('D').min().shift(1).reindex(df.index, method='ffill')
            daily_range = daily_high - daily_low + 1e-8
            prev_range_pos = ((raw_close - daily_low) / daily_range) * 2.0 - 1.0
            new_cols[f"{asset}_prev_session_range_pos"] = prev_range_pos.clip(-2.0, 2.0).fillna(0)

            # Asian range position (00:00-08:00 UTC)
            # Where current price sits within the Asian session range
            asian_mask = hours < 8
            asian_high = raw_high.where(asian_mask).resample('D').max().shift(1).reindex(df.index, method='ffill')
            asian_low = raw_low.where(asian_mask).resample('D').min().shift(1).reindex(df.index, method='ffill')
            asian_range = asian_high - asian_low + 1e-8
            asian_pos = ((raw_close - asian_low) / asian_range) * 2.0 - 1.0
            new_cols[f"{asset}_asian_range_pos"] = asian_pos.clip(-3.0, 3.0).fillna(0)

        return new_cols

    def _get_exhaustion_features(self, df):
        """Momentum exhaustion features — detect when a micro-trend is ending.

        5M FX trends last 15-90 minutes. These features identify reversal points
        where directional edge actually lives.
        """
        new_cols = {}

        for asset in self.assets:
            raw_close = df[f"{asset}_close"]
            raw_open = df[f"{asset}_open"]
            raw_high = df[f"{asset}_high"]
            raw_low = df[f"{asset}_low"]

            # Consecutive directional bars: count of same-sign bars in a row
            # 5+ same-direction bars → mean reversion incoming
            bar_dir = np.sign(raw_close.values - raw_open.values)
            consec = np.zeros(len(bar_dir), dtype=np.float32)
            count = 0.0
            prev_dir = 0.0
            for i in range(len(bar_dir)):
                d = bar_dir[i]
                if d == prev_dir and d != 0:
                    count += d
                elif d != 0:
                    count = d
                else:
                    count = 0.0
                consec[i] = count
                prev_dir = d
            new_cols[f"{asset}_consec_dir_bars"] = pd.Series(
                consec / 8.0, index=df.index
            ).clip(-1.0, 1.0)

            # Wick ratio: proportion of bar that is wick (indecision signal)
            # High wick ratio on a directional bar = exhaustion
            bar_range = raw_high - raw_low + 1e-8
            body = (raw_close - raw_open).abs()
            wick_ratio = 1.0 - (body / bar_range)
            # Smooth over 3 bars for stability
            wick_smooth = wick_ratio.rolling(3).mean()
            new_cols[f"{asset}_wick_ratio"] = wick_smooth.clip(0.0, 1.0).fillna(0.5)

            # ATR contraction after expansion: vol spike → vol drop = move over
            atr = df.get(f"{asset}_atr")
            if atr is None:
                atr = AverageTrueRange(raw_high, raw_low, raw_close, window=14).average_true_range().fillna(0)
            atr_fast = atr.rolling(5).mean()
            atr_slow = atr.rolling(20).mean()
            atr_contraction = (atr_fast - atr_slow) / (atr_slow + 1e-8)
            new_cols[f"{asset}_atr_contraction"] = atr_contraction.clip(-2.0, 2.0).fillna(0)

            # Return deceleration: momentum slowing = trend fading
            ret_1 = raw_close.pct_change()
            ret_3 = raw_close.pct_change(3) / 3.0
            decel = (ret_1 - ret_3) / (raw_close.pct_change().rolling(20).std() + 1e-8)
            new_cols[f"{asset}_return_decel"] = decel.clip(-3.0, 3.0).fillna(0)

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
                # Tier 1a cross-pair (small magnitude, benefits from z-score)
                f"{asset}_usd_index_return", f"{asset}_pair_residual",
                f"{asset}_cross_pair_divergence",
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
            f"{asset}_volatility", f"{asset}_atr_norm", 'hour_of_day',
            f"{asset}_regime",
            f"{asset}_return_12_atr",
            f"{asset}_ema_slope_atr", f"{asset}_di_spread", f"{asset}_breakout_position",
            f"{asset}_momentum_6",
            f"{asset}_bar_strength", f"{asset}_intraday_position",
            f"{asset}_adx_momentum", f"{asset}_vol_percentile",
            f"{asset}_activity_ratio",
            f"{asset}_trend_momentum",
            f"{asset}_breakout_conviction",
            # Tier 1a: Cross-pair flow
            f"{asset}_usd_index_return",
            f"{asset}_pair_residual",
            f"{asset}_cross_pair_divergence",
            # Tier 1b: Session structure
            f"{asset}_session_open_dist",
            f"{asset}_prev_session_range_pos",
            f"{asset}_asian_range_pos",
            # Tier 1c: Momentum exhaustion
            f"{asset}_consec_dir_bars",
            f"{asset}_wick_ratio",
            f"{asset}_atr_contraction",
            f"{asset}_return_decel",
        ]

        return df.reindex(columns=obs_cols, fill_value=0).values.astype(np.float32)

    def get_observation(self, current_step_data, portfolio_state, asset):
        obs = [
            current_step_data.get(f"{asset}_volatility", 0),
            current_step_data.get(f"{asset}_atr_norm", 0),
            current_step_data.get('hour_of_day', 0),
            current_step_data.get(f"{asset}_regime", 0),
            current_step_data.get(f"{asset}_return_12_atr", 0),
            current_step_data.get(f"{asset}_ema_slope_atr", 0),
            current_step_data.get(f"{asset}_di_spread", 0),
            current_step_data.get(f"{asset}_breakout_position", 0),
            current_step_data.get(f"{asset}_momentum_6", 0),
            current_step_data.get(f"{asset}_bar_strength", 0),
            current_step_data.get(f"{asset}_intraday_position", 0),
            current_step_data.get(f"{asset}_adx_momentum", 0),
            current_step_data.get(f"{asset}_vol_percentile", 0),
            current_step_data.get(f"{asset}_activity_ratio", 0),
            current_step_data.get(f"{asset}_trend_momentum", 0),
            current_step_data.get(f"{asset}_breakout_conviction", 0),
            # Tier 1a: Cross-pair flow
            current_step_data.get(f"{asset}_usd_index_return", 0),
            current_step_data.get(f"{asset}_pair_residual", 0),
            current_step_data.get(f"{asset}_cross_pair_divergence", 0),
            # Tier 1b: Session structure
            current_step_data.get(f"{asset}_session_open_dist", 0),
            current_step_data.get(f"{asset}_prev_session_range_pos", 0),
            current_step_data.get(f"{asset}_asian_range_pos", 0),
            # Tier 1c: Momentum exhaustion
            current_step_data.get(f"{asset}_consec_dir_bars", 0),
            current_step_data.get(f"{asset}_wick_ratio", 0),
            current_step_data.get(f"{asset}_atr_contraction", 0),
            current_step_data.get(f"{asset}_return_decel", 0),
        ]
        return np.array(obs, dtype=np.float32)
