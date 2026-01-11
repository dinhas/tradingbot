import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochRSIIndicator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, ADXIndicator, AroonIndicator, CCIIndicator
from ta.volatility import AverageTrueRange, BollingerBands, DonchianChannel, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, ForceIndexIndicator, MFIIndicator

class RiskFeatureEngine:
    """
    Feature Engine for the Risk Model (TradeGuard).
    Focuses on 'True Side', 'Pressure', and 'Structure' to determine SL/TP and Skip Trade actions.
    """
    def __init__(self):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.feature_names = []
        self._define_feature_names()

    def _define_feature_names(self):
        """Defines the list of features for reference."""
        # We will populate this as we implement the groups.
        # This list serves as a contract for the observation space.
        
        # 1. Market Pressure & Flow (20)
        self.pressure_features = [
            "rsi_14", "rsi_velocity", "macd_hist", "macd_div_proxy",
            "obv_slope", "cmf", "force_index", 
            "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
            "volume_ratio", "buying_pressure", "selling_pressure",
            "stoch_k", "stoch_d", "mfi", "williams_r", "roc",
            "gap_size", "close_loc_value"
        ]
        
        # 2. Volatility & Structure (18)
        self.volatility_features = [
            "atr_14", "atr_ratio", "bb_width", "bb_pct_b", "std_dev_20",
            "keltner_pos", "donchian_high_dist", "donchian_low_dist",
            "pivot_dist", "avg_candle_range", "parkinson_vol", "z_score",
            "prev_day_high_dist", "prev_day_low_dist", "vol_skew",
            "atr_slope", "high_low_range_pct", "shadow_symmetry"
        ]

        # 3. Trend Strength (15)
        self.trend_features = [
            "adx_14", "di_plus", "di_minus", "aroon_up", "aroon_down",
            "ema_9", "ema_21", "ema_50", "ema_200",
            "price_vs_ema50", "price_vs_ema200", "ema_ribbon_spread",
            "supertrend", "cloud_pos", "parabolic_sar_pos"
        ]

        # 4. Market Regime (10)
        self.regime_features = [
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "session_london", "session_ny", "session_asian",
            "efficiency_ratio", "hurst_exp", "fractal_dim"
        ]
        
        # 5. Alpha & Cross-Asset (15)
        self.alpha_features = [
            "alpha_signal", "alpha_conf", 
            "corr_usd", "corr_gold", "corr_sp500",
            "rel_strength", "beta", "spread",
            "tick_vol", "vol_variance", "price_variance",
            "cov_lead", "sector_mom", "global_vol", "global_trend"
        ]

        # Combine all features per asset
        for asset in self.assets:
            for feat in (self.pressure_features + self.volatility_features + 
                         self.trend_features + self.regime_features + self.alpha_features):
                self.feature_names.append(f"{asset}_{feat}")

    def preprocess_data(self, data_dict):
        """
        Main pipeline to preprocess data.
        """
        # 1. Align Data
        aligned_df = self._align_data(data_dict)
        
        # 2. Pre-calculate market returns once
        all_returns = pd.DataFrame()
        for a in self.assets:
            if f"{a}_close" in aligned_df.columns:
                all_returns[a] = aligned_df[f"{a}_close"].pct_change()
        
        # 3. Add Features per Asset
        for asset in self.assets:
            # Periodically defragment
            if asset != self.assets[0]:
                aligned_df = aligned_df.copy()
                
            # Group 1: Market Pressure & Flow
            aligned_df = self._add_market_pressure_features(aligned_df, asset)
            
            # Group 2: Volatility & Structure
            aligned_df = self._add_volatility_features(aligned_df, asset)
            
            # Group 3: Trend Strength
            aligned_df = self._add_trend_features(aligned_df, asset)
            
            # Group 4: Regime (Some are global, but assigned per asset for now)
            aligned_df = self._add_regime_features(aligned_df, asset)
            
            # Group 5: Alpha/Cross-Asset
            aligned_df = self._add_alpha_features(aligned_df, asset, all_returns)

        return aligned_df

    def _align_data(self, data_dict):
        """Aligns all asset DataFrames to the same index."""
        common_index = None
        for df in data_dict.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        aligned_df = pd.DataFrame(index=common_index)
        for asset, df in data_dict.items():
            df_subset = df.loc[common_index].copy()
            # Standardize column names
            df_subset.columns = [f"{asset}_{col}" for col in df_subset.columns]
            aligned_df = pd.concat([aligned_df, df_subset], axis=1)
            
        return aligned_df

    def _add_market_pressure_features(self, df, asset):
        """
        Group 1: Market Pressure & Flow (The 'True Side')
        """
        close = df[f"{asset}_close"]
        open_ = df[f"{asset}_open"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        volume = df[f"{asset}_volume"]
        
        # 1. RSI & Velocity
        rsi = RSIIndicator(close, window=14).rsi()
        df[f"{asset}_rsi_14"] = rsi
        df[f"{asset}_rsi_velocity"] = rsi.diff(3)
        
        # 2. MACD
        macd = MACD(close)
        df[f"{asset}_macd_hist"] = macd.macd_diff()
        # Proxy divergence: Slope of price vs Slope of MACD
        price_slope = close.diff(3)
        macd_slope = df[f"{asset}_macd_hist"].diff(3)
        # If price rising but MACD falling -> Divergence
        df[f"{asset}_macd_div_proxy"] = np.where(
            (price_slope > 0) & (macd_slope < 0), 1,
            np.where((price_slope < 0) & (macd_slope > 0), -1, 0)
        )

        # 3. Volume Flow (OBV, CMF, Force)
        df[f"{asset}_obv_slope"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume().diff(10)
        df[f"{asset}_cmf"] = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
        df[f"{asset}_force_index"] = ForceIndexIndicator(close, volume, window=13).force_index()
        
        # 4. Candle Structure
        range_ = high - low
        body = (open_ - close).abs()
        df[f"{asset}_candle_body_pct"] = body / range_.replace(0, 1e-9)
        
        # Wicks
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low
        df[f"{asset}_upper_wick_pct"] = upper_wick / range_.replace(0, 1e-9)
        df[f"{asset}_lower_wick_pct"] = lower_wick / range_.replace(0, 1e-9)
        
        # 5. Volume Pressure
        vol_sma = volume.rolling(20).mean()
        df[f"{asset}_volume_ratio"] = volume / vol_sma.replace(0, 1)
        
        # Buying/Selling Pressure (Approximate)
        # Buying Pressure = (Close - Low) * Volume
        # Selling Pressure = (High - Close) * Volume
        df[f"{asset}_buying_pressure"] = (close - low) * volume
        df[f"{asset}_selling_pressure"] = (high - close) * volume
        
        # 6. Oscillators
        stoch = StochRSIIndicator(close, window=14)
        df[f"{asset}_stoch_k"] = stoch.stochrsi_k()
        df[f"{asset}_stoch_d"] = stoch.stochrsi_d()
        
        df[f"{asset}_mfi"] = MFIIndicator(high, low, close, volume, window=14).money_flow_index()
        df[f"{asset}_williams_r"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        
        # 7. Velocity / Momentum
        df[f"{asset}_roc"] = ROCIndicator(close, window=12).roc()
        
        # 8. Price Action Context
        df[f"{asset}_gap_size"] = open_ - close.shift(1)
        
        # Close Location Value (CLV) = ((Close - Low) - (High - Close)) / (High - Low)
        # Values -1 to 1. -1 = Close at Low, 1 = Close at High.
        clv = ((close - low) - (high - close)) / range_.replace(0, 1e-9)
        df[f"{asset}_close_loc_value"] = clv
        
        return df

    def _add_volatility_features(self, df, asset):
        """
        Group 2: Volatility & Structure (For SL/TP Sizing)
        """
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        
        # 1. ATR & Regime
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_100 = AverageTrueRange(high, low, close, window=100).average_true_range()
        df[f"{asset}_atr_14"] = atr
        df[f"{asset}_atr_ratio"] = atr / atr_100.replace(0, 1e-9)
        df[f"{asset}_atr_slope"] = atr.diff(3)
        
        # 2. Bollinger Bands
        bb = BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        
        df[f"{asset}_bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, 1e-9)
        df[f"{asset}_bb_pct_b"] = bb.bollinger_pband()
        
        # 3. Standard Deviation
        df[f"{asset}_std_dev_20"] = close.rolling(20).std()
        
        # 4. Keltner Channels (Position)
        kc = KeltnerChannel(high, low, close, window=20)
        # Position 0..1 similar to BB %B
        kc_upper = kc.keltner_channel_hband()
        kc_lower = kc.keltner_channel_lband()
        df[f"{asset}_keltner_pos"] = (close - kc_lower) / (kc_upper - kc_lower).replace(0, 1e-9)
        
        # 5. Donchian Channels (Support/Resistance Distances)
        dc = DonchianChannel(high, low, close, window=50)
        dc_high = dc.donchian_channel_hband()
        dc_low = dc.donchian_channel_lband()
        
        # Normalized distance: (Level - Price) / ATR
        # Positive if level is above, Negative if below (for consistency)
        df[f"{asset}_donchian_high_dist"] = (dc_high - close) / atr.replace(0, 1e-9)
        df[f"{asset}_donchian_low_dist"] = (close - dc_low) / atr.replace(0, 1e-9)
        
        # 6. Pivot Points (Daily approximation using rolling window of ~288 5m bars)
        # We can't easily get true daily pivots without a daily timeframe, so we use a 24h rolling pivot.
        rolling_high_24h = high.rolling(288).max()
        rolling_low_24h = low.rolling(288).min()
        rolling_close_24h = close.shift(288) # Previous day close approximation
        pivot = (rolling_high_24h + rolling_low_24h + rolling_close_24h) / 3
        df[f"{asset}_pivot_dist"] = (close - pivot) / atr.replace(0, 1e-9)
        
        # 7. Range Analysis
        df[f"{asset}_avg_candle_range"] = (high - low).rolling(10).mean()
        
        # Parkinson Volatility (High-Low based)
        # Sum(ln(High/Low)^2) / (4 * ln(2))
        parkinson = np.sqrt((1.0 / (4.0 * np.log(2.0))) * np.log(high / low.replace(0, 1e-9))**2)
        df[f"{asset}_parkinson_vol"] = parkinson.rolling(20).mean()
        
        # Z-Score
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        df[f"{asset}_z_score"] = (close - sma_20) / std_20.replace(0, 1e-9)
        
        # Previous Day High/Low Dist (Approximated with 24h rolling)
        df[f"{asset}_prev_day_high_dist"] = (rolling_high_24h - close) / atr.replace(0, 1e-9)
        df[f"{asset}_prev_day_low_dist"] = (close - rolling_low_24h) / atr.replace(0, 1e-9)
        
        # Volatility Skew (Upside vol vs Downside vol)
        # Standard deviation of positive returns vs negative returns
        returns = close.pct_change()
        upside_vol = returns[returns > 0].rolling(50).std()
        downside_vol = returns[returns < 0].rolling(50).std()
        # FillNa is needed because rolling on filtered series creates gaps
        upside_vol = upside_vol.reindex(returns.index).ffill()
        downside_vol = downside_vol.reindex(returns.index).ffill()
        df[f"{asset}_vol_skew"] = upside_vol - downside_vol
        
        # Range %
        df[f"{asset}_high_low_range_pct"] = (high - low) / close
        
        # Shadow Symmetry (Upper wick / Lower wick)
        upper_wick = high - np.maximum(df[f"{asset}_open"], close)
        lower_wick = np.minimum(df[f"{asset}_open"], close) - low
        df[f"{asset}_shadow_symmetry"] = upper_wick / lower_wick.replace(0, 1e-9)
        
        return df

    def _add_trend_features(self, df, asset):
        """
        Group 3: Trend Strength (To Skip Trades)
        """
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        
        # 1. ADX & DIs
        adx_ind = ADXIndicator(high, low, close, window=14)
        df[f"{asset}_adx_14"] = adx_ind.adx()
        df[f"{asset}_di_plus"] = adx_ind.adx_pos()
        df[f"{asset}_di_minus"] = adx_ind.adx_neg()
        
        # 2. Aroon
        aroon = AroonIndicator(high, low, window=25)
        df[f"{asset}_aroon_up"] = aroon.aroon_up()
        df[f"{asset}_aroon_down"] = aroon.aroon_down()
        
        # 3. Moving Averages
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema21 = EMAIndicator(close, window=21).ema_indicator()
        ema50 = EMAIndicator(close, window=50).ema_indicator()
        ema200 = EMAIndicator(close, window=200).ema_indicator()
        
        df[f"{asset}_ema_9"] = ema9
        df[f"{asset}_ema_21"] = ema21
        df[f"{asset}_ema_50"] = ema50
        df[f"{asset}_ema_200"] = ema200
        
        # Distances
        df[f"{asset}_price_vs_ema50"] = (close - ema50) / ema50
        df[f"{asset}_price_vs_ema200"] = (close - ema200) / ema200
        
        # Ribbon Spread (Measure of trend expansion)
        df[f"{asset}_ema_ribbon_spread"] = (ema9 - ema50) / ema50
        
        # 4. SuperTrend (Implementation)
        # Basic SuperTrend logic:
        # ATR multiplier = 3, Period = 10
        # This is complex to vectorize efficiently in pandas without a loop or numba.
        # We will use a simplified vector approach or a library if available.
        # TA library doesn't have SuperTrend directly in all versions.
        # We will implement a simplified "Trend Signal" based on HL2 and ATR
        
        hl2 = (high + low) / 2
        atr = AverageTrueRange(high, low, close, window=10).average_true_range()
        # Simplified SuperTrend-like signal: 1 if close > prev_upper_band, -1 if close < prev_lower_band
        # For now, we'll use a placeholder or a simple trend check
        df[f"{asset}_supertrend"] = np.where(close > ema50, 1, -1) # Placeholder: EMA50 Trend
        
        # 5. Ichimoku Cloud Position (Simplified)
        # Conversion Line (9), Base Line (26)
        high_9 = high.rolling(9).max()
        low_9 = low.rolling(9).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        high_26 = high.rolling(26).max()
        low_26 = low.rolling(26).min()
        kijun_sen = (high_26 + low_26) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        high_52 = high.rolling(52).max()
        low_52 = low.rolling(52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        
        # 1 = Above Cloud, -1 = Below Cloud, 0 = Inside Cloud
        above_cloud = (close > senkou_span_a) & (close > senkou_span_b)
        below_cloud = (close < senkou_span_a) & (close < senkou_span_b)
        df[f"{asset}_cloud_pos"] = np.where(above_cloud, 1, np.where(below_cloud, -1, 0))
        
        # 6. Parabolic SAR (Using TA library if available, else simplified)
        # TA library has PSARIndicator
        from ta.trend import PSARIndicator
        psar = PSARIndicator(high, low, close).psar()
        # Position: 1 if Price > PSAR (Bullish), -1 if Price < PSAR (Bearish)
        df[f"{asset}_parabolic_sar_pos"] = np.where(close > psar, 1, -1)
        
        return df

    def _add_regime_features(self, df, asset):
        """
        Group 4: Market Regime & Context
        """
        # Time Features (Cyclical)
        # Note: These are identical for all assets, but we store them per asset for alignment
        # or we could store them once. For now, following the structure.
        
        df[f"{asset}_hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df[f"{asset}_hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df[f"{asset}_day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df[f"{asset}_day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        hours = df.index.hour
        # Sessions (UTC)
        df[f"{asset}_session_london"] = ((hours >= 8) & (hours < 17)).astype(int)
        df[f"{asset}_session_ny"] = ((hours >= 13) & (hours < 22)).astype(int)
        df[f"{asset}_session_asian"] = ((hours >= 0) & (hours < 9)).astype(int)
        
        # Efficiency Ratio (Kaufman)
        # Direction / Volatility
        change = df[f"{asset}_close"].diff(10).abs()
        volatility = df[f"{asset}_close"].diff().abs().rolling(10).sum()
        df[f"{asset}_efficiency_ratio"] = change / volatility.replace(0, 1e-9)
        
        # Hurst Exponent (Simplified Rolling)
        # A true rolling Hurst is expensive. We can use a proxy or a very short window.
        # Proxy: variance of longer window vs variance of shorter window
        # H = 0.5 * log(Var(2n)/Var(n)) / log(2)
        # Use returns to be stationary
        ret = df[f"{asset}_close"].pct_change()
        # std over 20 vs std over 10
        std_20 = ret.rolling(20).std()
        std_10 = ret.rolling(10).std()
        # simplified proxy for persistence
        df[f"{asset}_hurst_exp"] = std_20 / std_10.replace(0, 1e-9)
        
        # Fractal Dimension (Proxy)
        # Using Sevcik's method or similar simple approximation
        # We'll use a volatility-based proxy: 2 - H (from above)
        df[f"{asset}_fractal_dim"] = 2 - df[f"{asset}_hurst_exp"]
        
        return df

    def _add_alpha_features(self, df, asset, all_returns=None):
        """
        Group 5: Alpha Signals & Cross-Asset
        """
        close = df[f"{asset}_close"]
        ret = close.pct_change()
        
        # 1. Alpha Signal (Placeholder - Environment will overwrite this)
        if f"{asset}_alpha_signal" not in df.columns:
            df[f"{asset}_alpha_signal"] = 0.0
            df[f"{asset}_alpha_conf"] = 0.0
            
        # 2. Correlations
        if all_returns is None or all_returns.empty:
            # Fallback (slow)
            all_returns = pd.DataFrame()
            for a in self.assets:
                if f"{a}_close" in df.columns:
                    all_returns[a] = df[f"{a}_close"].pct_change()
        
        if not all_returns.empty:
            basket_return = all_returns.mean(axis=1)
            
            # Corr with Basket (USD Strength proxy if mostly USD pairs)
            df[f"{asset}_corr_usd"] = ret.rolling(50).corr(basket_return)
            
            # Relative Strength
            df[f"{asset}_rel_strength"] = ret - basket_return
            
            # Beta (Covariance / Variance of Market)
            cov = ret.rolling(50).cov(basket_return)
            var = basket_return.rolling(50).var()
            df[f"{asset}_beta"] = cov / var.replace(0, 1e-9)
            
            # Global Volatility (Mean of all assets' rolling std)
            global_vol = all_returns.rolling(20).std().mean(axis=1)
            df[f"{asset}_global_vol"] = global_vol
            
            # Global Trend (Mean of all assets' returns - crude proxy)
            df[f"{asset}_global_trend"] = all_returns.rolling(20).mean().mean(axis=1)
            
            # Sector Momentum (Mean of this asset + correlations > 0.5)
            # Simplified: just use basket for now
            df[f"{asset}_sector_mom"] = basket_return.rolling(20).mean()
            
        # Specific Correlations (Gold, Indices - require those columns to exist)
        # If XAUUSD is one of the assets, we can correlate with it.
        if "XAUUSD_close" in df.columns:
            xau_ret = all_returns["XAUUSD"] if "XAUUSD" in all_returns else df["XAUUSD_close"].pct_change()
            df[f"{asset}_corr_gold"] = ret.rolling(50).corr(xau_ret)
        else:
            df[f"{asset}_corr_gold"] = 0
            
        # SP500 would need to be in the data dict. Assuming 0 for now if missing.
        df[f"{asset}_corr_sp500"] = 0 # Placeholder
        
        # 3. Market Microstructure Proxies
        # Spread, Tick Vol (if available in raw data, else simulated/proxy)
        # We don't have Bid/Ask in standard OHLCV usually, unless provided.
        # We'll use High-Low as a volatility/spread proxy
        df[f"{asset}_spread"] = (df[f"{asset}_high"] - df[f"{asset}_low"]).rolling(5).mean()
        
        # Volume Variance
        df[f"{asset}_vol_variance"] = df[f"{asset}_volume"].rolling(20).var()
        df[f"{asset}_price_variance"] = close.rolling(20).var()
        
        # Covariance with Lead Asset (e.g. EURUSD as leader)
        if "EURUSD_close" in df.columns:
            eur_ret = all_returns["EURUSD"] if "EURUSD" in all_returns else df["EURUSD_close"].pct_change()
            df[f"{asset}_cov_lead"] = ret.rolling(50).cov(eur_ret)
        else:
            df[f"{asset}_cov_lead"] = 0
            
        # Tick Volume (using Volume as proxy)
        df[f"{asset}_tick_vol"] = df[f"{asset}_volume"]

        return df