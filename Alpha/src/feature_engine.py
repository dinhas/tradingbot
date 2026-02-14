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
        """Defines the list of 40 features (25 asset-specific + 15 global/market)."""
        # 1. Per-Asset Features (29 -> 25 kept + 11 new = 36? No, wait.)
        # Let's count properly based on the plan.
        # EXISTING KEPT (25? No, let's look at previous list)
        # Previous list had 25 per asset + 15 global = 40 total? No, that's not right.
        # standard obs was 40 dims.
        # Let's align with the user request: "TOTAL FEATURES REMAINS AT 40"
        
        # New Feature Set:
        # A. Multi-Timeframe (3)
        # B. Volume & Order Flow (3)
        # C. Price Action Structure (3)
        # D. Momentum Edge (2)
        # Total New = 11
        
        # Existing to Keep (29)
        # Price: close, return_1, return_12 (3)
        # Volatility: atr_14, atr_ratio, bb_position (3)
        # Trend: ema_9, ema_21, price_vs_ema9, ema9_vs_ema21 (4)
        # Momentum: rsi_14, macd_hist (2)
        # Volume: volume_ratio (1)
        # Cross-Asset: corr_basket, rel_strength, corr_xauusd, corr_eurusd, rank (5)
        # Global/Time: risk_on_score, asset_dispersion, market_volatility (3)
        # Time: hour_sin, hour_cos, day_sin, day_cos (4)
        # Session: session_asian, session_london, session_ny, session_overlap (4)
        
        # Total = 11 (new) + 29 (kept) = 40. Perfect.
        
        self.feature_names = [
            # --- EXISTING (13) ---
            "close", "return_1", "return_12",
            "atr_14", "atr_ratio", "bb_position",
            "ema_9", "ema_21", "price_vs_ema9", "ema9_vs_ema21",
            "rsi_14", "macd_hist",
            "volume_ratio",
            
            # --- NEW PRO FEATURES (11) ---
            "htf_ema_alignment", "htf_rsi_divergence", "swing_structure_proximity", # HTF (3)
            "vwap_deviation", "delta_pressure", "volume_shock",                     # Volume (3)
            "volatility_squeeze", "wick_rejection_strength", "breakout_velocity",   # PA (3)
            "rsi_slope_divergence", "macd_momentum_quality",                        # Momentum (2)
            
            # --- CROSS-ASSET (5) ---
            "corr_basket", "rel_strength", "corr_xauusd", "corr_eurusd", "rank",
            
            # --- GLOBAL / TIME (11) ---
            "risk_on_score", "asset_dispersion", "market_volatility",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "session_asian", "session_london", "session_ny", "session_overlap"
        ]

    def preprocess_data(self, data_dict):
        """
        Preprocesses raw OHLCV data for all assets.
        Args:
            data_dict: Dictionary {asset_name: pd.DataFrame} with columns [open, high, low, close, volume]
        Returns:
            pd.DataFrame: Aligned DataFrame with all features calculated.
        """
        logger = logging.getLogger(__name__)
        
        # 1. Align DataFrames
        logger.info("Aligning data for all assets...")
        aligned_df = self._align_data(data_dict)
        
        # Convert to float32 immediately to save 50% RAM
        for col in aligned_df.columns:
            aligned_df[col] = aligned_df[col].astype(np.float32)
        
        # 2. Calculate Technical Indicators per Asset
        for asset in self.assets:
            logger.info(f"Calculating technical indicators for {asset}...")
            aligned_df = self._add_technical_indicators(aligned_df, asset)
            # Ensure new columns are also float32
            for col in aligned_df.columns:
                if aligned_df[col].dtype == np.float64:
                    aligned_df[col] = aligned_df[col].astype(np.float32)
            
        # 3. Calculate Cross-Asset Features
        logger.info("Calculating cross-asset features...")
        aligned_df = self._add_cross_asset_features(aligned_df)
        
        # 4. Add Global/Session Features
        logger.info("Adding session and global features...")
        aligned_df = self._add_session_features(aligned_df)
        
        # 5. Normalize Features (Robust Scaling)
        logger.info("Normalizing features (this may take a minute)...")
        # Create a copy for the normalized observation set
        # We'll create a normalized version for the agent, and keep raw for env simulation if needed.
        # But usually TradingEnv wants the processed_data for observation.
        
        normalized_df = aligned_df.copy() # One copy is likely needed for the normalized observation set
        # But ensure it's float32
        for col in normalized_df.columns:
            normalized_df[col] = normalized_df[col].astype(np.float32)
            
        normalized_df = self._normalize_features(normalized_df)
        
        # 6. Handle Missing Values
        normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)
        aligned_df = aligned_df.ffill().fillna(0).astype(np.float32)
        
        logger.info(f"Preprocessing complete. Total features: {len(normalized_df.columns)}")
        return aligned_df, normalized_df

    def _align_data(self, data_dict):
        """Aligns all asset DataFrames to the same index."""
        # Find common index range
        common_index = None
        for df in data_dict.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        aligned_df = pd.DataFrame(index=common_index)
        for asset, df in data_dict.items():
            # Rename columns to include asset prefix
            df_subset = df.loc[common_index].copy()
            df_subset.columns = [f"{asset}_{col}" for col in df_subset.columns]
            aligned_df = pd.concat([aligned_df, df_subset], axis=1)
            
        return aligned_df

    def _add_technical_indicators(self, df, asset):
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        volume = df[f"{asset}_volume"]
        
        # Returns
        df[f"{asset}_return_1"] = close.pct_change(1)
        df[f"{asset}_return_12"] = close.pct_change(12)
        
        # Volatility
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_ma = atr.rolling(window=20).mean()
        df[f"{asset}_atr_14"] = atr
        df[f"{asset}_atr_ratio"] = atr / atr_ma
        
        bb = BollingerBands(close, window=20, window_dev=2)
        df[f"{asset}_bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Trend
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema21 = EMAIndicator(close, window=21).ema_indicator()
        df[f"{asset}_ema_9"] = ema9
        df[f"{asset}_ema_21"] = ema21
        df[f"{asset}_price_vs_ema9"] = (close - ema9) / ema9
        df[f"{asset}_ema9_vs_ema21"] = (ema9 - ema21) / ema21
        
        # Momentum
        df[f"{asset}_rsi_14"] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        df[f"{asset}_macd_hist"] = macd.macd_diff()
        
        # Volume
        vol_ma = volume.rolling(window=20).mean()
        df[f"{asset}_volume_ratio"] = volume / vol_ma
        
        # --- NEW PRO FEATURES ---
        df = self._add_pro_features(df, asset)
        
        return df

    def _add_pro_features(self, df, asset):
        """Calculates 11 new substitution features."""
        # Pre-fetch series for speed
        close = df[f"{asset}_close"]
        open_ = df[f"{asset}_open"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        volume = df[f"{asset}_volume"]
        atr = df[f"{asset}_atr_14"]
        rsi = df[f"{asset}_rsi_14"]
        macd_hist = df[f"{asset}_macd_hist"]
        
        # A. MULTI-TIMEFRAME CONFLUENCE
        # 1. HTF EMA Alignment (Proxy 1H EMA21 on 5M data - resampled)
        # Using simple resampling to avoid lookahead bias (closed bars only)
        close_1h = close.resample('60min', label='right', closed='right').last().reindex(close.index, method='ffill')
        ema21_1h = close_1h.ewm(span=21, adjust=False).mean()
        df[f"{asset}_htf_ema_alignment"] = (close - ema21_1h) / atr
        
        # 2. HTF RSI Divergence
        rsi_1h = RSIIndicator(close_1h, window=14).rsi().reindex(close.index, method='ffill')
        df[f"{asset}_htf_rsi_divergence"] = rsi - rsi_1h
        
        # 3. Swing Structure Proximity (40 bars ~ 3.3 hours)
        swing_high = high.rolling(40).max()
        swing_low = low.rolling(40).min()
        # Distance to nearest swing point
        dist_high = (swing_high - close) / atr
        dist_low = (close - swing_low) / atr
        df[f"{asset}_swing_structure_proximity"] = np.minimum(dist_high, dist_low)

        # B. VOLUME & ORDER FLOW
        # 4. VWAP Deviation
        # VWAP reset daily? Or rolling? Snippet implies cumulative. Let's strictly follow snippet but safe for rolling.
        # Snippet: (close * volume).cumsum() / volume.cumsum() -> This is lifetime VWAP. 
        # Better: Session VWAP (reset at 00:00 UTC) or Rolling VWAP.
        # Given snippet `cumsum()`, likely implies session or infinite. Infinite is bad for long datasets.
        # Let's use Rolling VWAP (1 day = 288 bars) to match "session" feel without complexity of resets, or just session reset.
        # Integrating Session VWAP properly:
        vwap_series = (close * volume).groupby(df.index.date).cumsum() / volume.groupby(df.index.date).cumsum()
        df[f"{asset}_vwap_deviation"] = (close - vwap_series) / atr
        
        # 5. Delta Pressure (Buying vs Selling Volume Proxy)
        # Money Flow approx: (Close - Open) * Volume?? Snippet: sign(close-open)
        direction = np.sign(close - open_)
        money_flow = volume * direction
        pressure = money_flow.rolling(20).sum() / volume.rolling(20).sum()
        df[f"{asset}_delta_pressure"] = pressure.fillna(0)
        
        # 6. Volume Shock
        vol_mean_20 = volume.rolling(20).mean()
        df[f"{asset}_volume_shock"] = np.log(volume / (vol_mean_20 + 1e-6))
        
        # C. PRICE ACTION STRUCTURE
        # 7. Volatility Squeeze (BB Width / ATR)
        bb = BollingerBands(close, window=20, window_dev=2)
        # bb_w = bb.bollinger_hwidth() - bb.bollinger_lwidth() # REMOVED: Incorrect attribute
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband())
        df[f"{asset}_volatility_squeeze"] = bb_width / atr
        
        # 8. Wick Rejection Strength (Wicks / Body)
        body = (close - open_).abs()
        upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
        total_wick = upper_wick + lower_wick
        rejection = (total_wick / (body + 1e-6)).rolling(3).mean()
        df[f"{asset}_wick_rejection_strength"] = rejection
        
        # 9. Breakout Velocity
        # Speed of penetration beyond 20-bar range
        range_h = high.rolling(20).max()
        range_l = low.rolling(20).min()
        # Logic: If close > range_h, how far?
        # Note: shift(1) needed for range to be "prior" range? 
        # Snippet doesn't shift, implies current breakout. 
        velocity = np.where(close > range_h.shift(1), (close - range_h.shift(1))/atr,
                           np.where(close < range_l.shift(1), (close - range_l.shift(1))/atr, 0))
        df[f"{asset}_breakout_velocity"] = velocity
        
        # D. MOMENTUM EDGE
        # 10. RSI Slope Divergence
        # Sign(Price Slope) - Sign(RSI Slope)
        price_slope = close.diff(10)
        rsi_slope = rsi.diff(10)
        df[f"{asset}_rsi_slope_divergence"] = np.sign(price_slope) - np.sign(rsi_slope)
        
        # 11. MACD Momentum Quality
        # (Hist - Hist_prev) * Sign(Signal)
        # Acceleration of histogram aligning with trend
        macd_signal = MACD(close).macd_signal()
        hist_slope = macd_hist.diff()
        df[f"{asset}_macd_momentum_quality"] = hist_slope * np.sign(macd_signal)
        
        return df

    def _add_cross_asset_features(self, df):
        # Calculate basket return (average of all assets)
        returns = df[[f"{a}_return_1" for a in self.assets]]
        basket_return = returns.mean(axis=1)
        
        for asset in self.assets:
            asset_ret = df[f"{asset}_return_1"]
            
            # Correlation to basket (50 period)
            df[f"{asset}_corr_basket"] = asset_ret.rolling(50).corr(basket_return)
            
            # Relative Strength
            df[f"{asset}_rel_strength"] = asset_ret - basket_return
            
            # Correlation to XAUUSD
            if f"XAUUSD_return_1" in df.columns:
                df[f"{asset}_corr_xauusd"] = asset_ret.rolling(50).corr(df["XAUUSD_return_1"])
            else:
                df[f"{asset}_corr_xauusd"] = 0
                
            # Correlation to EURUSD
            if f"EURUSD_return_1" in df.columns:
                df[f"{asset}_corr_eurusd"] = asset_ret.rolling(50).corr(df["EURUSD_return_1"])
            else:
                df[f"{asset}_corr_eurusd"] = 0
                
        # Asset Rank (1-5 based on return_12)
        ret12_cols = [f"{a}_return_12" for a in self.assets]
        ranks = df[ret12_cols].rank(axis=1, ascending=False)
        for asset, col in zip(self.assets, ret12_cols):
            df[f"{asset}_rank"] = ranks[col]
            
        return df

    def _add_session_features(self, df):
        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Sessions (UTC)
        # Asian: 00:00 - 09:00
        # London: 08:00 - 17:00
        # NY: 13:00 - 22:00
        # Overlap: 13:00 - 17:00
        
        hours = df.index.hour
        df['session_asian'] = ((hours >= 0) & (hours < 9)).astype(int)
        df['session_london'] = ((hours >= 8) & (hours < 17)).astype(int)
        df['session_ny'] = ((hours >= 13) & (hours < 22)).astype(int)
        df['session_overlap'] = ((hours >= 13) & (hours < 17)).astype(int)
        
        # Global Market Regime Features (Pre-calculated)
        # Risk on score: (GBPUSD_return + XAUUSD_return) / 2
        # Handle missing columns gracefully if assets are different
        gbp_ret = df["GBPUSD_return_1"] if "GBPUSD_return_1" in df.columns else 0
        xau_ret = df["XAUUSD_return_1"] if "XAUUSD_return_1" in df.columns else 0
        df['risk_on_score'] = (gbp_ret + xau_ret) / 2
        
        # Asset dispersion: std of returns across all assets
        ret_cols = [f"{a}_return_1" for a in self.assets if f"{a}_return_1" in df.columns]
        if ret_cols:
            df['asset_dispersion'] = df[ret_cols].std(axis=1)
        else:
            df['asset_dispersion'] = 0
            
        # Market volatility: mean ATR ratio
        atr_ratio_cols = [f"{a}_atr_ratio" for a in self.assets if f"{a}_atr_ratio" in df.columns]
        if atr_ratio_cols:
            df['market_volatility'] = df[atr_ratio_cols].mean(axis=1)
        else:
            df['market_volatility'] = 0
        
        return df

    def _normalize_features(self, df):
        # Robust Scaling: (value - median) / IQR
        # We use rolling statistics to avoid lookahead bias
        
        cols_to_normalize = []
        for asset in self.assets:
            cols_to_normalize.extend([
                f"{asset}_close", f"{asset}_ema_9", f"{asset}_ema_21",
                f"{asset}_return_1", f"{asset}_return_12",
                f"{asset}_atr_14", f"{asset}_atr_ratio",
                f"{asset}_rsi_14", f"{asset}_macd_hist",
                f"{asset}_volume_ratio",
                # New Features
                f"{asset}_htf_ema_alignment", f"{asset}_htf_rsi_divergence", f"{asset}_swing_structure_proximity",
                f"{asset}_vwap_deviation", f"{asset}_delta_pressure", f"{asset}_volume_shock",
                f"{asset}_volatility_squeeze", f"{asset}_wick_rejection_strength", f"{asset}_breakout_velocity",
                f"{asset}_rsi_slope_divergence", f"{asset}_macd_momentum_quality"
            ])
            
        # Add Global Features to normalization
        cols_to_normalize.extend(['risk_on_score', 'asset_dispersion', 'market_volatility'])
            
        for col in cols_to_normalize:
            if col in df.columns:
                rolling_median = df[col].rolling(window=50).median()
                rolling_q75 = df[col].rolling(window=50).quantile(0.75)
                rolling_q25 = df[col].rolling(window=50).quantile(0.25)
                iqr = rolling_q75 - rolling_q25
                
                # Avoid division by zero
                iqr = iqr.replace(0, 1e-6)
                
                df[col] = (df[col] - rolling_median) / iqr
                
                # Clip outliers
                df[col] = df[col].clip(-5, 5)
                
        # Bounded features (0-1 or similar) don't need robust scaling, 
        # but we can center them or scale them to [-1, 1] if needed.
        # For now, we leave them as is or apply simple scaling.
        
        return df

    def get_observation(self, current_step_data, portfolio_state, asset):
        """
        Constructs the 40-feature observation vector for a single asset.
        Args:
            current_step_data: Row of preprocessed DataFrame for current timestamp.
            portfolio_state: Dictionary containing portfolio metrics (UNUSED NOW - kept for API compat).
            asset: The specific asset to get features for.
        Returns:
            np.array: 40-dimensional vector.
        """
        obs = []
        
        # --- 1. EXISTING FEATURES (13) ---
        # Price (3)
        obs.extend([
            current_step_data.get(f"{asset}_close", 0),
            current_step_data.get(f"{asset}_return_1", 0),
            current_step_data.get(f"{asset}_return_12", 0)
        ])
        # Volatility (3)
        obs.extend([
            current_step_data.get(f"{asset}_atr_14", 0),
            current_step_data.get(f"{asset}_atr_ratio", 0),
            current_step_data.get(f"{asset}_bb_position", 0)
        ])
        # Trend (4)
        obs.extend([
            current_step_data.get(f"{asset}_ema_9", 0),
            current_step_data.get(f"{asset}_ema_21", 0),
            current_step_data.get(f"{asset}_price_vs_ema9", 0),
            current_step_data.get(f"{asset}_ema9_vs_ema21", 0)
        ])
        # Momentum (2)
        obs.extend([
            current_step_data.get(f"{asset}_rsi_14", 0),
            current_step_data.get(f"{asset}_macd_hist", 0)
        ])
        # Volume (1)
        obs.append(current_step_data.get(f"{asset}_volume_ratio", 0))
        
        # --- 2. NEW PRO FEATURES (11) ---
        obs.extend([
            current_step_data.get(f"{asset}_htf_ema_alignment", 0),
            current_step_data.get(f"{asset}_htf_rsi_divergence", 0),
            current_step_data.get(f"{asset}_swing_structure_proximity", 0),
            current_step_data.get(f"{asset}_vwap_deviation", 0),
            current_step_data.get(f"{asset}_delta_pressure", 0),
            current_step_data.get(f"{asset}_volume_shock", 0),
            current_step_data.get(f"{asset}_volatility_squeeze", 0),
            current_step_data.get(f"{asset}_wick_rejection_strength", 0),
            current_step_data.get(f"{asset}_breakout_velocity", 0),
            current_step_data.get(f"{asset}_rsi_slope_divergence", 0),
            current_step_data.get(f"{asset}_macd_momentum_quality", 0)
        ])

        # --- 3. CROSS-ASSET (5) ---
        obs.extend([
            current_step_data.get(f"{asset}_corr_basket", 0),
            current_step_data.get(f"{asset}_rel_strength", 0),
            current_step_data.get(f"{asset}_corr_xauusd", 0),
            current_step_data.get(f"{asset}_corr_eurusd", 0),
            current_step_data.get(f"{asset}_rank", 0)
        ])

        # --- 4. GLOBAL / TIME (11) ---
        # Market Regime (3)
        gbp_ret = current_step_data.get("GBPUSD_return_1", 0)
        xau_ret = current_step_data.get("XAUUSD_return_1", 0)
        risk_on = (gbp_ret + xau_ret) / 2
        
        returns = [current_step_data.get(f"{a}_return_1", 0) for a in self.assets]
        dispersion = np.std(returns) if returns else 0
        
        atrs = [current_step_data.get(f"{a}_atr_ratio", 0) for a in self.assets]
        mkt_vol = np.mean(atrs) if atrs else 0
        
        obs.extend([risk_on, dispersion, mkt_vol])
        
        # Session (8)
        obs.extend([
            current_step_data.get('hour_sin', 0),
            current_step_data.get('hour_cos', 0),
            current_step_data.get('day_sin', 0),
            current_step_data.get('day_cos', 0),
            current_step_data.get('session_asian', 0),
            current_step_data.get('session_london', 0),
            current_step_data.get('session_ny', 0),
            current_step_data.get('session_overlap', 0)
        ])
        
        return np.array(obs, dtype=np.float32)
