import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands

# ── Feature count constant ──
# Used across the pipeline to size observation matrices.
NUM_FEATURES = 28

class FeatureEngine:
    def __init__(self):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.num_features = NUM_FEATURES
        self.feature_names = []
        self._define_feature_names()

    def _define_feature_names(self):
        """
        28 features: high-signal, low-noise.
        
        Removed all tick-volume features (useless in forex),
        removed redundant raw prices, removed noisy rolling correlations,
        added candlestick structure and ADX trend strength.
        """
        self.feature_names = [
            # --- PRICE ACTION (7) ---
            "return_1", "return_5", "return_12",
            "close_vs_range", "body_ratio",
            "upper_wick_ratio", "lower_wick_ratio",

            # --- VOLATILITY (4) ---
            "atr_14", "atr_ratio", "bb_position", "atr_percentile",

            # --- TREND / MOMENTUM (6) ---
            "price_vs_ema9", "ema9_vs_ema21",
            "rsi_14", "macd_hist_norm", "adx_14", "rsi_momentum",

            # --- HIGHER TIMEFRAME (3) ---
            "htf_ema_alignment", "htf_rsi", "htf_trend_strength",

            # --- TIME / SESSION (5) ---
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "session_overlap",

            # --- CROSS-ASSET / MARKET (3) ---
            "rel_strength", "market_volatility", "asset_dispersion"
        ]

    # ──────────────────────────────────────────────────────────────
    #  Main preprocessing entry point
    # ──────────────────────────────────────────────────────────────
    def preprocess_data(self, data_dict):
        """
        Preprocesses raw OHLCV data for all assets.
        Args:
            data_dict: Dictionary {asset_name: pd.DataFrame} with columns [open, high, low, close, volume]
        Returns:
            (aligned_df, normalized_df)
        """
        logger = logging.getLogger(__name__)

        # 1. Align DataFrames
        logger.info("Aligning data for all assets...")
        aligned_df = self._align_data(data_dict)

        # Convert to float32 immediately to save 50% RAM
        for col in aligned_df.columns:
            aligned_df[col] = aligned_df[col].astype(np.float32)

        # 2. Calculate Technical Indicators per Asset
        all_new_cols = {}
        for asset in self.assets:
            logger.info(f"Calculating technical indicators for {asset}...")
            asset_cols = self._get_technical_indicators(aligned_df, asset)
            all_new_cols.update(asset_cols)

        # 3. Calculate Cross-Asset Features
        logger.info("Calculating cross-asset features...")
        cross_asset_cols = self._get_cross_asset_features(aligned_df, all_new_cols)
        all_new_cols.update(cross_asset_cols)

        # 4. Add Global/Session Features
        logger.info("Adding session and global features...")
        session_cols = self._get_session_features(aligned_df, all_new_cols)
        all_new_cols.update(session_cols)

        # Concatenate all new features at once
        new_features_df = pd.DataFrame(all_new_cols, index=aligned_df.index).astype(np.float32)
        aligned_df = pd.concat([aligned_df, new_features_df], axis=1)

        # 5. Normalize Features (Robust Scaling)
        logger.info("Normalizing features (this may take a minute)...")
        normalized_df = aligned_df.copy()
        normalized_df = self._normalize_features(normalized_df)

        # 6. Handle Missing Values
        normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)
        aligned_df = aligned_df.ffill().fillna(0).astype(np.float32)

        logger.info(f"Preprocessing complete. Total features: {len(normalized_df.columns)}")
        return aligned_df, normalized_df

    # ──────────────────────────────────────────────────────────────
    #  Data alignment
    # ──────────────────────────────────────────────────────────────
    def _align_data(self, data_dict):
        """Aligns all asset DataFrames to the same index."""
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

        if aligned_parts:
            return pd.concat(aligned_parts, axis=1)
        return pd.DataFrame(index=common_index)

    # ──────────────────────────────────────────────────────────────
    #  Per-asset technical indicators
    # ──────────────────────────────────────────────────────────────
    def _get_technical_indicators(self, df, asset):
        close = df[f"{asset}_close"]
        open_ = df[f"{asset}_open"]
        high  = df[f"{asset}_high"]
        low   = df[f"{asset}_low"]

        new_cols = {}

        # ── A. PRICE ACTION (7) ──
        new_cols[f"{asset}_return_1"]  = close.pct_change(1)
        new_cols[f"{asset}_return_5"]  = close.pct_change(5)
        new_cols[f"{asset}_return_12"] = close.pct_change(12)

        bar_range = high - low
        bar_range_safe = bar_range.replace(0, np.nan)

        new_cols[f"{asset}_close_vs_range"] = (close - low) / bar_range_safe
        new_cols[f"{asset}_body_ratio"] = (close - open_).abs() / bar_range_safe

        upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
        lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low
        new_cols[f"{asset}_upper_wick_ratio"] = upper_wick / bar_range_safe
        new_cols[f"{asset}_lower_wick_ratio"] = lower_wick / bar_range_safe

        # ── B. VOLATILITY (4) ──
        atr_indicator = AverageTrueRange(high, low, close, window=14)
        atr = atr_indicator.average_true_range()
        atr_ma = atr.rolling(window=20).mean()
        new_cols[f"{asset}_atr_14"] = atr
        new_cols[f"{asset}_atr_ratio"] = atr / atr_ma

        bb = BollingerBands(close, window=20, window_dev=2)
        new_cols[f"{asset}_bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

        # ATR percentile: where current ATR sits in rolling 200-bar history
        atr_min_200 = atr.rolling(200).min()
        atr_max_200 = atr.rolling(200).max()
        new_cols[f"{asset}_atr_percentile"] = (atr - atr_min_200) / (atr_max_200 - atr_min_200 + 1e-9)

        # ── C. TREND / MOMENTUM (6) ──
        ema9  = EMAIndicator(close, window=9).ema_indicator()
        ema21 = EMAIndicator(close, window=21).ema_indicator()
        new_cols[f"{asset}_price_vs_ema9"]  = (close - ema9) / ema9
        new_cols[f"{asset}_ema9_vs_ema21"] = (ema9 - ema21) / ema21

        rsi = RSIIndicator(close, window=14).rsi()
        new_cols[f"{asset}_rsi_14"] = rsi

        macd_hist = MACD(close).macd_diff()
        new_cols[f"{asset}_macd_hist_norm"] = macd_hist / (atr + 1e-9)  # Normalized by ATR

        adx = ADXIndicator(high, low, close, window=14).adx()
        new_cols[f"{asset}_adx_14"] = adx

        rsi_ma = rsi.rolling(14).mean()
        new_cols[f"{asset}_rsi_momentum"] = rsi - rsi_ma

        # ── D. HIGHER TIMEFRAME (3) ──
        close_1h = close.resample('60min', label='right', closed='right').last().reindex(close.index, method='ffill')

        ema21_1h = close_1h.ewm(span=21, adjust=False).mean()
        new_cols[f"{asset}_htf_ema_alignment"] = (close - ema21_1h) / (atr + 1e-9)

        rsi_1h = RSIIndicator(close_1h, window=14).rsi().reindex(close.index, method='ffill')
        new_cols[f"{asset}_htf_rsi"] = rsi_1h

        ema21_1h_slope = ema21_1h.diff(3)
        new_cols[f"{asset}_htf_trend_strength"] = ema21_1h_slope / (atr + 1e-9)

        return new_cols

    # ──────────────────────────────────────────────────────────────
    #  Cross-asset features (simplified from 5 → 1 per asset)
    # ──────────────────────────────────────────────────────────────
    def _get_cross_asset_features(self, df, technical_cols):
        new_cols = {}
        returns_list = []
        for a in self.assets:
            if f"{a}_return_1" in technical_cols:
                returns_list.append(technical_cols[f"{a}_return_1"])
            else:
                returns_list.append(df[f"{a}_return_1"])

        returns_df = pd.concat(returns_list, axis=1)
        basket_return = returns_df.mean(axis=1)

        for asset in self.assets:
            asset_ret = technical_cols.get(f"{asset}_return_1", df.get(f"{asset}_return_1"))
            new_cols[f"{asset}_rel_strength"] = asset_ret - basket_return

        return new_cols

    # ──────────────────────────────────────────────────────────────
    #  Session and global features
    # ──────────────────────────────────────────────────────────────
    def _get_session_features(self, df, technical_cols):
        new_cols = {}

        # Time features
        new_cols['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        new_cols['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        new_cols['day_sin']  = np.sin(2 * np.pi * df.index.dayofweek / 7)
        new_cols['day_cos']  = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Only the highest-signal session indicator: London/NY overlap
        hours = df.index.hour
        new_cols['session_overlap'] = ((hours >= 13) & (hours < 17)).astype(int)

        # Market-wide measures
        ret_cols = []
        for a in self.assets:
            r = technical_cols.get(f"{a}_return_1", df.get(f"{a}_return_1"))
            if r is not None:
                ret_cols.append(r)

        if ret_cols:
            new_cols['asset_dispersion'] = pd.concat(ret_cols, axis=1).std(axis=1)
        else:
            new_cols['asset_dispersion'] = pd.Series(0, index=df.index)

        atr_ratio_cols = []
        for a in self.assets:
            r = technical_cols.get(f"{a}_atr_ratio", df.get(f"{a}_atr_ratio"))
            if r is not None:
                atr_ratio_cols.append(r)

        if atr_ratio_cols:
            new_cols['market_volatility'] = pd.concat(atr_ratio_cols, axis=1).mean(axis=1)
        else:
            new_cols['market_volatility'] = pd.Series(0, index=df.index)

        return new_cols

    # ──────────────────────────────────────────────────────────────
    #  Normalization (robust scaling)
    # ──────────────────────────────────────────────────────────────
    def _normalize_features(self, df):
        """Robust Scaling: (value - median) / IQR using rolling window to avoid lookahead."""

        cols_to_normalize = []
        for asset in self.assets:
            cols_to_normalize.extend([
                # Price action (returns, ratios already bounded but benefit from centering)
                f"{asset}_return_1", f"{asset}_return_5", f"{asset}_return_12",
                # Volatility
                f"{asset}_atr_14", f"{asset}_atr_ratio",
                # Trend/Momentum
                f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21",
                f"{asset}_rsi_14", f"{asset}_macd_hist_norm",
                f"{asset}_adx_14", f"{asset}_rsi_momentum",
                # Higher Timeframe
                f"{asset}_htf_ema_alignment", f"{asset}_htf_rsi",
                f"{asset}_htf_trend_strength",
                # Cross-asset
                f"{asset}_rel_strength",
            ])

        # Global features to normalize
        cols_to_normalize.extend(['asset_dispersion', 'market_volatility'])

        for col in cols_to_normalize:
            if col in df.columns:
                rolling_median = df[col].rolling(window=50).median()
                rolling_q75 = df[col].rolling(window=50).quantile(0.75)
                rolling_q25 = df[col].rolling(window=50).quantile(0.25)
                iqr = rolling_q75 - rolling_q25
                iqr = iqr.replace(0, 1e-6)
                df[col] = (df[col] - rolling_median) / iqr
                df[col] = df[col].clip(-5, 5)

        # Features already bounded [0, 1] — no robust scaling needed:
        # bb_position, atr_percentile, close_vs_range, body_ratio,
        # upper_wick_ratio, lower_wick_ratio, session_overlap,
        # hour_sin, hour_cos, day_sin, day_cos

        return df

    # ──────────────────────────────────────────────────────────────
    #  Observation extraction (vectorized, for batch processing)
    # ──────────────────────────────────────────────────────────────
    def get_observation_vectorized(self, df, asset):
        """
        Efficiently extracts the 28-feature observation matrix for all rows.
        Returns: np.array of shape (len(df), 28)
        """
        obs_cols = []

        # --- 1. PRICE ACTION (7) ---
        obs_cols.extend([
            f"{asset}_return_1", f"{asset}_return_5", f"{asset}_return_12",
            f"{asset}_close_vs_range", f"{asset}_body_ratio",
            f"{asset}_upper_wick_ratio", f"{asset}_lower_wick_ratio",
        ])

        # --- 2. VOLATILITY (4) ---
        obs_cols.extend([
            f"{asset}_atr_14", f"{asset}_atr_ratio",
            f"{asset}_bb_position", f"{asset}_atr_percentile",
        ])

        # --- 3. TREND / MOMENTUM (6) ---
        obs_cols.extend([
            f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21",
            f"{asset}_rsi_14", f"{asset}_macd_hist_norm",
            f"{asset}_adx_14", f"{asset}_rsi_momentum",
        ])

        # --- 4. HIGHER TIMEFRAME (3) ---
        obs_cols.extend([
            f"{asset}_htf_ema_alignment", f"{asset}_htf_rsi",
            f"{asset}_htf_trend_strength",
        ])

        # --- 5. TIME / SESSION (5) ---
        obs_cols.extend([
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'session_overlap',
        ])

        # --- 6. CROSS-ASSET / MARKET (3) ---
        obs_cols.extend([
            f"{asset}_rel_strength",
            'market_volatility', 'asset_dispersion',
        ])

        # Extract and handle missing columns with zeros
        existing_cols = [c for c in obs_cols if c in df.columns]
        missing_cols = [c for c in obs_cols if c not in df.columns]

        if missing_cols:
            data = df.reindex(columns=obs_cols, fill_value=0).values
        else:
            data = df[existing_cols].values

        return data.astype(np.float32)

    def get_observation(self, current_step_data, portfolio_state, asset):
        """
        Constructs the 28-feature observation vector for a single asset.
        Args:
            current_step_data: Row of preprocessed DataFrame for current timestamp.
            portfolio_state: Dictionary containing portfolio metrics (UNUSED - kept for API compat).
            asset: The specific asset to get features for.
        Returns:
            np.array: 28-dimensional vector.
        """
        obs = []

        # --- 1. PRICE ACTION (7) ---
        obs.extend([
            current_step_data.get(f"{asset}_return_1", 0),
            current_step_data.get(f"{asset}_return_5", 0),
            current_step_data.get(f"{asset}_return_12", 0),
            current_step_data.get(f"{asset}_close_vs_range", 0),
            current_step_data.get(f"{asset}_body_ratio", 0),
            current_step_data.get(f"{asset}_upper_wick_ratio", 0),
            current_step_data.get(f"{asset}_lower_wick_ratio", 0),
        ])

        # --- 2. VOLATILITY (4) ---
        obs.extend([
            current_step_data.get(f"{asset}_atr_14", 0),
            current_step_data.get(f"{asset}_atr_ratio", 0),
            current_step_data.get(f"{asset}_bb_position", 0),
            current_step_data.get(f"{asset}_atr_percentile", 0),
        ])

        # --- 3. TREND / MOMENTUM (6) ---
        obs.extend([
            current_step_data.get(f"{asset}_price_vs_ema9", 0),
            current_step_data.get(f"{asset}_ema9_vs_ema21", 0),
            current_step_data.get(f"{asset}_rsi_14", 0),
            current_step_data.get(f"{asset}_macd_hist_norm", 0),
            current_step_data.get(f"{asset}_adx_14", 0),
            current_step_data.get(f"{asset}_rsi_momentum", 0),
        ])

        # --- 4. HIGHER TIMEFRAME (3) ---
        obs.extend([
            current_step_data.get(f"{asset}_htf_ema_alignment", 0),
            current_step_data.get(f"{asset}_htf_rsi", 0),
            current_step_data.get(f"{asset}_htf_trend_strength", 0),
        ])

        # --- 5. TIME / SESSION (5) ---
        obs.extend([
            current_step_data.get('hour_sin', 0),
            current_step_data.get('hour_cos', 0),
            current_step_data.get('day_sin', 0),
            current_step_data.get('day_cos', 0),
            current_step_data.get('session_overlap', 0),
        ])

        # --- 6. CROSS-ASSET / MARKET (3) ---
        obs.extend([
            current_step_data.get(f"{asset}_rel_strength", 0),
            current_step_data.get('market_volatility', 0),
            current_step_data.get('asset_dispersion', 0),
        ])

        return np.array(obs, dtype=np.float32)
