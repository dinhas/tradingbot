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
        """Defines the list of 40 features (25 asset-specific + 15 global)."""
        # 1. Per-Asset Features (25)
        self.feature_names = [
            "close", "return_1", "return_12",
            "atr_14", "atr_ratio", "bb_position",
            "ema_9", "ema_21", "price_vs_ema9", "ema9_vs_ema21",
            "rsi_14", "macd_hist",
            "volume_ratio",
            "has_position", "position_size", "unrealized_pnl",
            "position_age", "entry_price", "current_sl", "current_tp",
            "corr_basket", "rel_strength", "corr_xauusd", "corr_eurusd", "rank"
        ]
        
        # 2. Global Features (15)
        self.feature_names.extend([
            "equity", "margin_usage_pct", "drawdown", "num_open_positions",
            "risk_on_score", "asset_dispersion", "market_volatility",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "session_asian", "session_london", "session_ny", "session_overlap"
        ])

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
        # Do NOT copy, just pass the reference and modify if needed or create small temp arrays
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
        new_features = {}
        
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        volume = df[f"{asset}_volume"]
        
        # Returns
        new_features[f"{asset}_return_1"] = close.pct_change(1)
        new_features[f"{asset}_return_12"] = close.pct_change(12)
        
        # Volatility
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_ma = atr.rolling(window=20).mean()
        new_features[f"{asset}_atr_14"] = atr
        new_features[f"{asset}_atr_ratio"] = atr / atr_ma
        
        bb = BollingerBands(close, window=20, window_dev=2)
        new_features[f"{asset}_bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Trend
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema21 = EMAIndicator(close, window=21).ema_indicator()
        new_features[f"{asset}_ema_9"] = ema9
        new_features[f"{asset}_ema_21"] = ema21
        new_features[f"{asset}_price_vs_ema9"] = (close - ema9) / ema9
        new_features[f"{asset}_ema9_vs_ema21"] = (ema9 - ema21) / ema21
        
        # Momentum
        new_features[f"{asset}_rsi_14"] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        new_features[f"{asset}_macd_hist"] = macd.macd_diff()
        
        # Volume
        vol_ma = volume.rolling(window=20).mean()
        new_features[f"{asset}_volume_ratio"] = volume / vol_ma
        
        new_features_df = pd.DataFrame(new_features, index=df.index).astype(np.float32)
        return pd.concat([df, new_features_df], axis=1)

    def _add_cross_asset_features(self, df):
        new_features = {}
        
        # Calculate basket return (average of all assets)
        # We need to access return_1 which might have just been added
        returns_cols = [f"{a}_return_1" for a in self.assets if f"{a}_return_1" in df.columns]
        if not returns_cols:
             return df
             
        basket_return = df[returns_cols].mean(axis=1)
        
        for asset in self.assets:
            if f"{asset}_return_1" not in df.columns: continue
            
            asset_ret = df[f"{asset}_return_1"]
            
            # Correlation to basket (50 period)
            new_features[f"{asset}_corr_basket"] = asset_ret.rolling(50).corr(basket_return)
            
            # Relative Strength
            new_features[f"{asset}_rel_strength"] = asset_ret - basket_return
            
            # Correlation to XAUUSD
            if f"XAUUSD_return_1" in df.columns:
                new_features[f"{asset}_corr_xauusd"] = asset_ret.rolling(50).corr(df["XAUUSD_return_1"])
            else:
                new_features[f"{asset}_corr_xauusd"] = pd.Series(0, index=df.index)
                
            # Correlation to EURUSD
            if f"EURUSD_return_1" in df.columns:
                new_features[f"{asset}_corr_eurusd"] = asset_ret.rolling(50).corr(df["EURUSD_return_1"])
            else:
                new_features[f"{asset}_corr_eurusd"] = pd.Series(0, index=df.index)
                
        # Asset Rank (1-5 based on return_12)
        ret12_cols = [f"{a}_return_12" for a in self.assets if f"{a}_return_12" in df.columns]
        if ret12_cols:
            ranks = df[ret12_cols].rank(axis=1, ascending=False)
            for asset, col in zip(self.assets, ret12_cols):
                 # ranks columns match ret12_cols order
                 # ranks is a DF, so ranks[col] gives the rank series for that asset
                 new_features[f"{asset}_rank"] = ranks[col]
            
        new_features_df = pd.DataFrame(new_features, index=df.index).astype(np.float32)
        return pd.concat([df, new_features_df], axis=1)

    def _add_session_features(self, df):
        new_features = {}
        
        # Time features
        new_features['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        new_features['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        new_features['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        new_features['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Sessions (UTC)
        hours = df.index.hour
        new_features['session_asian'] = ((hours >= 0) & (hours < 9)).astype(int)
        new_features['session_london'] = ((hours >= 8) & (hours < 17)).astype(int)
        new_features['session_ny'] = ((hours >= 13) & (hours < 22)).astype(int)
        new_features['session_overlap'] = ((hours >= 13) & (hours < 17)).astype(int)
        
        # Global Market Regime Features
        gbp_ret = df["GBPUSD_return_1"] if "GBPUSD_return_1" in df.columns else pd.Series(0, index=df.index)
        xau_ret = df["XAUUSD_return_1"] if "XAUUSD_return_1" in df.columns else pd.Series(0, index=df.index)
        new_features['risk_on_score'] = (gbp_ret + xau_ret) / 2
        
        # Asset dispersion
        ret_cols = [f"{a}_return_1" for a in self.assets if f"{a}_return_1" in df.columns]
        if ret_cols:
            new_features['asset_dispersion'] = df[ret_cols].std(axis=1)
        else:
            new_features['asset_dispersion'] = pd.Series(0, index=df.index)
            
        # Market volatility
        atr_ratio_cols = [f"{a}_atr_ratio" for a in self.assets if f"{a}_atr_ratio" in df.columns]
        if atr_ratio_cols:
            new_features['market_volatility'] = df[atr_ratio_cols].mean(axis=1)
        else:
            new_features['market_volatility'] = pd.Series(0, index=df.index)
        
        new_features_df = pd.DataFrame(new_features, index=df.index).astype(np.float32)
        return pd.concat([df, new_features_df], axis=1)

    def _normalize_features(self, df):
        # Robust Scaling: (value - median) / IQR
        # To avoid fragmentation here, we can calculate all normalized series and then 
        # replace the columns or create a new dataframe. 
        # However, modifying columns in place IS NOT fragmentation if columns already exist.
        # Fragmentation happens when INSERTING new columns repeatedly.
        # Here we are updating existing columns.
        
        cols_to_normalize = []
        for asset in self.assets:
            cols_to_normalize.extend([
                f"{asset}_close", f"{asset}_ema_9", f"{asset}_ema_21",
                f"{asset}_return_1", f"{asset}_return_12",
                f"{asset}_atr_14", f"{asset}_atr_ratio",
                f"{asset}_rsi_14", f"{asset}_macd_hist",
                f"{asset}_volume_ratio"
            ])
            
        cols_to_normalize.extend(['risk_on_score', 'asset_dispersion', 'market_volatility'])
        
        # We can do this in a vectorized way if we select the subset
        # But rolling operations are per-series usually or per-dataframe.
        # Let's check if we can optimize. 
        # Ideally, we calculate all rolling stats at once if possible, but they are different per column.
        
        # Since we are modifying existing columns, fragmentation warning shouldn't trigger 
        # UNLESS we are doing Copy-on-Write which pandas 2.0+ might do or if memory layout is weird.
        # But the warning specifically mentions "frame.insert". 
        # So in-place updates should be fine.
        
        # However, to be safe and cleaner, we can iterate.
        # If the warning persists here, we'd need to reconstruct the DF.
        
        # Let's keep loop but ensure we are not adding new columns, just updating.
        
        for col in cols_to_normalize:
            if col in df.columns:
                series = df[col]
                rolling_median = series.rolling(window=50).median()
                rolling_q75 = series.rolling(window=50).quantile(0.75)
                rolling_q25 = series.rolling(window=50).quantile(0.25)
                iqr = rolling_q75 - rolling_q25
                
                iqr = iqr.replace(0, 1e-6)
                
                # In-place update
                df[col] = ((series - rolling_median) / iqr).clip(-5, 5)
                
        return df

    def get_observation(self, current_step_data, portfolio_state, asset):
        """
        Constructs the 40-feature observation vector for a single asset.
        Args:
            current_step_data: Row of preprocessed DataFrame for current timestamp.
            portfolio_state: Dictionary containing portfolio metrics.
            asset: The specific asset to get features for.
        Returns:
            np.array: 40-dimensional vector.
        """
        obs = []
        
        # 1. Per-Asset Features (25)
        # Price Action & Returns (3)
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
        
        # Position State (7)
        asset_state = portfolio_state.get(asset, {})
        obs.extend([
            asset_state.get('has_position', 0),
            asset_state.get('position_size', 0),
            asset_state.get('unrealized_pnl', 0),
            asset_state.get('position_age', 0),
            asset_state.get('entry_price', 0),
            asset_state.get('current_sl', 0),
            asset_state.get('current_tp', 0)
        ])
        # Cross-Asset (5)
        obs.extend([
            current_step_data.get(f"{asset}_corr_basket", 0),
            current_step_data.get(f"{asset}_rel_strength", 0),
            current_step_data.get(f"{asset}_corr_xauusd", 0),
            current_step_data.get(f"{asset}_corr_eurusd", 0),
            current_step_data.get(f"{asset}_rank", 0)
        ])

        # 2. Global Features (15)
        # Portfolio State (4)
        obs.extend([
            portfolio_state.get('equity', 0),
            portfolio_state.get('margin_usage_pct', 0),
            portfolio_state.get('drawdown', 0),
            portfolio_state.get('num_open_positions', 0)
        ])
        
        # Market Regime (3)
        gbp_ret = current_step_data.get("GBPUSD_return_1", 0)
        xau_ret = current_step_data.get("XAUUSD_return_1", 0)
        risk_on = (gbp_ret + xau_ret) / 2
        
        returns = [current_step_data.get(f"{a}_return_1", 0) for a in self.assets]
        dispersion = np.std(returns)
        
        atrs = [current_step_data.get(f"{a}_atr_ratio", 0) for a in self.assets]
        mkt_vol = np.mean(atrs)
        
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
