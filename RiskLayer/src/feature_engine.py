import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

class FeatureEngine:
    def __init__(self):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.feature_names = []
        self._define_feature_names()

    def _define_feature_names(self):
        """Defines the list of 140 features for reference."""
        # Per-Asset Features (25 * 5 = 125)
        for asset in self.assets:
            # Price Action (3)
            self.feature_names.extend([f"{asset}_close", f"{asset}_return_1", f"{asset}_return_12"])
            # Volatility (3)
            self.feature_names.extend([f"{asset}_atr_14", f"{asset}_atr_ratio", f"{asset}_bb_position"])
            # Trend (4)
            self.feature_names.extend([f"{asset}_ema_9", f"{asset}_ema_21", f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21"])
            # Momentum (2)
            self.feature_names.extend([f"{asset}_rsi_14", f"{asset}_macd_hist"])
            # Volume (1)
            self.feature_names.extend([f"{asset}_volume_ratio"])
            # Position State (7)
            self.feature_names.extend([
                f"{asset}_has_position", f"{asset}_position_size", f"{asset}_unrealized_pnl",
                f"{asset}_position_age", f"{asset}_entry_price", f"{asset}_current_sl", f"{asset}_current_tp"
            ])
            # Cross-Asset (5)
            self.feature_names.extend([
                f"{asset}_corr_basket", f"{asset}_rel_strength", f"{asset}_corr_xauusd",
                f"{asset}_corr_eurusd", f"{asset}_rank"
            ])

        # Global Features (15)
        # Portfolio State (4)
        self.feature_names.extend(["equity", "margin_usage_pct", "drawdown", "num_open_positions"])
        # Market Regime (3)
        self.feature_names.extend(["risk_on_score", "asset_dispersion", "market_volatility"])
        # Session (8)
        self.feature_names.extend([
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
        # 1. Align DataFrames
        aligned_df = self._align_data(data_dict)
        
        # 2. Calculate Technical Indicators per Asset
        for asset in self.assets:
            aligned_df = self._add_technical_indicators(aligned_df, asset)
            
        # 3. Calculate Cross-Asset Features
        aligned_df = self._add_cross_asset_features(aligned_df)
        
        # 4. Add Global/Session Features
        aligned_df = self._add_session_features(aligned_df)
        
        # 5. Normalize Features (Robust Scaling)
        # Create a copy for normalization to preserve raw values
        normalized_df = aligned_df.copy()
        normalized_df = self._normalize_features(normalized_df)
        
        # 6. Handle Missing Values
        normalized_df = normalized_df.ffill().fillna(0)
        aligned_df = aligned_df.ffill().fillna(0)
        
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
                f"{asset}_volume_ratio"
            ])
            
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

    def get_observation(self, current_step_data, portfolio_state):
        """
        Constructs the 140-feature observation vector for a single step.
        Args:
            current_step_data: Row of preprocessed DataFrame for current timestamp.
            portfolio_state: Dictionary containing portfolio metrics.
        Returns:
            np.array: 140-dimensional vector.
        """
        obs = []
        
        # 1. Per-Asset Features
        for asset in self.assets:
            # From Data
            obs.extend([
                current_step_data.get(f"{asset}_close", 0),
                current_step_data.get(f"{asset}_return_1", 0),
                current_step_data.get(f"{asset}_return_12", 0),
                current_step_data.get(f"{asset}_atr_14", 0),
                current_step_data.get(f"{asset}_atr_ratio", 0),
                current_step_data.get(f"{asset}_bb_position", 0),
                current_step_data.get(f"{asset}_ema_9", 0),
                current_step_data.get(f"{asset}_ema_21", 0),
                current_step_data.get(f"{asset}_price_vs_ema9", 0),
                current_step_data.get(f"{asset}_ema9_vs_ema21", 0),
                current_step_data.get(f"{asset}_rsi_14", 0),
                current_step_data.get(f"{asset}_macd_hist", 0),
                current_step_data.get(f"{asset}_volume_ratio", 0)
            ])
            
            # From Portfolio State (Asset specific)
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
            
            # From Data (Cross-Asset)
            obs.extend([
                current_step_data.get(f"{asset}_corr_basket", 0),
                current_step_data.get(f"{asset}_rel_strength", 0),
                current_step_data.get(f"{asset}_corr_xauusd", 0),
                current_step_data.get(f"{asset}_corr_eurusd", 0),
                current_step_data.get(f"{asset}_rank", 0)
            ])
            
        # 2. Global Features
        # From Portfolio State (Global)
        obs.extend([
            portfolio_state.get('equity', 0),
            portfolio_state.get('margin_usage_pct', 0),
            portfolio_state.get('drawdown', 0),
            portfolio_state.get('num_open_positions', 0)
        ])
        
        # From Data (Market Regime & Session)
        # Risk on score: (GBPUSD_return + XAUUSD_return) / 2
        gbp_ret = current_step_data.get("GBPUSD_return_1", 0)
        xau_ret = current_step_data.get("XAUUSD_return_1", 0)
        risk_on = (gbp_ret + xau_ret) / 2
        
        # Asset dispersion: std of returns
        returns = [current_step_data.get(f"{a}_return_1", 0) for a in self.assets]
        dispersion = np.std(returns)
        
        # Market volatility: mean ATR ratio
        atrs = [current_step_data.get(f"{a}_atr_ratio", 0) for a in self.assets]
        mkt_vol = np.mean(atrs)
        
        obs.extend([risk_on, dispersion, mkt_vol])
        
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
