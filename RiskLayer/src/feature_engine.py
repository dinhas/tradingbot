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
            "atr_14", "atr_ratio", "atr_percentile",
            "ema_9", "ema_21", "price_vs_ema9", "ema9_vs_ema21",
            "rsi_14", "macd_hist",
            "volume_ratio", "body_range_ratio",
            "spread", "spread_atr_ratio", "spread_expansion",
            "alpha_direction", "alpha_quality", "alpha_meta",
            "corr_basket", "rel_strength", "corr_xauusd", "corr_eurusd", "rank"
        ]
        
        # 2. Global Features (15)
        self.feature_names.extend([
            "market_regime", "volatility_regime", "global_momentum", "num_open_positions",
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
        # 1. Align DataFrames
        aligned_df = self._align_data(data_dict)
        
        # 2. Calculate Technical Indicators per Asset
        all_new_cols = {}
        for asset in self.assets:
            asset_cols = self._get_technical_indicators(aligned_df, asset)
            all_new_cols.update(asset_cols)
            
        # 3. Calculate Cross-Asset Features
        cross_asset_cols = self._get_cross_asset_features(aligned_df, all_new_cols)
        all_new_cols.update(cross_asset_cols)
        
        # 4. Add Global/Session Features
        session_cols = self._get_session_features(aligned_df, all_new_cols)
        all_new_cols.update(session_cols)

        # Concatenate all new features at once
        new_features_df = pd.DataFrame(all_new_cols, index=aligned_df.index).astype(np.float32)
        aligned_df = pd.concat([aligned_df, new_features_df], axis=1)
        
        # 5. Normalize Features (Robust Scaling)
        normalized_df = aligned_df.copy()
        normalized_df = self._normalize_features(normalized_df)
        
        # 6. Handle Missing Values
        normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)
        aligned_df = aligned_df.ffill().fillna(0).astype(np.float32)
        
        return aligned_df, normalized_df

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
            aligned_df = pd.concat(aligned_parts, axis=1)
        else:
            aligned_df = pd.DataFrame(index=common_index)
            
        return aligned_df

    def _get_technical_indicators(self, df, asset):
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        open_p = df[f"{asset}_open"]
        volume = df[f"{asset}_volume"]
        
        # Spread handling: look for column, else use default
        spread_col = f"{asset}_spread"
        if spread_col in df.columns:
            spread = df[spread_col]
        else:
            # Default spread simulation if not provided
            default_val = 0.3 if asset == 'XAUUSD' else 0.00002
            spread = pd.Series(default_val, index=df.index)
            
        new_cols = {}
        # Returns
        new_cols[f"{asset}_return_1"] = close.pct_change(1)
        new_cols[f"{asset}_return_12"] = close.pct_change(12)
        
        # Volatility
        atr_indicator = AverageTrueRange(high, low, close, window=14)
        atr = atr_indicator.average_true_range()
        atr_ma = atr.rolling(window=20).mean()
        new_cols[f"{asset}_atr_14"] = atr
        new_cols[f"{asset}_atr_ratio"] = atr / atr_ma
        
        # ATR Percentile (rolling 200)
        new_cols[f"{asset}_atr_percentile"] = atr.rolling(200).apply(lambda x: (x[-1] >= x).mean() if len(x) > 0 else 0)
        
        # Trend
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema21 = EMAIndicator(close, window=21).ema_indicator()
        new_cols[f"{asset}_ema_9"] = ema9
        new_cols[f"{asset}_ema_21"] = ema21
        new_cols[f"{asset}_price_vs_ema9"] = (close - ema9) / (ema9 + 1e-9)
        new_cols[f"{asset}_ema9_vs_ema21"] = (ema9 - ema21) / (ema21 + 1e-9)
        
        # Momentum
        new_cols[f"{asset}_rsi_14"] = RSIIndicator(close, window=14).rsi()
        macd = MACD(close)
        new_cols[f"{asset}_macd_hist"] = macd.macd_diff()
        
        # Volume
        vol_ma = volume.rolling(window=20).mean()
        new_cols[f"{asset}_volume_ratio"] = volume / (vol_ma + 1e-9)
        
        # Candle Body / Range Ratio
        body = (close - open_p).abs()
        candle_range = (high - low).abs()
        new_cols[f"{asset}_body_range_ratio"] = body / (candle_range + 1e-9)
        
        # Spread Features
        new_cols[f"{asset}_spread"] = spread
        new_cols[f"{asset}_spread_atr_ratio"] = spread / (atr + 1e-9)
        new_cols[f"{asset}_spread_expansion"] = spread.diff().rolling(20).mean() / (atr + 1e-9)
        
        # Alpha outputs (placeholders - will be populated by FeatureManager or generate_sl_dataset)
        new_cols[f"{asset}_alpha_direction"] = df.get(f"{asset}_alpha_direction", pd.Series(0, index=df.index))
        new_cols[f"{asset}_alpha_quality"] = df.get(f"{asset}_alpha_quality", pd.Series(0, index=df.index))
        new_cols[f"{asset}_alpha_meta"] = df.get(f"{asset}_alpha_meta", pd.Series(0, index=df.index))
        
        return new_cols

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
            
            new_cols[f"{asset}_corr_basket"] = asset_ret.rolling(50).corr(basket_return)
            new_cols[f"{asset}_rel_strength"] = asset_ret - basket_return
            
            xau_ret = technical_cols.get("XAUUSD_return_1", df.get("XAUUSD_return_1"))
            if xau_ret is not None:
                new_cols[f"{asset}_corr_xauusd"] = asset_ret.rolling(50).corr(xau_ret)
            else:
                new_cols[f"{asset}_corr_xauusd"] = pd.Series(0, index=df.index)
                
            eur_ret = technical_cols.get("EURUSD_return_1", df.get("EURUSD_return_1"))
            if eur_ret is not None:
                new_cols[f"{asset}_corr_eurusd"] = asset_ret.rolling(50).corr(eur_ret)
            else:
                new_cols[f"{asset}_corr_eurusd"] = pd.Series(0, index=df.index)
                
        # Asset Rank
        ret12_list = []
        for a in self.assets:
            ret12_list.append(technical_cols.get(f"{a}_return_12", df.get(f"{a}_return_12")))
        
        ret12_df = pd.concat(ret12_list, axis=1)
        ranks = ret12_df.rank(axis=1, ascending=False)
        for i, asset in enumerate(self.assets):
            new_cols[f"{asset}_rank"] = ranks.iloc[:, i]
            
        return new_cols

    def _get_session_features(self, df, technical_cols):
        new_cols = {}
        # Time features
        new_cols['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        new_cols['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        new_cols['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        new_cols['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Sessions (UTC)
        hours = df.index.hour
        new_cols['session_asian'] = ((hours >= 0) & (hours < 9)).astype(int)
        new_cols['session_london'] = ((hours >= 8) & (hours < 17)).astype(int)
        new_cols['session_ny'] = ((hours >= 13) & (hours < 22)).astype(int)
        new_cols['session_overlap'] = ((hours >= 13) & (hours < 17)).astype(int)
        
        # Global Market Regime Features
        gbp_ret = technical_cols.get("GBPUSD_return_1", df.get("GBPUSD_return_1", 0))
        xau_ret = technical_cols.get("XAUUSD_return_1", df.get("XAUUSD_return_1", 0))
        new_cols['risk_on_score'] = (gbp_ret + xau_ret) / 2
        
        # Asset dispersion
        ret_cols = []
        for a in self.assets:
            r = technical_cols.get(f"{a}_return_1", df.get(f"{a}_return_1"))
            if r is not None:
                ret_cols.append(r)
        
        if ret_cols:
            new_cols['asset_dispersion'] = pd.concat(ret_cols, axis=1).std(axis=1)
        else:
            new_cols['asset_dispersion'] = pd.Series(0, index=df.index)
            
        # Market volatility
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

    def _normalize_features(self, df):
        """Robust Scaling: (value - median) / IQR"""
        cols_to_normalize = []
        for asset in self.assets:
            cols_to_normalize.extend([
                f"{asset}_close", f"{asset}_return_1", f"{asset}_return_12",
                f"{asset}_atr_14", f"{asset}_atr_ratio", f"{asset}_atr_percentile",
                f"{asset}_ema_9", f"{asset}_ema_21", f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21",
                f"{asset}_rsi_14", f"{asset}_macd_hist",
                f"{asset}_volume_ratio", f"{asset}_body_range_ratio",
                f"{asset}_spread", f"{asset}_spread_atr_ratio", f"{asset}_spread_expansion"
            ])
            # Note: Alpha outputs are typically already normalized [0,1] or similar, 
            # but we can apply robust scaling if they are raw logits. 
            # Here we leave them as is or add them if needed.
            
        # Add Global Features to normalization
        cols_to_normalize.extend(['risk_on_score', 'asset_dispersion', 'market_volatility'])
            
        for col in cols_to_normalize:
            if col in df.columns:
                rolling_median = df[col].rolling(window=50).median()
                rolling_q75 = df[col].rolling(window=50).quantile(0.75)
                rolling_q25 = df[col].rolling(window=50).quantile(0.25)
                iqr = rolling_q75 - rolling_q25
                
                iqr = iqr.replace(0, 1e-6)
                df[col] = (df[col] - rolling_median) / iqr
                df[col] = df[col].clip(-5, 5)
        
        return df

    def get_observation_vectorized(self, df, asset):
        """Vectorized version of observation extraction for dataset generation."""
        obs_cols = []
        # Per-Asset (25)
        obs_cols.extend([f"{asset}_close", f"{asset}_return_1", f"{asset}_return_12"])
        obs_cols.extend([f"{asset}_atr_14", f"{asset}_atr_ratio", f"{asset}_atr_percentile"])
        obs_cols.extend([f"{asset}_ema_9", f"{asset}_ema_21", f"{asset}_price_vs_ema9", f"{asset}_ema9_vs_ema21"])
        obs_cols.extend([f"{asset}_rsi_14", f"{asset}_macd_hist", f"{asset}_volume_ratio", f"{asset}_body_range_ratio"])
        obs_cols.extend([f"{asset}_spread", f"{asset}_spread_atr_ratio", f"{asset}_spread_expansion"])
        obs_cols.extend([f"{asset}_alpha_direction", f"{asset}_alpha_quality", f"{asset}_alpha_meta"])
        obs_cols.extend([f"{asset}_corr_basket", f"{asset}_rel_strength", f"{asset}_corr_xauusd", f"{asset}_corr_eurusd", f"{asset}_rank"])
        
        # Global (15)
        obs_cols.extend([
            "risk_on_score", "asset_dispersion", "market_volatility", "num_open_positions",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "session_asian", "session_london", "session_ny", "session_overlap"
        ])
        # Add 3 placeholders for market_regime, volatility_regime, global_momentum if not calculated
        for c in ["market_regime", "volatility_regime", "global_momentum"]:
            if c not in df.columns:
                df[c] = 0
            obs_cols.append(c)
            
        return df.reindex(columns=obs_cols, fill_value=0).values.astype(np.float32)

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
        obs.extend([
            current_step_data.get(f"{asset}_close", 0),
            current_step_data.get(f"{asset}_return_1", 0),
            current_step_data.get(f"{asset}_return_12", 0),
            current_step_data.get(f"{asset}_atr_14", 0),
            current_step_data.get(f"{asset}_atr_ratio", 0),
            current_step_data.get(f"{asset}_atr_percentile", 0),
            current_step_data.get(f"{asset}_ema_9", 0),
            current_step_data.get(f"{asset}_ema_21", 0),
            current_step_data.get(f"{asset}_price_vs_ema9", 0),
            current_step_data.get(f"{asset}_ema9_vs_ema21", 0),
            current_step_data.get(f"{asset}_rsi_14", 0),
            current_step_data.get(f"{asset}_macd_hist", 0),
            current_step_data.get(f"{asset}_volume_ratio", 0),
            current_step_data.get(f"{asset}_body_range_ratio", 0),
            current_step_data.get(f"{asset}_spread", 0),
            current_step_data.get(f"{asset}_spread_atr_ratio", 0),
            current_step_data.get(f"{asset}_spread_expansion", 0),
            # These are typically passed in portfolio_state or as separate keys in current_step_data
            current_step_data.get(f"{asset}_alpha_direction", portfolio_state.get('alpha_direction', 0)),
            current_step_data.get(f"{asset}_alpha_quality", portfolio_state.get('alpha_quality', 0)),
            current_step_data.get(f"{asset}_alpha_meta", portfolio_state.get('alpha_meta', 0)),
            current_step_data.get(f"{asset}_corr_basket", 0),
            current_step_data.get(f"{asset}_rel_strength", 0),
            current_step_data.get(f"{asset}_corr_xauusd", 0),
            current_step_data.get(f"{asset}_corr_eurusd", 0),
            current_step_data.get(f"{asset}_rank", 0)
        ])

        # 2. Global Features (15)
        obs.extend([
            current_step_data.get('risk_on_score', 0),
            current_step_data.get('asset_dispersion', 0),
            current_step_data.get('market_volatility', 0),
            portfolio_state.get('num_open_positions', 0),
            current_step_data.get('hour_sin', 0),
            current_step_data.get('hour_cos', 0),
            current_step_data.get('day_sin', 0),
            current_step_data.get('day_cos', 0),
            current_step_data.get('session_asian', 0),
            current_step_data.get('session_london', 0),
            current_step_data.get('session_ny', 0),
            current_step_data.get('session_overlap', 0),
            current_step_data.get('market_regime', 0),
            current_step_data.get('volatility_regime', 0),
            current_step_data.get('global_momentum', 0)
        ])
        
        return np.array(obs, dtype=np.float32)
