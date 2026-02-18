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
        
        # Typical spreads (simulated)
        self.typical_spreads = {
            'EURUSD': 0.00008,
            'GBPUSD': 0.00012,
            'USDJPY': 0.012,
            'USDCHF': 0.00015,
            'XAUUSD': 0.25
        }

    def _define_feature_names(self):
        """Defines the list of 48 features for the Risk Model."""
        # 1. Base Alpha-style Features (40)
        self.base_features = [
            "close", "return_1", "return_12",
            "atr_14", "atr_ratio", "bb_position",
            "ema_9", "ema_21", "price_vs_ema9", "ema9_vs_ema21",
            "rsi_14", "macd_hist", "volume_ratio",
            "htf_ema_alignment", "htf_rsi_divergence", "swing_structure_proximity",
            "vwap_deviation", "delta_pressure", "volume_shock",
            "volatility_squeeze", "wick_rejection_strength", "breakout_velocity",
            "rsi_slope_divergence", "macd_momentum_quality",
            "corr_basket", "rel_strength", "corr_xauusd", "corr_eurusd", "rank",
            "risk_on_score", "asset_dispersion", "market_volatility",
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "session_asian", "session_london", "session_ny", "session_overlap"
        ]
        
        # 2. Risk Specific Enhanced Features (8)
        self.risk_features = [
            "alpha_direction", "alpha_meta", "alpha_quality",
            "spread", "spread_atr_ratio", "atr_percentile",
            "wick_body_ratio", "spread_expansion"
        ]
        
        self.feature_names = self.base_features + self.risk_features

    def preprocess_data(self, data_dict):
        """
        Preprocesses raw OHLCV data for all assets.
        """
        logger = logging.getLogger(__name__)
        
        # 1. Align DataFrames
        aligned_df = self._align_data(data_dict)
        
        # 2. Calculate Technical Indicators per Asset
        all_new_cols = {}
        for asset in self.assets:
            logger.info(f"Calculating features for {asset}...")
            asset_cols = self._get_technical_indicators(aligned_df, asset)
            all_new_cols.update(asset_cols)
            
        # 3. Calculate Cross-Asset Features
        cross_asset_cols = self._get_cross_asset_features(aligned_df, all_new_cols)
        all_new_cols.update(cross_asset_cols)
        
        # 4. Add Global/Session Features
        session_cols = self._get_session_features(aligned_df, all_new_cols)
        all_new_cols.update(session_cols)

        # Concatenate all new features
        new_features_df = pd.DataFrame(all_new_cols, index=aligned_df.index).astype(np.float32)
        aligned_df = pd.concat([aligned_df, new_features_df], axis=1)
        
        # 5. Normalize Features
        normalized_df = aligned_df.copy()
        normalized_df = self._normalize_features(normalized_df)
        
        # 6. Final Clean
        normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)
        aligned_df = aligned_df.ffill().fillna(0).astype(np.float32)
        
        return aligned_df, normalized_df

    def _align_data(self, data_dict):
        common_index = None
        for df in data_dict.values():
            if common_index is None: common_index = df.index
            else: common_index = common_index.intersection(df.index)
        
        aligned_parts = []
        for asset, df in data_dict.items():
            df_subset = df.loc[common_index].copy()
            df_subset.columns = [f"{asset}_{col}" for col in df_subset.columns]
            aligned_parts.append(df_subset)
            
        return pd.concat(aligned_parts, axis=1)

    def _get_technical_indicators(self, df, asset):
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        open_ = df[f"{asset}_open"]
        volume = df[f"{asset}_volume"]
        
        new_cols = {}
        # Basic
        new_cols[f"{asset}_return_1"] = close.pct_change(1)
        new_cols[f"{asset}_return_12"] = close.pct_change(12)
        
        atr_indicator = AverageTrueRange(high, low, close, window=14)
        atr = atr_indicator.average_true_range()
        new_cols[f"{asset}_atr_14"] = atr
        new_cols[f"{asset}_atr_ratio"] = atr / atr.rolling(20).mean()
        
        bb = BollingerBands(close, window=20, window_dev=2)
        new_cols[f"{asset}_bb_position"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
        
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema21 = EMAIndicator(close, window=21).ema_indicator()
        new_cols[f"{asset}_ema_9"] = ema9
        new_cols[f"{asset}_ema_21"] = ema21
        new_cols[f"{asset}_price_vs_ema9"] = (close - ema9) / (ema9 + 1e-9)
        new_cols[f"{asset}_ema9_vs_ema21"] = (ema9 - ema21) / (ema21 + 1e-9)
        
        rsi = RSIIndicator(close, window=14).rsi()
        new_cols[f"{asset}_rsi_14"] = rsi
        macd_hist = MACD(close).macd_diff()
        new_cols[f"{asset}_macd_hist"] = macd_hist
        new_cols[f"{asset}_volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)
        
        # Pro Features
        pro = self._calculate_pro_features(df, asset, new_cols)
        new_cols.update(pro)
        
        # Risk Specific
        spread_val = self.typical_spreads.get(asset, 0.0001)
        # Simulate some spread volatility based on ATR
        sim_spread = spread_val * (1.0 + 0.2 * (new_cols[f"{asset}_atr_ratio"] - 1.0))
        new_cols[f"{asset}_spread"] = sim_spread
        new_cols[f"{asset}_spread_atr_ratio"] = sim_spread / (atr + 1e-9)
        
        # ATR Percentile (rolling 200)
        new_cols[f"{asset}_atr_percentile"] = atr.rolling(200).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 200 else 0.5)
        
        body = (close - open_).abs()
        range_ = high - low
        new_cols[f"{asset}_wick_body_ratio"] = (range_ - body) / (body + 1e-9)
        new_cols[f"{asset}_spread_expansion"] = sim_spread.pct_change(5)
        
        # Placeholders for Alpha Outputs (filled during dataset gen)
        new_cols[f"{asset}_alpha_direction"] = 0.0
        new_cols[f"{asset}_alpha_meta"] = 0.0
        new_cols[f"{asset}_alpha_quality"] = 0.0
        
        return new_cols

    def _calculate_pro_features(self, df, asset, technical_cols):
        close = df[f"{asset}_close"]
        open_ = df[f"{asset}_open"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        volume = df[f"{asset}_volume"]
        atr = technical_cols[f"{asset}_atr_14"]
        rsi = technical_cols[f"{asset}_rsi_14"]
        macd_hist = technical_cols[f"{asset}_macd_hist"]
        
        new_features = {}
        
        # Simplified HTF for speed
        ema21_12 = close.ewm(span=21*12, adjust=False).mean() # Approx 1h on 5m
        new_features[f"{asset}_htf_ema_alignment"] = (close - ema21_12) / (atr + 1e-9)
        
        rsi_12 = RSIIndicator(close, window=14*12).rsi()
        new_features[f"{asset}_htf_rsi_divergence"] = rsi - rsi_12
        
        swing_high = high.rolling(40).max()
        swing_low = low.rolling(40).min()
        new_features[f"{asset}_swing_structure_proximity"] = np.minimum((swing_high - close), (close - swing_low)) / (atr + 1e-9)

        vwap = (close * volume).cumsum() / volume.cumsum() # Simplified
        new_features[f"{asset}_vwap_deviation"] = (close - vwap) / (atr + 1e-9)
        
        direction = np.sign(close - open_)
        new_features[f"{asset}_delta_pressure"] = (volume * direction).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
        new_features[f"{asset}_volume_shock"] = np.log(volume / (volume.rolling(20).mean() + 1e-9) + 1e-6)
        
        bb = BollingerBands(close, window=20, window_dev=2)
        new_features[f"{asset}_volatility_squeeze"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (atr + 1e-9)
        
        body = (close - open_).abs()
        wicks = (high - low) - body
        new_features[f"{asset}_wick_rejection_strength"] = wicks / (body + 1e-9)
        
        range_h = high.rolling(20).max().shift(1)
        new_features[f"{asset}_breakout_velocity"] = np.where(close > range_h, (close - range_h)/atr, 0)
        
        new_features[f"{asset}_rsi_slope_divergence"] = close.diff(10).apply(np.sign) - rsi.diff(10).apply(np.sign)
        new_features[f"{asset}_macd_momentum_quality"] = macd_hist.diff() * np.sign(macd_hist)
        
        return new_features

    def _get_cross_asset_features(self, df, technical_cols):
        new_cols = {}
        rets = pd.concat([technical_cols[f"{a}_return_1"] for a in self.assets], axis=1)
        basket_ret = rets.mean(axis=1)
        
        for asset in self.assets:
            new_cols[f"{asset}_corr_basket"] = technical_cols[f"{asset}_return_1"].rolling(50).corr(basket_ret)
            new_cols[f"{asset}_rel_strength"] = technical_cols[f"{asset}_return_1"] - basket_ret
            new_cols[f"{asset}_corr_xauusd"] = technical_cols[f"{asset}_return_1"].rolling(50).corr(technical_cols["XAUUSD_return_1"])
            new_cols[f"{asset}_corr_eurusd"] = technical_cols[f"{asset}_return_1"].rolling(50).corr(technical_cols["EURUSD_return_1"])
            
        ret12_df = pd.concat([technical_cols[f"{a}_return_12"] for a in self.assets], axis=1)
        ranks = ret12_df.rank(axis=1, ascending=False)
        for i, asset in enumerate(self.assets):
            new_cols[f"{asset}_rank"] = ranks.iloc[:, i]
        return new_cols

    def _get_session_features(self, df, technical_cols):
        new_cols = {}
        h = df.index.hour
        new_cols['hour_sin'] = np.sin(2 * np.pi * h / 24)
        new_cols['hour_cos'] = np.cos(2 * np.pi * h / 24)
        new_cols['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        new_cols['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        new_cols['session_asian'] = ((h >= 0) & (h < 9)).astype(float)
        new_cols['session_london'] = ((h >= 8) & (h < 17)).astype(float)
        new_cols['session_ny'] = ((h >= 13) & (h < 22)).astype(float)
        new_cols['session_overlap'] = ((h >= 13) & (h < 17)).astype(float)
        
        new_cols['risk_on_score'] = (technical_cols["GBPUSD_return_1"] + technical_cols["XAUUSD_return_1"]) / 2
        rets = pd.concat([technical_cols[f"{a}_return_1"] for a in self.assets], axis=1)
        new_cols['asset_dispersion'] = rets.std(axis=1)
        atrs = pd.concat([technical_cols[f"{a}_atr_ratio"] for a in self.assets], axis=1)
        new_cols['market_volatility'] = atrs.mean(axis=1)
        return new_cols

    def _normalize_features(self, df):
        cols = []
        for asset in self.assets:
            cols.extend([f"{asset}_{f}" for f in self.base_features if f not in self._get_global_names()])
            cols.extend([f"{asset}_{f}" for f in self.risk_features])
        cols.extend(self._get_global_names())
        
        for col in cols:
            if col in df.columns:
                # Use faster rolling median/iqr
                roll = df[col].rolling(window=100)
                m = roll.median()
                q25 = roll.quantile(0.25)
                q75 = roll.quantile(0.75)
                iqr = q75 - q25 + 1e-9
                df[col] = ((df[col] - m) / iqr).clip(-5, 5)
        return df

    def _get_global_names(self):
        return ["risk_on_score", "asset_dispersion", "market_volatility", "hour_sin", "hour_cos", "day_sin", "day_cos", "session_asian", "session_london", "session_ny", "session_overlap"]

    def get_observation_vectorized(self, df, asset):
        obs_cols = []
        for name in self.feature_names:
            if name in self._get_global_names(): obs_cols.append(name)
            else: obs_cols.append(f"{asset}_{name}")
        return df.reindex(columns=obs_cols, fill_value=0).values.astype(np.float32)

    def get_observation(self, current_step_data, portfolio_state, asset):
        obs = []
        for name in self.feature_names:
            if name in self._get_global_names(): val = current_step_data.get(name, 0)
            else: val = current_step_data.get(f"{asset}_{name}", 0)
            obs.append(val)
        return np.array(obs, dtype=np.float32)
