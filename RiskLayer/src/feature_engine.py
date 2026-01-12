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
        Optimized to reduce memory fragmentation.
        """
        # 1. Align Data
        aligned_df = self._align_data(data_dict)
        
        # 2. Pre-calculate market returns once
        all_returns = pd.DataFrame()
        for a in self.assets:
            if f"{a}_close" in aligned_df.columns:
                all_returns[a] = aligned_df[f"{a}_close"].pct_change()
        
        # 3. Add Features per Asset
        # To avoid fragmentation, we collect ALL new features for ALL assets first
        all_new_features = []
        
        for asset in self.assets:
            # Collect features for this asset in a list
            asset_features_list = []
            
            # Group 1: Market Pressure & Flow
            asset_features_list.append(self._add_market_pressure_features(aligned_df, asset))
            
            # Group 2: Volatility & Structure
            asset_features_list.append(self._add_volatility_features(aligned_df, asset))
            
            # Group 3: Trend Strength
            asset_features_list.append(self._add_trend_features(aligned_df, asset))
            
            # Group 4: Regime
            asset_features_list.append(self._add_regime_features(aligned_df, asset))
            
            # Group 5: Alpha/Cross-Asset
            asset_features_list.append(self._add_alpha_features(aligned_df, asset, all_returns))
            
            # Concatenate all features for this asset
            asset_features_df = pd.concat(asset_features_list, axis=1)
            all_new_features.append(asset_features_df)

        # 4. Final Concatenation
        # Combine original data with all new features at once
        final_df = pd.concat([aligned_df] + all_new_features, axis=1)
        
        return final_df

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
        """Group 1: Market Pressure & Flow"""
        features = pd.DataFrame(index=df.index)
        close = df[f"{asset}_close"]
        open_ = df[f"{asset}_open"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        volume = df[f"{asset}_volume"]
        
        # 1. RSI & Velocity
        rsi = RSIIndicator(close, window=14).rsi()
        features[f"{asset}_rsi_14"] = rsi
        features[f"{asset}_rsi_velocity"] = rsi.diff(3)
        
        # 2. MACD
        macd = MACD(close)
        features[f"{asset}_macd_hist"] = macd.macd_diff()
        # Proxy divergence
        price_slope = close.diff(3)
        macd_slope = features[f"{asset}_macd_hist"].diff(3)
        features[f"{asset}_macd_div_proxy"] = np.where(
            (price_slope > 0) & (macd_slope < 0), 1,
            np.where((price_slope < 0) & (macd_slope > 0), -1, 0)
        )

        # 3. Volume Flow
        features[f"{asset}_obv_slope"] = OnBalanceVolumeIndicator(close, volume).on_balance_volume().diff(10)
        features[f"{asset}_cmf"] = ChaikinMoneyFlowIndicator(high, low, close, volume, window=20).chaikin_money_flow()
        features[f"{asset}_force_index"] = ForceIndexIndicator(close, volume, window=13).force_index()
        
        # 4. Candle Structure
        range_ = high - low
        body = (open_ - close).abs()
        features[f"{asset}_candle_body_pct"] = body / range_.replace(0, 1e-9)
        
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low
        features[f"{asset}_upper_wick_pct"] = upper_wick / range_.replace(0, 1e-9)
        features[f"{asset}_lower_wick_pct"] = lower_wick / range_.replace(0, 1e-9)
        
        # 5. Volume Pressure
        vol_sma = volume.rolling(20).mean()
        features[f"{asset}_volume_ratio"] = volume / vol_sma.replace(0, 1)
        
        features[f"{asset}_buying_pressure"] = (close - low) * volume
        features[f"{asset}_selling_pressure"] = (high - close) * volume
        
        # 6. Oscillators
        stoch = StochRSIIndicator(close, window=14)
        features[f"{asset}_stoch_k"] = stoch.stochrsi_k()
        features[f"{asset}_stoch_d"] = stoch.stochrsi_d()
        
        features[f"{asset}_mfi"] = MFIIndicator(high, low, close, volume, window=14).money_flow_index()
        features[f"{asset}_williams_r"] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()
        
        # 7. Velocity
        features[f"{asset}_roc"] = ROCIndicator(close, window=12).roc()
        
        # 8. Price Action
        features[f"{asset}_gap_size"] = open_ - close.shift(1)
        
        clv = ((close - low) - (high - close)) / range_.replace(0, 1e-9)
        features[f"{asset}_close_loc_value"] = clv
        
        return features

    def _add_volatility_features(self, df, asset):
        """Group 2: Volatility & Structure"""
        features = pd.DataFrame(index=df.index)
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        
        # 1. ATR
        atr = AverageTrueRange(high, low, close, window=14).average_true_range()
        atr_100 = AverageTrueRange(high, low, close, window=100).average_true_range()
        features[f"{asset}_atr_14"] = atr
        features[f"{asset}_atr_ratio"] = atr / atr_100.replace(0, 1e-9)
        features[f"{asset}_atr_slope"] = atr.diff(3)
        
        # 2. BB
        bb = BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_mid = bb.bollinger_mavg()
        
        features[f"{asset}_bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, 1e-9)
        features[f"{asset}_bb_pct_b"] = bb.bollinger_pband()
        
        # 3. Std Dev
        features[f"{asset}_std_dev_20"] = close.rolling(20).std()
        
        # 4. Keltner
        kc = KeltnerChannel(high, low, close, window=20)
        kc_upper = kc.keltner_channel_hband()
        kc_lower = kc.keltner_channel_lband()
        features[f"{asset}_keltner_pos"] = (close - kc_lower) / (kc_upper - kc_lower).replace(0, 1e-9)
        
        # 5. Donchian
        dc = DonchianChannel(high, low, close, window=50)
        dc_high = dc.donchian_channel_hband()
        dc_low = dc.donchian_channel_lband()
        features[f"{asset}_donchian_high_dist"] = (dc_high - close) / atr.replace(0, 1e-9)
        features[f"{asset}_donchian_low_dist"] = (close - dc_low) / atr.replace(0, 1e-9)
        
        # 6. Pivot
        rolling_high_24h = high.rolling(288).max()
        rolling_low_24h = low.rolling(288).min()
        rolling_close_24h = close.shift(288)
        pivot = (rolling_high_24h + rolling_low_24h + rolling_close_24h) / 3
        features[f"{asset}_pivot_dist"] = (close - pivot) / atr.replace(0, 1e-9)
        
        # 7. Range
        features[f"{asset}_avg_candle_range"] = (high - low).rolling(10).mean()
        
        parkinson = np.sqrt((1.0 / (4.0 * np.log(2.0))) * np.log(high / low.replace(0, 1e-9))**2)
        features[f"{asset}_parkinson_vol"] = parkinson.rolling(20).mean()
        
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        features[f"{asset}_z_score"] = (close - sma_20) / std_20.replace(0, 1e-9)
        
        features[f"{asset}_prev_day_high_dist"] = (rolling_high_24h - close) / atr.replace(0, 1e-9)
        features[f"{asset}_prev_day_low_dist"] = (close - rolling_low_24h) / atr.replace(0, 1e-9)
        
        returns = close.pct_change()
        upside_vol = returns[returns > 0].rolling(50).std().reindex(returns.index).ffill()
        downside_vol = returns[returns < 0].rolling(50).std().reindex(returns.index).ffill()
        features[f"{asset}_vol_skew"] = upside_vol - downside_vol
        
        features[f"{asset}_high_low_range_pct"] = (high - low) / close
        
        upper_wick = high - np.maximum(df[f"{asset}_open"], close)
        lower_wick = np.minimum(df[f"{asset}_open"], close) - low
        features[f"{asset}_shadow_symmetry"] = upper_wick / lower_wick.replace(0, 1e-9)
        
        return features

    def _add_trend_features(self, df, asset):
        """Group 3: Trend Strength"""
        features = pd.DataFrame(index=df.index)
        close = df[f"{asset}_close"]
        high = df[f"{asset}_high"]
        low = df[f"{asset}_low"]
        
        # 1. ADX
        adx_ind = ADXIndicator(high, low, close, window=14)
        features[f"{asset}_adx_14"] = adx_ind.adx()
        features[f"{asset}_di_plus"] = adx_ind.adx_pos()
        features[f"{asset}_di_minus"] = adx_ind.adx_neg()
        
        # 2. Aroon
        aroon = AroonIndicator(high, low, window=25)
        features[f"{asset}_aroon_up"] = aroon.aroon_up()
        features[f"{asset}_aroon_down"] = aroon.aroon_down()
        
        # 3. MA
        ema9 = EMAIndicator(close, window=9).ema_indicator()
        ema21 = EMAIndicator(close, window=21).ema_indicator()
        ema50 = EMAIndicator(close, window=50).ema_indicator()
        ema200 = EMAIndicator(close, window=200).ema_indicator()
        
        features[f"{asset}_ema_9"] = ema9
        features[f"{asset}_ema_21"] = ema21
        features[f"{asset}_ema_50"] = ema50
        features[f"{asset}_ema_200"] = ema200
        
        features[f"{asset}_price_vs_ema50"] = (close - ema50) / ema50
        features[f"{asset}_price_vs_ema200"] = (close - ema200) / ema200
        
        features[f"{asset}_ema_ribbon_spread"] = (ema9 - ema50) / ema50
        
        # 4. SuperTrend (Simplified)
        features[f"{asset}_supertrend"] = np.where(close > ema50, 1, -1)
        
        # 5. Ichimoku
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
        
        above_cloud = (close > senkou_span_a) & (close > senkou_span_b)
        below_cloud = (close < senkou_span_a) & (close < senkou_span_b)
        features[f"{asset}_cloud_pos"] = np.where(above_cloud, 1, np.where(below_cloud, -1, 0))
        
        # 6. PSAR
        from ta.trend import PSARIndicator
        psar = PSARIndicator(high, low, close).psar()
        features[f"{asset}_parabolic_sar_pos"] = np.where(close > psar, 1, -1)
        
        return features

    def _add_regime_features(self, df, asset):
        """Group 4: Market Regime"""
        features = pd.DataFrame(index=df.index)
        
        features[f"{asset}_hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        features[f"{asset}_hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        features[f"{asset}_day_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        features[f"{asset}_day_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        hours = df.index.hour
        features[f"{asset}_session_london"] = ((hours >= 8) & (hours < 17)).astype(int)
        features[f"{asset}_session_ny"] = ((hours >= 13) & (hours < 22)).astype(int)
        features[f"{asset}_session_asian"] = ((hours >= 0) & (hours < 9)).astype(int)
        
        change = df[f"{asset}_close"].diff(10).abs()
        volatility = df[f"{asset}_close"].diff().abs().rolling(10).sum()
        features[f"{asset}_efficiency_ratio"] = change / volatility.replace(0, 1e-9)
        
        ret = df[f"{asset}_close"].pct_change()
        std_20 = ret.rolling(20).std()
        std_10 = ret.rolling(10).std()
        features[f"{asset}_hurst_exp"] = std_20 / std_10.replace(0, 1e-9)
        
        features[f"{asset}_fractal_dim"] = 2 - features[f"{asset}_hurst_exp"]
        
        return features

    def _add_alpha_features(self, df, asset, all_returns=None):
        """Group 5: Alpha Signals & Cross-Asset"""
        features = pd.DataFrame(index=df.index)
        close = df[f"{asset}_close"]
        ret = close.pct_change()
        
        # 1. Alpha Signal (Placeholder)
        features[f"{asset}_alpha_signal"] = 0.0
        features[f"{asset}_alpha_conf"] = 0.0
            
        # 2. Correlations
        if all_returns is None or all_returns.empty:
            # Fallback (slow)
            all_returns = pd.DataFrame()
            for a in self.assets:
                if f"{a}_close" in df.columns:
                    all_returns[a] = df[f"{a}_close"].pct_change()
        
        if not all_returns.empty:
            basket_return = all_returns.mean(axis=1)
            
            features[f"{asset}_corr_usd"] = ret.rolling(50).corr(basket_return)
            features[f"{asset}_rel_strength"] = ret - basket_return
            
            cov = ret.rolling(50).cov(basket_return)
            var = basket_return.rolling(50).var()
            features[f"{asset}_beta"] = cov / var.replace(0, 1e-9)
            
            global_vol = all_returns.rolling(20).std().mean(axis=1)
            features[f"{asset}_global_vol"] = global_vol
            
            features[f"{asset}_global_trend"] = all_returns.rolling(20).mean().mean(axis=1)
            features[f"{asset}_sector_mom"] = basket_return.rolling(20).mean()
            
        if "XAUUSD_close" in df.columns:
            xau_ret = all_returns["XAUUSD"] if "XAUUSD" in all_returns else df["XAUUSD_close"].pct_change()
            features[f"{asset}_corr_gold"] = ret.rolling(50).corr(xau_ret)
        else:
            features[f"{asset}_corr_gold"] = 0
            
        features[f"{asset}_corr_sp500"] = 0
        
        # 3. Market Microstructure
        features[f"{asset}_spread"] = (df[f"{asset}_high"] - df[f"{asset}_low"]).rolling(5).mean()
        
        features[f"{asset}_vol_variance"] = df[f"{asset}_volume"].rolling(20).var()
        features[f"{asset}_price_variance"] = close.rolling(20).var()
        
        if "EURUSD_close" in df.columns:
            eur_ret = all_returns["EURUSD"] if "EURUSD" in all_returns else df["EURUSD_close"].pct_change()
            features[f"{asset}_cov_lead"] = ret.rolling(50).cov(eur_ret)
        else:
            features[f"{asset}_cov_lead"] = 0
            
        features[f"{asset}_tick_vol"] = df[f"{asset}_volume"]

        return features