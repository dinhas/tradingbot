import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from numba import njit

@njit
def fast_hurst(x):
    lags = np.arange(2, 20); tau = np.zeros(len(lags))
    for i in range(len(lags)):
        lag = lags[i]; diff = x[lag:] - x[:-lag]; tau[i] = np.std(diff)
    log_lags, log_tau = np.log(lags.astype(np.float64)), np.log(tau)
    n = len(lags); x_mean, y_mean = np.mean(log_lags), np.mean(log_tau)
    num = np.sum((log_lags - x_mean) * (log_tau - y_mean)); den = np.sum((log_lags - x_mean)**2)
    return num / den if den != 0 else 0.5

@njit
def rolling_hurst_numba(values, window):
    n = len(values); res = np.full(n, np.nan)
    for i in range(window, n): res[i] = fast_hurst(values[i-window:i])
    return res

class FeatureEngine:
    def __init__(self, use_research_pipeline: bool = False):
        self.assets = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'XAUUSD']
        self.use_research_pipeline = use_research_pipeline
        self.feature_names = []
        self._define_feature_names()
        self.best_process_noise = 0.01

    def kalman_smooth(self, series, process_noise=0.01):
        from pykalman import KalmanFilter
        vals = series.ffill().fillna(0).values
        if len(vals) == 0:
            return pd.Series(dtype=np.float32)

        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=vals[0],
            initial_state_covariance=1,
            observation_covariance=1,
            transition_covariance=process_noise
        )
        state_means, _ = kf.filter(vals)
        return pd.Series(state_means.flatten(), index=series.index).astype(np.float32)

    def _fracdiff(self, series, d=0.4, window=100):
        weights = [1.0]
        for k in range(1, window):
            weights.append(-weights[-1] * (d - k + 1) / k)
        weights = np.array(weights[::-1])
        result = np.full(len(series), np.nan)
        vals = series.values
        for i in range(window - 1, len(vals)):
            result[i] = np.dot(weights, vals[i-window+1:i+1])
        return pd.Series(result, index=series.index).astype(np.float32)

    def _calculate_hurst(self, series, window=300):
        return pd.Series(rolling_hurst_numba(series.values.astype(np.float64), window), index=series.index)

    def _define_feature_names(self):
        if self.use_research_pipeline:
            self.feature_names = [
                "ema_diff_kalman", "rsi_kalman", "macd_hist_kalman",
                "rsi_momentum_kalman", "fracdiff_close"
            ]
        else:
            self.feature_names = [
                "bollinger_pB", "ema_diff", "macd_hist", "rsi_momentum", "rsi",
                "bb_width", "volatility", "atr_norm", "hour", "regime", "is_tradeable"
            ]

    def preprocess_data(self, data_dict):
        logger = logging.getLogger(__name__)
        aligned_df = self._align_data(data_dict)
        
        # Don't convert everything to float32 yet if we have strings (like regime)
        # aligned_df = aligned_df.astype(np.float32)

        all_new_cols = {}
        for asset in self.assets:
            logger.info(f"Calculating features for {asset}...")
            asset_cols = self._get_asset_features(aligned_df, asset)
            all_new_cols.update(asset_cols)
            
        if not self.use_research_pipeline:
            logger.info("Adding time features...")
            time_cols = self._get_time_features(aligned_df)
            all_new_cols.update(time_cols)

        # Separate numeric and categorical cols
        numeric_cols = {k: v for k, v in all_new_cols.items() if not isinstance(v.iloc[0], str)}
        categ_cols = {k: v for k, v in all_new_cols.items() if isinstance(v.iloc[0], str)}

        new_features_df = pd.DataFrame(numeric_cols, index=aligned_df.index).astype(np.float32)
        if categ_cols:
            categ_df = pd.DataFrame(categ_cols, index=aligned_df.index)
            new_features_df = pd.concat([new_features_df, categ_df], axis=1)

        aligned_df = pd.concat([aligned_df, new_features_df], axis=1)
        
        if not self.use_research_pipeline:
            logger.info("Normalizing features...")
            normalized_df = aligned_df.copy()
            normalized_df = self._normalize_features(normalized_df)
            normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)
        else:
            # Drop strings for normalized_df used by the model
            normalized_df = aligned_df.select_dtypes(include=[np.number]).copy()
            normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)

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
        logger = logging.getLogger(__name__)
        raw_close = df[f"{asset}_close"]
        raw_high = df[f"{asset}_high"]
        raw_low = df[f"{asset}_low"]
        
        new_cols = {}
        
        if self.use_research_pipeline:
            rsi = RSIIndicator(raw_close, window=14).rsi()
            macd = MACD(raw_close)
            macd_hist = macd.macd_diff()
            ema_diff = (raw_close - raw_close.rolling(20).mean()) / (raw_close.rolling(20).std() + 1e-8)
            rsi_momentum = rsi - rsi.shift(1)
            atr_raw = AverageTrueRange(raw_high, raw_low, raw_close, window=14).average_true_range()
            adx_14 = ADXIndicator(raw_high, raw_low, raw_close, window=14).adx()

            logger.info(f"Calculating Hurst(300) for {asset}...")
            hurst_300 = self._calculate_hurst(raw_close, window=300)
            new_cols[f"{asset}_hurst_300"] = hurst_300.astype(np.float32)

            regime = np.where((adx_14 < 20) & (hurst_300 < 0.48), 'RANGING', 'OTHER')
            new_cols[f"{asset}_regime"] = pd.Series(regime, index=df.index)

            logger.info(f"Tuning Kalman Filter for {asset} ATR...")
            target_range = (0.95, 0.98)
            best_noise = 0.01
            best_corr = 0
            atr_raw_valid = atr_raw.dropna()

            for noise in [0.001, 0.005, 0.01, 0.05, 0.1]:
                smoothed = self.kalman_smooth(atr_raw, process_noise=noise)
                valid_idx = atr_raw_valid.index.intersection(smoothed.dropna().index)
                if len(valid_idx) > 10:
                    corr = np.corrcoef(atr_raw.loc[valid_idx], smoothed.loc[valid_idx])[0,1]
                    if target_range[0] <= corr <= target_range[1]:
                        best_noise = noise
                        best_corr = corr
                        break
                    if abs(corr - 0.965) < abs(best_corr - 0.965):
                        best_noise = noise
                        best_corr = corr

            self.best_process_noise = best_noise
            logger.info(f"[FEATURE ENGINE] {asset} process_noise used: {best_noise}, Kalman ATR correlation: {best_corr:.4f}")

            new_cols[f"{asset}_ema_diff_kalman"] = self.kalman_smooth(ema_diff, process_noise=best_noise)
            new_cols[f"{asset}_rsi_kalman"] = self.kalman_smooth(rsi, process_noise=best_noise)
            new_cols[f"{asset}_macd_hist_kalman"] = self.kalman_smooth(macd_hist, process_noise=best_noise)
            new_cols[f"{asset}_rsi_momentum_kalman"] = self.kalman_smooth(rsi_momentum, process_noise=best_noise)
            new_cols[f"{asset}_atr_kalman"] = self.kalman_smooth(atr_raw, process_noise=best_noise)
            new_cols[f"{asset}_fracdiff_close"] = self._fracdiff(raw_close, d=0.4, window=100)
            new_cols[f"{asset}_adx"] = adx_14
            new_cols[f"{asset}_atr"] = atr_raw

        else:
            close_filled = raw_close.ffill().bfill().to_numpy(copy=True)
            high_filled = raw_high.ffill().bfill().to_numpy(copy=True)
            low_filled = raw_low.ffill().bfill().to_numpy(copy=True)

            def _legacy_kalman(data, Q_base=1e-4, R_base=1e-4):
                if len(data) == 0: return np.array([], dtype=np.float32)
                xhat = data[0]; P = 1.0; filtered = []; var_innovation = 1e-5; alpha = 0.1
                for z in data:
                    P = P + Q_base; innovation = z - xhat; var_innovation = (1 - alpha) * var_innovation + alpha * (innovation ** 2)
                    Q_adaptive = max(Q_base, 0.05 * var_innovation); P = P + Q_adaptive; K = P / (P + R_base); xhat = xhat + K * innovation; P = (1 - K) * P
                    filtered.append(xhat)
                return np.array(filtered, dtype=np.float32)

            close = pd.Series(_legacy_kalman(close_filled), index=raw_close.index)
            high = pd.Series(_legacy_kalman(high_filled), index=raw_close.index)
            low = pd.Series(_legacy_kalman(low_filled), index=raw_close.index)

            new_cols[f"{asset}_rsi"] = RSIIndicator(close, window=14).rsi()
            macd = MACD(close)
            new_cols[f"{asset}_macd_hist"] = macd.macd_diff()
            bb = BollingerBands(close, window=20, window_dev=2)
            new_cols[f"{asset}_bollinger_pB"] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
            new_cols[f"{asset}_bb_width"] = bb.bollinger_wband()
            new_cols[f"{asset}_ema_diff"] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)
            new_cols[f"{asset}_rsi_momentum"] = new_cols[f"{asset}_rsi"] * (close / close.shift(5) - 1)
            new_cols[f"{asset}_volatility"] = close.pct_change().rolling(20).std()
            new_cols[f"{asset}_atr_norm"] = AverageTrueRange(high, low, close, window=14).average_true_range() / (close + 1e-8)

            atr_indicator = AverageTrueRange(raw_high, raw_low, raw_close, window=14)
            atr = atr_indicator.average_true_range().fillna(0)
            adx = ADXIndicator(raw_high, raw_low, raw_close, window=14).adx().fillna(0)
            atr_norm_raw = atr / (raw_close + 1e-8)
            atr_q75 = atr_norm_raw.rolling(500).quantile(0.75)
            is_trending = (adx > 25) & (atr_norm_raw < atr_q75)
            new_cols[f"{asset}_regime"] = is_trending.astype(np.float32)
            new_cols[f"{asset}_is_tradeable"] = is_trending.astype(np.float32)
            new_cols[f"{asset}_atr"] = atr
            new_cols[f"{asset}_adx"] = adx
        
        return new_cols

    def _get_time_features(self, df):
        new_cols = {}
        hours = df.index.hour
        new_cols['hour_of_day'] = hours.astype(np.float32)
        new_cols['is_late_session'] = ((hours >= 14) & (hours <= 20)).astype(np.float32)
        new_cols['is_friday'] = (df.index.dayofweek == 4).astype(np.float32)
        return new_cols

    def _normalize_features(self, df):
        for asset in self.assets:
            cols_to_scale = [
                f"{asset}_bollinger_pB", f"{asset}_ema_diff", f"{asset}_macd_hist",
                f"{asset}_rsi_momentum", f"{asset}_rsi", f"{asset}_bb_width",
                f"{asset}_volatility", f"{asset}_atr_norm"
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
        if self.use_research_pipeline:
            obs_cols = [f"{asset}_{c}" for c in self.feature_names]
        else:
            obs_cols = [
                f"{asset}_bollinger_pB", f"{asset}_ema_diff", f"{asset}_macd_hist",
                f"{asset}_rsi_momentum", f"{asset}_rsi", f"{asset}_bb_width",
                f"{asset}_volatility", f"{asset}_atr_norm", 'hour_of_day',
                f"{asset}_regime", f"{asset}_is_tradeable"
            ]
        return df.reindex(columns=obs_cols, fill_value=0).values.astype(np.float32)

    def get_observation(self, current_step_data, portfolio_state, asset):
        if self.use_research_pipeline:
            obs = [current_step_data.get(f"{asset}_{c}", 0) for c in self.feature_names]
        else:
            obs = [
                current_step_data.get(f"{asset}_bollinger_pB", 0),
                current_step_data.get(f"{asset}_ema_diff", 0),
                current_step_data.get(f"{asset}_macd_hist", 0),
                current_step_data.get(f"{asset}_rsi_momentum", 0),
                current_step_data.get(f"{asset}_rsi", 0),
                current_step_data.get(f"{asset}_bb_width", 0),
                current_step_data.get(f"{asset}_volatility", 0),
                current_step_data.get(f"{asset}_atr_norm", 0),
                current_step_data.get('hour_of_day', 0),
                current_step_data.get(f"{asset}_regime", 0),
                current_step_data.get(f"{asset}_is_tradeable", 0)
            ]
        return np.array(obs, dtype=np.float32)
