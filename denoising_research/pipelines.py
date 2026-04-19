import pandas as pd
import numpy as np
import pywt
from scipy.signal import savgol_filter
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange

# --- BASELINE KALMAN ---
def kalman_filter(data):
    Q_base = 1e-5
    R_base = 1e-3
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
    return np.array(filtered)

def apply_kalman(df):
    df = df.copy()
    df['close'] = kalman_filter(df['close'].values)
    return df

# --- WAVELET (CAUSAL) ---
def causal_wavelet_denoising(series, wavelet='db4', level=2, window=128):
    """
    Rolling window wavelet denoising to preserve causality.
    """
    def denoise_window(x):
        if len(x) < window: return x[-1]
        coeffs = pywt.wavedec(x, wavelet, level=level)
        # Universal thresholding
        sigma = (1/0.6745) * np.median(np.abs(coeffs[-1] - np.median(coeffs[-1])))
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='hard') for i in coeffs[1:]]
        reconstructed = pywt.waverec(coeffs, wavelet)
        return reconstructed[-1]

    return series.rolling(window=window).apply(denoise_window, raw=True)

def apply_wavelet(df):
    df = df.copy()
    df['close'] = causal_wavelet_denoising(df['close'])
    return df.dropna()

# --- EMA-BASED ---
def apply_ema_smoothing(df):
    df = df.copy()
    # Use ffill/bfill instead of dropna to keep index alignment for ensemble
    ema1 = EMAIndicator(df['close'], window=10).ema_indicator()
    ema2 = EMAIndicator(ema1, window=10).ema_indicator()
    df['close'] = (2 * ema1 - ema2).ffill().bfill()
    return df

# --- MEDIAN + SAVITZKY-GOLAY ---
def apply_median_savgol(df):
    df = df.copy()
    # Rolling median
    df['close'] = df['close'].rolling(window=5).median()
    # Savitzky-Golay (causal approximation: use only past values by taking the last point of the filter)
    # Since savgol_filter is usually centered, we apply it on a rolling window.
    def causal_savgol(x):
        if len(x) < 11: return x[-1]
        return savgol_filter(x, window_length=11, polyorder=2)[-1]

    df['close'] = df['close'].rolling(window=15).apply(causal_savgol, raw=True)
    return df.dropna()

# --- FRACTIONAL DIFFERENCING ---
def get_weights_ffd(d, size):
    w = [1.]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])

def frac_diff_fixed(series, d=0.4, window=20):
    weights = get_weights_ffd(d, window)
    prices = series.values
    # Manual convolution to ensure causality
    fd = []
    for i in range(len(prices)):
        if i < window - 1:
            fd.append(np.nan)
        else:
            window_slice = prices[i-window+1:i+1]
            fd.append(np.dot(window_slice, weights))
    return pd.Series(fd, index=series.index)

def apply_frac_diff(df):
    df = df.copy()
    df['close'] = frac_diff_fixed(np.log(df['close'] + 1e-8), d=0.4, window=20)
    return df.dropna()

# --- VOLATILITY FILTER ---
def apply_volatility_filter(df, threshold_pct=0.2):
    df = df.copy()
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    atr_norm = atr / df['close']
    # Instead of removing (which breaks time series), we could damp moves or just label
    # but the task says "Remove or downweight". For ML training, removing is common.
    # However, for consistency in this research, we'll mark them.
    mask = atr_norm > atr_norm.quantile(threshold_pct)
    return df[mask].copy()

# --- REGIME SEGMENTATION ---
def apply_regime_filter(df):
    df = df.copy()
    # Simplified ADX-based regime
    atr = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    df['adx'] = 0 # Placeholder if not calculated
    # For now, let's just use it as a passthrough or simple trend indicator
    df['ema_long'] = EMAIndicator(df['close'], window=50).ema_indicator()
    df['regime'] = np.where(df['close'] > df['ema_long'], 1, -1)
    return df.dropna()

# --- FEATURE TRANSFORMATIONS ---
def apply_feature_transforms(df):
    df = df.copy()
    # Log-transform to stabilize variance without breaking indicators
    df['close'] = np.log(df['close'] + 1e-8)
    if 'high' in df: df['high'] = np.log(df['high'] + 1e-8)
    if 'low' in df: df['low'] = np.log(df['low'] + 1e-8)
    if 'open' in df: df['open'] = np.log(df['open'] + 1e-8)
    return df.dropna()
