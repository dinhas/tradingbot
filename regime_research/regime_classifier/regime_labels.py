import pandas as pd
import numpy as np
import talib
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
import nolds
import antropy
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_weights(d, size):
    """
    Get weights for fractional differentiation.
    """
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w[::-1])

def calculate_fracdiff(series, d, window=100):
    """
    Calculate fractional difference using a fixed window.
    """
    weights = get_weights(d, window)
    return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

def find_min_d(series):
    """
    Finds minimum d for fractional differencing where ADF p < 0.05.
    """
    # We use a subsample to find d for speed
    subsample = series.iloc[-1000:]
    for d in np.linspace(0, 1, 11):
        fd = calculate_fracdiff(subsample, d).dropna()
        if len(fd) < 50: continue
        try:
            res = adfuller(fd, maxlag=1)
            if res[1] < 0.05:
                return d
        except:
            continue
    return 1.0

def compute_features(df):
    """
    Step 1: Feature Engineering
    """
    df = df.copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values

    # Log returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # Fractionally differenced close
    print("Computing FracDiff...")
    d_opt = find_min_d(df['close'])
    df['frac_diff'] = calculate_fracdiff(df['close'], d_opt)

    # VWAP and deviation from VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['vwap']

    # Volatility Features
    df['atr14'] = talib.ATR(high, low, close, timeperiod=14)
    df['atr100'] = talib.ATR(high, low, close, timeperiod=100)
    df['atr_ratio'] = df['atr14'] / df['atr100']

    # Parkinson volatility
    df['parkinson'] = ((np.log(df['high'] / df['low']))**2) / (4 * np.log(2))

    # Realized volatility (rolling 20-bar std of log returns)
    df['realized_vol'] = df['log_ret'].rolling(20).std()

    # GARCH(1,1) conditional volatility
    # This might be slow if run on every bar. Usually calculated on windows.
    # We'll use a placeholder or simplified version if needed.
    # For now, let's try to fit it on the whole series (not ideal for research, but as requested)
    try:
        am = arch_model(df['log_ret'].dropna() * 100, vol='Garch', p=1, q=1)
        res = am.fit(disp='off')
        df['garch_vol'] = np.nan
        df.loc[df.index[1:], 'garch_vol'] = res.conditional_volatility / 100
    except:
        df['garch_vol'] = df['realized_vol']

    # Momentum/Structure Features
    df['rsi14'] = talib.RSI(close, timeperiod=14)
    df['roc5'] = talib.ROC(close, timeperiod=5)
    df['roc10'] = talib.ROC(close, timeperiod=10)
    df['roc20'] = talib.ROC(close, timeperiod=20)
    df['adx14'] = talib.ADX(high, low, close, timeperiod=14)

    # Rolling autocorrelation at lags 1, 2, 3, 5
    # for lag in [1, 2, 3, 5]:
    #     df[f'autocorr_{lag}'] = df['log_ret'].rolling(50).apply(lambda x: x.autocorr(lag=lag), raw=False)

    # Regime Detection Features
    # Note: These are slow.
    # df['hurst'] = df['close'].rolling(100).apply(lambda x: nolds.hurst_rs(x), raw=False)

    # Entropy features
    # df['perm_entropy'] = df['close'].rolling(50).apply(lambda x: antropy.perm_entropy(x, normalize=True), raw=False)
    # df['spectral_entropy'] = df['close'].rolling(50).apply(lambda x: antropy.spectral_entropy(x, sf=1, normalize=True), raw=False)
    # df['sample_entropy'] = df['close'].rolling(50).apply(lambda x: antropy.sample_entropy(x), raw=False)

    # Faster implementations or placeholders for now
    df['autocorr_1'] = df['log_ret'].rolling(50).corr(df['log_ret'].shift(1))
    df['autocorr_2'] = df['log_ret'].rolling(50).corr(df['log_ret'].shift(2))
    df['autocorr_3'] = df['log_ret'].rolling(50).corr(df['log_ret'].shift(3))
    df['autocorr_5'] = df['log_ret'].rolling(50).corr(df['log_ret'].shift(5))

    # Regime Detection Features
    # Optimized Hurst and Entropy
    def get_hurst(x):
        try:
            return nolds.hurst_rs(x)
        except:
            return 0.5

    def get_perm_entropy(x):
        return antropy.perm_entropy(x, normalize=True)

    # Apply on sampled basis or just try to optimize
    print("Computing Hurst...")
    df['hurst'] = df['close'].rolling(100).apply(get_hurst, raw=True)
    print("Computing Perm Entropy...")
    df['perm_entropy'] = df['close'].rolling(50).apply(get_perm_entropy, raw=True)
    print("Computing Spectral Entropy...")
    df['spectral_entropy'] = df['close'].rolling(50).apply(lambda x: antropy.spectral_entropy(x, sf=1, normalize=True), raw=True)

    print("Computing Sample Entropy (Optimized)...")
    # Sample entropy is O(N^2), so we use a small window and larger step if needed
    # To maintain dataframe length, we fill gaps
    def get_sample_entropy(x):
        try:
            return antropy.sample_entropy(x)
        except:
            return 0.5

    # Compute every 10 bars to speed up
    sample_ent = df['close'].rolling(50).apply(get_sample_entropy, raw=True)
    df['sample_entropy'] = sample_ent.ffill()

    return df

def label_regimes(df):
    """
    Step 2: Regime Classification
    """
    df = df.copy()
    df['regime'] = 'NOISE'

    atr_mean_50 = df['atr14'].rolling(50).mean()

    # Priority order
    # 1. BREAKOUT
    breakout_mask = (df['atr_ratio'] > 1.3) & (df['atr14'] > atr_mean_50 * 1.2)
    df.loc[breakout_mask, 'regime'] = 'BREAKOUT'

    # 2. TRENDING
    trending_mask = (df['regime'] == 'NOISE') & (df['adx14'] > 25) & (df['hurst'] > 0.55)
    df.loc[trending_mask, 'regime'] = 'TRENDING'

    # 3. RANGING
    ranging_mask = (df['regime'] == 'NOISE') & (df['adx14'] < 20) & (df['hurst'] < 0.50) & (df['atr_ratio'] < 1.1)
    df.loc[ranging_mask, 'regime'] = 'RANGING'

    return df

def plot_regimes(df, symbol, output_path):
    plt.figure(figsize=(15, 7))
    colors = {'RANGING': 'blue', 'TRENDING': 'green', 'BREAKOUT': 'red', 'NOISE': 'gray'}

    # For plotting efficiency, we might want to plot segments
    plt.plot(df.index, df['close'], color='black', alpha=0.3, label='Price')

    for regime, color in colors.items():
        mask = df['regime'] == regime
        if mask.any():
            plt.scatter(df.index[mask], df['close'][mask], color=color, label=regime, s=1)

    plt.title(f'Regime Overview - {symbol}')
    plt.legend()
    plt.savefig(output_path, dpi=150)
    plt.close()
