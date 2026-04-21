import pandas as pd
import numpy as np
import os
import logging
import datetime
import json
import glob
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
import nolds
import antropy
from PyEMD import EMD
import stumpy
from dtaidistance import dtw
from pykalman import KalmanFilter
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from statsmodels.tsa.filters.hp_filter import hpfilter
import pywt
from numba import njit

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dirs():
    dirs = [
        'regime_research/plots',
        'regime_research/regime_ranging',
        'regime_research/regime_trending',
        'regime_research/regime_breakout',
        'regime_research/smoothing_comparison'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    logger.info("Directories created.")

def calculate_baseline_features(df):
    """
    Calculate baseline features based on Alpha/src/feature_engine.py logic
    """
    close = df['close']
    high = df['high']
    low = df['low']

    features = pd.DataFrame(index=df.index)

    # Technical Features
    features['rsi'] = RSIIndicator(close, window=14).rsi()
    macd = MACD(close)
    features['macd_hist'] = macd.macd_diff()

    bb = BollingerBands(close, window=20, window_dev=2)
    features['bollinger_pB'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-8)
    features['bb_width'] = bb.bollinger_wband()

    features['ema_diff'] = (close - close.rolling(20).mean()) / (close.rolling(20).std() + 1e-8)
    features['rsi_momentum'] = features['rsi'] * (close / close.shift(5) - 1)

    features['volatility'] = close.pct_change().rolling(20).std()
    features['atr_norm'] = AverageTrueRange(high, low, close, window=14).average_true_range() / (close + 1e-8)

    return features

def calculate_hurst(series, window=100):
    """Calculates the rolling Hurst exponent using a faster method."""
    def get_hurst(x):
        try:
            lags = range(2, 20)
            tau = [np.std(np.subtract(x[lag:], x[:-lag])) for lag in lags]
            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]
        except:
            return np.nan
    return series.rolling(window=window).apply(get_hurst, raw=True)

def phase_0():
    logger.info("Starting Phase 0: Data Loading and Baseline Feature Extraction")
    create_dirs()

    data_files = glob.glob('data/*.parquet')
    if not data_files:
        logger.error("No parquet data files found in ./data/")
        return None

    data_path = data_files[0] # Use the first one
    logger.info(f"Using data file: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} bars of data.")

    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ['timestamp', 'time', 'date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df.set_index(col, inplace=True)
                break

    df.sort_index(inplace=True)

    baseline_features = calculate_baseline_features(df)

    cols_to_keep = ['open', 'high', 'low', 'close']
    for v_col in ['volume', 'tick_volume']:
        if v_col in df.columns:
            cols_to_keep.append(v_col)
            break

    baseline_df = pd.concat([df[cols_to_keep], baseline_features], axis=1)
    baseline_df.to_parquet('regime_research/baseline_features.parquet')

    logger.info(f"Saved baseline features. Shape: {baseline_df.shape}")
    return baseline_df

def phase_1(df):
    logger.info("Starting Phase 1: Regime Classification")

    close = df['close']
    high = df['high']
    low = df['low']

    logger.info("Calculating Hurst Exponent...")
    df['hurst'] = calculate_hurst(close, window=100)
    df['adx'] = ADXIndicator(high, low, close, window=14).adx()

    atr14 = AverageTrueRange(high, low, close, window=14).average_true_range()
    atr100 = AverageTrueRange(high, low, close, window=100).average_true_range()
    df['atr_ratio'] = atr14 / (atr100 + 1e-8)

    bb = BollingerBands(close, window=20, window_dev=2)
    bbw = bb.bollinger_wband()
    df['bb_width_pct'] = bbw.rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan) * 100

    conditions = [
        (df['atr_ratio'] > 1.3) & (df['bb_width_pct'] > 80),
        (df['adx'] > 25) & (df['hurst'] > 0.55),
        (df['adx'] < 20) & (df['hurst'] < 0.48)
    ]
    choices = ['BREAKOUT', 'TRENDING', 'RANGING']
    df['regime'] = np.select(conditions, choices, default='NOISE')

    dist = df['regime'].value_counts(normalize=True) * 100
    logger.info(f"Regime Distribution:\n{dist}")

    plt.figure(figsize=(15, 7))
    df_resampled = df.groupby([df.index.date, 'regime']).size().unstack(fill_value=0)
    df_resampled = df_resampled.div(df_resampled.sum(axis=1), axis=0)
    df_resampled.plot(kind='bar', stacked=True, figsize=(15, 7), width=1.0)
    plt.title('Regime Distribution Over Time')
    plt.ylabel('Proportion')
    plt.xlabel('Date')
    n = max(1, len(df_resampled) // 20)
    labels = [d.strftime('%Y-%m') if hasattr(d, 'strftime') else str(d) for d in df_resampled.index[::n]]
    plt.xticks(np.arange(0, len(df_resampled), n), labels)
    plt.savefig('regime_research/plots/regime_distribution.png')
    plt.close()

    df.to_parquet('regime_research/baseline_features_with_regime.parquet')
    return df

def phase_2_stats_and_signal(df):
    logger.info("Starting Phase 2: Regime Research (Statistical & Signal Quality)")
    regimes = ['RANGING', 'TRENDING', 'BREAKOUT']
    base_features = ['bollinger_pB', 'ema_diff', 'macd_hist', 'rsi_momentum', 'rsi', 'bb_width', 'volatility', 'atr_norm']

    df['fwd_ret_1'] = df['close'].shift(-1) / df['close'] - 1
    df['fwd_ret_5'] = df['close'].shift(-5) / df['close'] - 1

    all_stats = {}
    signal_quality = {}

    for regime in regimes:
        logger.info(f"Analyzing regime: {regime}")
        rdf = df[df['regime'] == regime].copy()
        if len(rdf) < 50: continue

        stats = {}
        try: stats['adf_p'] = adfuller(rdf['close'].dropna())[1]
        except: stats['adf_p'] = np.nan
        try:
            lb = acorr_ljungbox(rdf['close'].pct_change().dropna(), lags=[10])
            stats['lb_p'] = lb.lb_pvalue.iloc[0]
        except: stats['lb_p'] = np.nan
        stats['hurst_avg'] = rdf['hurst'].mean()
        try: stats['sampen'] = nolds.sampen(rdf['close'].iloc[:min(len(rdf), 500)].values)
        except: stats['sampen'] = np.nan
        try: stats['permen'] = antropy.perm_entropy(rdf['close'].values, normalize=True)
        except: stats['permen'] = np.nan
        try: stats['jb_p'] = jarque_bera(rdf['close'].pct_change().dropna())[1]
        except: stats['jb_p'] = np.nan
        all_stats[regime] = stats

        regime_signals = {}
        for feat in base_features:
            if feat not in df.columns: continue
            ic1 = rdf[feat].corr(rdf['fwd_ret_1'], method='spearman')
            ic5 = rdf[feat].corr(rdf['fwd_ret_5'], method='spearman')
            rolling_ic1 = df[feat].rolling(50).corr(df['fwd_ret_1'])
            icir = rolling_ic1.mean() / (rolling_ic1.std() + 1e-8) if not rolling_ic1.dropna().empty else 0
            regime_signals[feat] = {'IC_1': ic1, 'IC_5': ic5, 'ICIR': icir, 'strength': 'strong' if abs(icir) > 0.4 else ('weak/dead' if abs(icir) < 0.2 else 'neutral')}
        signal_quality[regime] = regime_signals

    with open('regime_research/stats_results.json', 'w') as f: json.dump(all_stats, f, indent=4)
    with open('regime_research/signal_quality.json', 'w') as f: json.dump(signal_quality, f, indent=4)
    return all_stats, signal_quality

def phase_2_visual_and_patterns(df, all_stats, signal_quality):
    logger.info("Starting Phase 2: Visual & Pattern Simulation")
    regimes = ['RANGING', 'TRENDING', 'BREAKOUT']
    regime_scores = {}

    for regime in regimes:
        logger.info(f"Visualizing: {regime}")
        rdf = df[df['regime'] == regime].copy()
        if len(rdf) < 100: continue

        plt.figure(figsize=(8, 8))
        sample = rdf['close'].iloc[:500].values
        if len(sample) > 0:
            dists = np.abs(sample[:, None] - sample)
            plt.imshow(dists < (np.std(sample) * 0.5), cmap='Greys', origin='lower')
            plt.title(f'Recurrence Plot - {regime}')
            plt.savefig(f'regime_research/plots/recurrence_{regime}.png')
        plt.close()

        plt.figure(figsize=(8, 8))
        try:
            acf_vals = acf(rdf['close'].dropna(), nlags=50)
            lag = (np.where(acf_vals < 0)[0] or [1])[0]
        except: lag = 1
        plt.plot(rdf['close'].iloc[:-lag].values, rdf['close'].iloc[lag:].values, '.', alpha=0.5)
        plt.title(f'Phase Space (lag={lag}) - {regime}')
        plt.savefig(f'regime_research/plots/phase_space_{regime}.png')
        plt.close()

        try:
            emd = EMD()
            imfs = emd(rdf['close'].iloc[:500].values)
            plt.figure(figsize=(10, 12))
            for i in range(min(4, imfs.shape[0])):
                plt.subplot(5, 1, i+1); plt.plot(imfs[i]); plt.title(f'IMF {i+1}')
            plt.subplot(5, 1, 5); plt.plot(rdf['close'].iloc[:500].values - np.sum(imfs[:4], axis=0)); plt.title('Residual')
            plt.tight_layout(); plt.savefig(f'regime_research/plots/emd_{regime}.png'); plt.close()
        except: pass

        plt.figure(figsize=(10, 5))
        try:
            acf_vals = acf(rdf['close'].pct_change().dropna(), nlags=30)
            plt.bar(range(len(acf_vals)), acf_vals); plt.title(f'ACF - {regime}')
            plt.savefig(f'regime_research/plots/acf_{regime}.png')
        except: pass
        plt.close()

        if regime in signal_quality:
            feats = list(signal_quality[regime].keys())
            ics = [signal_quality[regime][f]['IC_1'] for f in feats]
            plt.figure(figsize=(12, 6)); sorted_idx = np.argsort(ics)
            plt.barh(np.array(feats)[sorted_idx], np.array(ics)[sorted_idx]); plt.title(f'Feature IC - {regime}')
            plt.savefig(f'regime_research/plots/feature_ic_{regime}.png'); plt.close()

        motif_win_rate = 0.5
        try:
            mp = stumpy.stump(rdf['close'].values, m=20)
            motif_idx = np.argsort(mp[:, 0])[:5]
            plt.figure(figsize=(10, 6)); win_count = 0
            for idx in motif_idx:
                motif = rdf['close'].iloc[idx:idx+20].values
                plt.plot((motif - np.mean(motif)) / np.std(motif), alpha=0.5)
                if idx + 25 < len(rdf) and rdf['close'].iloc[idx+25] > rdf['close'].iloc[idx+20]: win_count += 1
            plt.title(f'Motifs - {regime}'); plt.savefig(f'regime_research/plots/motifs_{regime}.png'); plt.close()
            motif_win_rate = win_count / 5.0
        except: pass

        avg_icir = np.mean(sorted([abs(signal_quality[regime][f]['ICIR']) for f in signal_quality[regime]], reverse=True)[:3]) if regime in signal_quality else 0
        inv_permen = 1.0 - (all_stats[regime]['permen'] if (regime in all_stats and not np.isnan(all_stats[regime]['permen'])) else 0.5)
        score = (avg_icir * 10) + (motif_win_rate * 5) + (inv_permen * 5) + (min(len(rdf)/10000.0, 1.0) * 2)
        regime_scores[regime] = {'Signal Strength': avg_icir, 'Pattern Repeatability': motif_win_rate, 'Predictability': inv_permen, 'Data Volume': len(rdf), 'Total Score': score}

    with open('regime_research/regime_scores.json', 'w') as f: json.dump(regime_scores, f, indent=4)
    return regime_scores

@njit
def fast_causal_rolling(x, window, func_idx):
    # func_idx: 0=mean, 1=last_of_fit (placeholder for complex logic)
    res = np.zeros_like(x)
    for i in range(window, len(x)):
        win = x[i-window:i]
        if func_idx == 0: res[i] = np.mean(win)
    return res

def phase_3_smoothing_comparison(df, regime_scores, signal_quality):
    logger.info("Starting Phase 3: Smoothing Comparison (Fixed Causality)")
    top_regime = max(regime_scores, key=lambda k: regime_scores[k]['Total Score'])
    top_3_feats = sorted(signal_quality[top_regime].keys(), key=lambda k: abs(signal_quality[top_regime][k]['ICIR']), reverse=True)[:3]
    results = []

    for feat in top_3_feats:
        raw_full = df[feat].fillna(0).values
        techniques = {'NONE': raw_full, 'EMA_10': df[feat].ewm(span=10).mean().values}

        kf = KalmanFilter(initial_state_mean=raw_full[0], n_dim_obs=1)
        state_means, _ = kf.filter(raw_full)
        techniques['Kalman'] = state_means.flatten()

        def causal_savgol(x, window, poly):
            res = np.zeros_like(x)
            for i in range(window, len(x)):
                res[i] = savgol_filter(x[i-window:i], window, poly)[-1]
            return res
        techniques['Savitzky-Golay'] = causal_savgol(raw_full, 11, 3)

        def causal_wavelet(x, window=64):
            res = np.zeros_like(x)
            for i in range(window, len(x)):
                window_data = np.array(x[i-window:i], copy=True)
                coeffs = pywt.wavedec(window_data, 'db4', level=2)
                sigma = np.median(np.abs(coeffs[-1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(window_data)))
                coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
                res[i] = pywt.waverec(coeffs, 'db4')[-1]
            return res
        techniques['Wavelet'] = causal_wavelet(raw_full)

        # Strictly Causal HP Trend (using rolling window)
        def causal_hp(x, window=100, lamb=1600):
            res = np.zeros_like(x)
            for i in range(window, len(x)):
                _, trend = hpfilter(x[i-window:i], lamb=lamb)
                res[i] = trend[-1]
            return res
        techniques['HP_Filter'] = causal_hp(raw_full)

        # EMD Reconstruction (Causal)
        def causal_emd(x, window=100):
            res = np.zeros_like(x)
            emd_obj = EMD()
            for i in range(window, len(x)):
                imfs = emd_obj(x[i-window:i])
                if imfs.shape[0] > 1: res[i] = np.sum(imfs[1:], axis=0)[-1]
                else: res[i] = imfs[0][-1]
            return res
        techniques['EMD_Reconstruct'] = causal_emd(raw_full[:min(len(raw_full), 2000)]) # Truncate for speed
        if len(techniques['EMD_Reconstruct']) < len(raw_full):
            techniques['EMD_Reconstruct'] = np.concatenate([techniques['EMD_Reconstruct'], raw_full[len(techniques['EMD_Reconstruct']):]])

        top_regime_mask = (df['regime'] == top_regime)
        rdf = df[top_regime_mask].copy()
        for name, smoothed_full in techniques.items():
            smoothed = smoothed_full[top_regime_mask]
            raw_regime = raw_full[top_regime_mask]
            fwd1 = rdf['fwd_ret_1'].values
            ic1 = pd.Series(smoothed).corr(pd.Series(fwd1), method='spearman')
            snr = np.var(smoothed) / (np.var(smoothed - raw_regime) + 1e-8)
            corr = np.correlate(smoothed_full - np.mean(smoothed_full), raw_full - np.mean(raw_full), mode='full')
            lag = abs(np.argmax(corr) - (len(raw_full) - 1))
            rolling_ic = pd.Series(smoothed).rolling(50).corr(pd.Series(fwd1))
            icir = rolling_ic.mean() / (rolling_ic.std() + 1e-8) if not rolling_ic.dropna().empty else 0
            motif_wr = regime_scores[top_regime].get('Pattern Repeatability', 0.5)
            score = (abs(icir) * 0.35) + (motif_wr * 0.25) + (min(snr, 10)/10 * 0.20) - (min(lag, 20)/20 * 0.20)
            results.append({'Feature': feat, 'Technique': name, 'IC_1bar': ic1, 'ICIR': icir, 'SNR': snr, 'Lag': lag, 'Motif_winrate': motif_wr, 'Score': score})

            plt.figure(figsize=(12, 6)); plt.plot(raw_regime[:200], label='Raw', alpha=0.5); plt.plot(smoothed[:200], label=name)
            plt.title(f'{feat} - {name} ({top_regime})'); plt.legend(); plt.savefig(f'regime_research/smoothing_comparison/{feat}_{name}.png'); plt.close()

    comp_df = pd.DataFrame(results)
    comp_df.to_csv('regime_research/smoothing_comparison_matrix.csv', index=False)
    return comp_df

def phase_4_documentation(all_stats, signal_quality, regime_scores, comp_df):
    logger.info("Starting Phase 4: Documentation")
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    top_regime = max(regime_scores, key=lambda k: regime_scores[k]['Total Score'])
    winner_row = comp_df.loc[comp_df['Score'].idxmax()]

    with open('regime_research/01_regime_analysis.md', 'w') as f:
        f.write(f"# Regime Analysis Report\nRun Time: {ts}\n\n## Methodology\nRegime classification using Hurst, ADX, ATR, BB.\n\n## Distribution\n![Regime Distribution](plots/regime_distribution.png)\n")
        f.write("\n## Statistical Results\n| Regime | ADF p | LB p | Hurst | PermEn |\n|---|---|---|---|---|\n")
        for r, s in all_stats.items(): f.write(f"| {r} | {s.get('adf_p',0):.4f} | {s.get('lb_p',0):.4f} | {s.get('hurst_avg',0):.4f} | {s.get('permen',0):.4f} |\n")

    with open('regime_research/02_pattern_research.md', 'w') as f:
        f.write(f"# Pattern Research Report\nRun Time: {ts}\n\n")
        for r in regime_scores.keys(): f.write(f"### {r}\n- Motifs: ![Motifs {r}](plots/motifs_{r}.png)\n- Recurrence: ![RP {r}](plots/recurrence_{r}.png)\n- Phase Space: ![PS {r}](plots/phase_space_{r}.png)\n")

    with open('regime_research/03_smoothing_comparison.md', 'w') as f:
        f.write(f"# Smoothing Comparison Report\nRun Time: {ts}\n\n## Comparison Matrix\n{comp_df.to_markdown(index=False)}\n\n**FINAL WINNER:** {winner_row['Technique']} on {winner_row['Feature']}\n")

    with open('regime_research/00_MASTER_REPORT.md', 'w') as f:
        f.write(f"# Master Research Report\nRun Time: {ts}\n\n## Executive Summary\nBest Regime: **{top_regime}**. Best Smoothing: **{winner_row['Technique']}**.\n\n## Key Findings\n- Best regime: {top_regime} (Score: {regime_scores[top_regime]['Total Score']:.2f})\n- Best smoothing: {winner_row['Technique']}\n- Strongest signal: {winner_row['Feature']} in {top_regime}\n")
        f.write(f"\n## Dataset Blueprint\n1. Filter for {top_regime}.\n2. Apply {winner_row['Technique']} smoothing.\n3. Label with Triple Barrier (2.0*ATR / 1.0*ATR).\n4. Train LSTM.\n")

if __name__ == "__main__":
    df = phase_0()
    if df is not None:
        if len(df) > 10000: df = df.tail(10000).copy() # Use 10k for speed but representative
        df = phase_1(df)
        all_stats, signal_quality = phase_2_stats_and_signal(df)
        regime_scores = phase_2_visual_and_patterns(df, all_stats, signal_quality)
        comp_df = phase_3_smoothing_comparison(df, regime_scores, signal_quality)
        phase_4_documentation(all_stats, signal_quality, regime_scores, comp_df)
