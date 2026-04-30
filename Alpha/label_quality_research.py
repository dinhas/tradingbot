import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Alpha.src.data_loader import DataLoader as MyDataLoader
from Alpha.src.labeling import Labeler
from Alpha.src.feature_engine import FeatureEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PLOT_DIR = "./regime_research/plots/label_quality/"
os.makedirs(PLOT_DIR, exist_ok=True)

def calculate_snr(returns):
    if len(returns) < 2: return 0.0
    signal = np.abs(np.mean(returns))
    noise = np.std(returns)
    return signal / noise if noise > 1e-9 else 0.0

def calculate_cohens_d(long_rets, short_rets):
    if len(long_rets) < 2 or len(short_rets) < 2: return 0.0
    var1, var2 = np.var(long_rets, ddof=1), np.var(short_rets, ddof=1)
    pooled_std = np.sqrt((var1 + var2) / 2)
    return np.abs(np.mean(long_rets) - np.mean(short_rets)) / pooled_std if pooled_std > 1e-9 else 0.0

def get_metrics(df, label_col, ret_col='fwd_ret_20'):
    sub = df.dropna(subset=[label_col])
    long_rets = sub[sub[label_col] == 1][ret_col].dropna()
    short_rets = sub[sub[label_col] == -1][ret_col].dropna()

    snr_long = calculate_snr(long_rets)
    snr_short = calculate_snr(short_rets)
    combined_snr = (snr_long + snr_short) / 2
    d = calculate_cohens_d(long_rets, short_rets)

    counts = sub[label_col].value_counts()
    imbalance = counts.max() / counts.min() if not counts.empty and counts.min() > 0 else 0
    expiry_pct = (counts.get(0, 0) / len(sub) * 100) if len(sub) > 0 else 0

    return {
        'snr_long': snr_long,
        'snr_short': snr_short,
        'combined_snr': combined_snr,
        'cohen_d': d,
        'imbalance': imbalance,
        'expiry_pct': expiry_pct,
        'count': len(sub)
    }

def get_stable_regime_mask(regime_series, min_duration=20):
    mask = pd.Series(False, index=regime_series.index)
    current_regime = None
    streak = 0
    confirmed_regime = None
    for i, val in enumerate(regime_series):
        if val == current_regime:
            streak += 1
        else:
            current_regime = val
            streak = 1
        if streak >= min_duration:
            confirmed_regime = current_regime
        if confirmed_regime == 'RANGING':
            mask.iloc[i] = True
    return mask

def run_research():
    logger.info("PHASE 0: Setup & Data Loading")
    loader = MyDataLoader(data_dir="./data")
    engine = FeatureEngine(use_research_pipeline=True)

    aligned_df, _ = loader.get_features(engine=engine)
    total_bars = len(aligned_df)

    asset_data_list = []

    for asset in loader.assets:
        stable_mask = get_stable_regime_mask(aligned_df[f"{asset}_regime"], min_duration=20)
        df_asset = aligned_df[stable_mask].copy()
        if df_asset.empty: continue

        # Calculate forward returns
        fwd_ret = aligned_df[f"{asset}_close"].shift(-20) / aligned_df[f"{asset}_close"] - 1
        df_asset['fwd_ret_20'] = fwd_ret.loc[df_asset.index]
        df_asset['asset'] = asset

        # PHASE 1 & Fix 1 labels for this asset
        close = aligned_df[f"{asset}_close"].values
        high = aligned_df[f"{asset}_high"].values
        low = aligned_df[f"{asset}_low"].values
        atr_raw = aligned_df[f"{asset}_atr"].values
        atr_kalman = aligned_df[f"{asset}_atr_kalman"].values

        l_raw = np.zeros(len(aligned_df))
        l_fix1 = np.zeros(len(aligned_df))

        for i in range(len(aligned_df) - 20):
            # Raw
            pt, sl = close[i] + 2.0*atr_raw[i], close[i] - 1.0*atr_raw[i]
            found = False
            for j in range(i+1, i+21):
                if high[j] >= pt: l_raw[i] = 1; found = True; break
                if low[j] <= sl: l_raw[i] = -1; found = True; break
            if not found: l_raw[i] = 0

            # Fix 1
            ptk, slk = close[i] + 2.0*atr_kalman[i], close[i] - 1.0*atr_kalman[i]
            found = False
            for j in range(i+1, i+21):
                if high[j] >= ptk: l_fix1[i] = 1; found = True; break
                if low[j] <= slk: l_fix1[i] = -1; found = True; break
            if not found: l_fix1[i] = 0

        df_asset['label_raw'] = l_raw[aligned_df.index.get_indexer(df_asset.index)]
        df_asset['label_fix1'] = l_fix1[aligned_df.index.get_indexer(df_asset.index)]

        # Fix 2: Boundary Purging
        df_asset['label_fix2'] = df_asset['label_fix1']
        r_start = (stable_mask == True) & (stable_mask.shift(1) == False)
        r_end = (stable_mask == False) & (stable_mask.shift(1) == True)
        t_idx = aligned_df.index[r_start | r_end]
        purge_set = set()
        for t in t_idx:
            loc = aligned_df.index.get_loc(t)
            for offset in range(-5, 6):
                if 0 <= loc + offset < len(aligned_df): purge_set.add(aligned_df.index[loc + offset])
        to_purge = df_asset.index.intersection(list(purge_set))
        df_asset.loc[to_purge, 'label_fix2'] = np.nan

        # Fix 3: Drop Time Expiry
        df_asset['label_fix3'] = df_asset['label_fix2']
        df_asset.loc[df_asset['label_fix3'] == 0, 'label_fix3'] = np.nan

        # Fix 4: Min Return Threshold
        df_asset['label_fix4'] = df_asset['label_fix3']
        min_move = 0.40 * aligned_df[f"{asset}_atr_kalman"].loc[df_asset.index]
        df_asset.loc[df_asset['fwd_ret_20'].abs() < min_move, 'label_fix4'] = np.nan

        asset_data_list.append(df_asset)

    working_df = pd.concat(asset_data_list)
    logger.info(f"Total Stable RANGING bars: {len(working_df)}")

    # PHASE 2-4: Global Measurement
    baseline_metrics = get_metrics(working_df, 'label_raw')
    fix1_metrics = get_metrics(working_df, 'label_fix1')
    fix2_metrics = get_metrics(working_df, 'label_fix2')
    fix3_metrics = get_metrics(working_df, 'label_fix3')

    fix4_metrics = get_metrics(working_df, 'label_fix4')
    if fix4_metrics['count'] < 10000: # Lowered threshold for 10% data
        logger.warning(f"Fix 4 aggressive: reverting. Count={fix4_metrics['count']}")
        working_df['label_fix4'] = working_df['label_fix3']
        fix4_metrics = get_metrics(working_df, 'label_fix4')

    # Combined & Undersampling
    df_clean = working_df.dropna(subset=['label_fix4']).copy()
    long_df = df_clean[df_clean['label_fix4'] == 1]
    short_df = df_clean[df_clean['label_fix4'] == -1]
    if not long_df.empty and not short_df.empty:
        maj = long_df if len(long_df) > len(short_df) else short_df
        min_df = short_df if maj is long_df else long_df
        max_maj = int(len(min_df) * 0.55 / 0.45)
        if len(maj) > max_maj: maj = maj.tail(max_maj)
        df_clean = pd.concat([maj, min_df]).sort_index()

    final_metrics = get_metrics(df_clean, 'label_fix4')

    fix_results = [
        ('Baseline', baseline_metrics),
        ('Fix 1', fix1_metrics),
        ('Fix 2', fix2_metrics),
        ('Fix 3', fix3_metrics),
        ('Fix 4', fix4_metrics),
        ('Combined', final_metrics)
    ]

    # Plots
    plt.figure(figsize=(10, 6))
    for val, name in [(1, 'Long'), (-1, 'Short')]:
        sub = df_clean[df_clean['label_fix4'] == val]['fwd_ret_20']
        if not sub.empty: sns.histplot(sub, label=name, kde=True, element="step")
    plt.title("Final Return Distribution"); plt.legend(); plt.savefig(os.path.join(PLOT_DIR, "final_return_dist.png")); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot([r[0] for r in fix_results], [r[1]['combined_snr'] for r in fix_results], marker='o')
    plt.title("Label SNR Progression"); plt.grid(True); plt.savefig(os.path.join(PLOT_DIR, "snr_progression.png")); plt.close()

    # PHASE 6: Documentation
    report = f"""# Label Quality Analysis Report
Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Source Data: {total_bars} bars
Regime Filter: Stable RANGING ({len(working_df)} bars)

## Executive Summary
Final Label SNR reached {final_metrics['combined_snr']:.4f} (Baseline: {baseline_metrics['combined_snr']:.4f}).
Cohen's d improved to {final_metrics['cohen_d']:.4f}. Dataset size: {len(df_clean)} samples.

## Fix-by-Fix Improvement
| Fix | Label SNR | Cohen's d | Labels Remaining |
|-----|-----------|-----------|-----------------|
| Baseline | {baseline_metrics['combined_snr']:.4f} | {baseline_metrics['cohen_d']:.4f} | {baseline_metrics['count']} |
| Fix 1 | {fix1_metrics['combined_snr']:.4f} | {fix1_metrics['cohen_d']:.4f} | {fix1_metrics['count']} |
| Fix 2 | {fix2_metrics['combined_snr']:.4f} | {fix2_metrics['cohen_d']:.4f} | {fix2_metrics['count']} |
| Fix 3 | {fix3_metrics['combined_snr']:.4f} | {fix3_metrics['cohen_d']:.4f} | {fix3_metrics['count']} |
| Fix 4 | {fix4_metrics['combined_snr']:.4f} | {fix4_metrics['cohen_d']:.4f} | {fix4_metrics['count']} |
| Combined | {final_metrics['combined_snr']:.4f} | {final_metrics['cohen_d']:.4f} | {final_metrics['count']} |

![SNR Progression](./plots/label_quality/snr_progression.png)
"""
    with open("./regime_research/04_label_quality.md", "w") as f: f.write(report)
    print(f"\nLABEL QUALITY RESEARCH COMPLETE. Report generated at ./regime_research/04_label_quality.md")

if __name__ == "__main__":
    run_research()
