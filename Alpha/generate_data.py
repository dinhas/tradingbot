import os
import sys
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from joblib import Parallel, delayed

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Alpha.src.data_loader import DataLoader as MyDataLoader
from Alpha.src.labeling import Labeler
from Alpha.src.feature_engine import FeatureEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 50
USE_RESEARCH_PIPELINE = True

def _build_sequences_for_asset(features: np.ndarray, labels: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if len(features) < seq_len:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X_seq, y_seq = [], []
    for end_idx in range(seq_len - 1, len(features)):
        X_seq.append(features[end_idx - seq_len + 1 : end_idx + 1])
        y_seq.append(labels[end_idx])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)

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
        if confirmed_regime in ['RANGING', 'TRENDING']:
            mask.iloc[i] = True
    return mask

def process_asset_data(asset, aligned_df, normalized_df, engine, labeler, seq_len):
    logger.info(f"Processing {asset}...")
    stable_ranging_mask = get_stable_regime_mask(aligned_df[f"{asset}_regime"], min_duration=20)
    df_ranging = aligned_df[stable_ranging_mask].copy()

    if df_ranging.empty:
        logger.warning(f"No stable RANGING bars for {asset}; skipping.")
        return None, None, None

    df_ranging['label'] = labeler.label_data(aligned_df, asset)['direction']

    # Boundary Purge
    ranging_start = (stable_ranging_mask == True) & (stable_ranging_mask.shift(1) == False)
    ranging_end = (stable_ranging_mask == False) & (stable_ranging_mask.shift(1) == True)
    transition_idx = aligned_df.index[ranging_start | ranging_end]

    purge_set = set()
    for t in transition_idx:
        loc = aligned_df.index.get_loc(t)
        for offset in range(-5, 6):
            if 0 <= loc + offset < len(aligned_df):
                purge_set.add(aligned_df.index[loc + offset])

    df_clean = df_ranging[~df_ranging.index.isin(purge_set)].copy()
    df_clean = df_clean[df_clean['label'] != 0]

    fwd_ret = aligned_df[f"{asset}_close"].shift(-20) / aligned_df[f"{asset}_close"] - 1
    min_move = 0.40 * df_clean[f"{asset}_atr_kalman"]
    df_clean = df_clean[fwd_ret.loc[df_clean.index].abs() >= min_move]

    long_df = df_clean[df_clean['label'] == 1]
    short_df = df_clean[df_clean['label'] == -1]
    if not long_df.empty and not short_df.empty:
        majority = long_df if len(long_df) > len(short_df) else short_df
        minority = short_df if majority is long_df else long_df
        max_majority = int(len(minority) * 0.55 / 0.45)
        if len(majority) > max_majority:
            majority = majority.tail(max_majority)
        df_clean = pd.concat([long_df if majority is short_df else majority,
                            short_df if majority is long_df else majority]).sort_index()

    if df_clean.empty:
        return None, None, None

    asset_feat_cols = [f"{asset}_{c}" for c in engine.feature_names]
    asset_X = normalized_df.loc[df_clean.index, asset_feat_cols].values
    asset_y = df_clean['label'].values.astype(np.float32)

    X_seq, y_seq = _build_sequences_for_asset(asset_X, asset_y, seq_len)

    df_check = df_clean.copy()
    df_check['asset'] = asset

    return X_seq, y_seq, df_check

def generate_data():
    data_dir = os.path.join(PROJECT_ROOT, "data")
    output_dir = os.path.join(PROJECT_ROOT, "Alpha", "data", "training_set")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Generating dataset from {data_dir} (Pipeline: Research)...")
    loader = MyDataLoader(data_dir=data_dir)
    engine = FeatureEngine(use_research_pipeline=USE_RESEARCH_PIPELINE)
    labeler = Labeler(use_research_pipeline=USE_RESEARCH_PIPELINE)

    # Parallelize feature calculation
    raw_data = loader.load_raw_data()
    aligned_df = engine._align_data(raw_data)

    logger.info("Calculating features in parallel...")
    asset_features_list = Parallel(n_jobs=4)(
        delayed(engine._get_asset_features)(aligned_df, asset) for asset in loader.assets
    )

    all_new_cols = {}
    for asset_cols in asset_features_list:
        all_new_cols.update(asset_cols)

    numeric_cols = {k: v for k, v in all_new_cols.items() if not isinstance(v.iloc[0], str)}
    categ_cols = {k: v for k, v in all_new_cols.items() if isinstance(v.iloc[0], str)}

    new_features_df = pd.DataFrame(numeric_cols, index=aligned_df.index).astype(np.float32)
    if categ_cols:
        categ_df = pd.DataFrame(categ_cols, index=aligned_df.index)
        new_features_df = pd.concat([new_features_df, categ_df], axis=1)

    aligned_df = pd.concat([aligned_df, new_features_df], axis=1)
    normalized_df = aligned_df.select_dtypes(include=[np.number]).copy()
    normalized_df = normalized_df.ffill().fillna(0).astype(np.float32)

    logger.info("Processing assets for sequences in parallel...")
    results = Parallel(n_jobs=4)(
        delayed(process_asset_data)(asset, aligned_df, normalized_df, engine, labeler, SEQUENCE_LENGTH)
        for asset in loader.assets
    )

    all_sequences = [r[0] for r in results if r[0] is not None]
    all_labels = [r[1] for r in results if r[1] is not None]
    all_final_df_list = [r[2] for r in results if r[2] is not None]

    if not all_sequences:
        raise RuntimeError("No sequences generated.")

    X_np = np.concatenate(all_sequences, axis=0).astype(np.float32)
    y_dir_np = np.concatenate(all_labels, axis=0).astype(np.float32)

    df_final_all = pd.concat(all_final_df_list).sort_index()

    n_final = len(y_dir_np)
    train_end = int(n_final * 0.70)
    val_end = int(n_final * 0.85)

    def save_df_parquet(df_part, filename):
        out_df = pd.DataFrame(index=df_part.index)
        for c in engine.feature_names:
            col_vals = []
            for idx, row in df_part.iterrows():
                col_vals.append(row[f"{row['asset']}_{c}"])
            out_df[c] = col_vals
        out_df['label'] = df_part['label'].values
        out_df.to_parquet(os.path.join(output_dir, filename))
        return out_df

    train_p = df_final_all.iloc[:train_end]
    val_p = df_final_all.iloc[train_end:val_end]
    test_p = df_final_all.iloc[val_end:]

    save_df_parquet(train_p, 'train_dataset.parquet')
    save_df_parquet(val_p, 'val_dataset.parquet')
    save_df_parquet(test_p, 'test_dataset.parquet')

    np.save(os.path.join(output_dir, "sequences.npy"), X_np)
    np.savez(os.path.join(output_dir, "labels.npz"), direction=y_dir_np)

    logger.info(f"Dataset generated and saved to {output_dir}")
    logger.info(f"Total sequences: {len(X_np)}")
    logger.info(f"Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

if __name__ == "__main__":
    generate_data()
