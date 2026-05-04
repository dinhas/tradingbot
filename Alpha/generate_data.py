import os
import sys
import argparse
import numpy as np
import logging
from datetime import datetime

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

def _build_sequences_for_asset(features: np.ndarray, labels: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Builds in-session rolling windows for one asset."""
    if len(features) < seq_len:
        return np.empty((0, seq_len, features.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    X_seq = []
    y_seq = []

    for end_idx in range(seq_len - 1, len(features)):
        start_idx = end_idx - seq_len + 1
        X_seq.append(features[start_idx:end_idx + 1])
        y_seq.append(labels[end_idx])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def generate_dataset(data_dir, output_dir, smoke_test=False, seq_len=SEQUENCE_LENGTH, limit=None):
    logger.info(f"Generating session-only dataset from {data_dir}...")
    loader = MyDataLoader(data_dir=data_dir)
    labeler = Labeler()
    engine = FeatureEngine()

    aligned_df, normalized_df = loader.get_features()

    all_sequences = []
    all_labels = []
    total_rows = 0

    for asset in loader.assets:
        logger.info(f"Processing {asset}...")
        labels_df = labeler.label_data(aligned_df, asset)
        if limit:
            labels_df = labels_df.head(limit)
        elif smoke_test:
            labels_df = labels_df.head(5000)

        common_indices = labels_df.index.intersection(normalized_df.index)
        if len(common_indices) == 0:
            logger.warning(f"No overlapping rows for {asset}; skipping.")
            continue

        filtered_norm_df = normalized_df.loc[common_indices]
        filtered_labels_df = labels_df.loc[common_indices]

        if filtered_norm_df.empty:
            logger.warning(f"No rows after filtering for {asset}; skipping.")
            continue

        asset_X = engine.get_observation_vectorized(filtered_norm_df, asset)
        asset_y = filtered_labels_df['direction'].values.astype(np.float32)

        X_seq, y_seq = _build_sequences_for_asset(asset_X, asset_y, seq_len)
        if len(y_seq) == 0:
            continue

        all_sequences.append(X_seq)
        all_labels.append(y_seq)
        asset_total = len(y_seq)

        total_rows += asset_total
        logger.info(f"{asset}: generated {asset_total} sequences.")

    if not all_sequences:
        raise RuntimeError("No training sequences were generated. Check data quality and session filters.")

    X_np = np.concatenate(all_sequences, axis=0).astype(np.float32)
    y_dir_np = np.concatenate(all_labels, axis=0).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    sequences_path = os.path.join(output_dir, "sequences.npy")
    labels_path = os.path.join(output_dir, "labels.npz")

    np.save(sequences_path, X_np)
    np.savez(labels_path, direction=y_dir_np)

    logger.info(f"Dataset generated. Total sequences: {len(X_np)}")
    return sequences_path, labels_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="Alpha/data/training_set")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=SEQUENCE_LENGTH)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.abspath(os.path.join(base_dir, args.data_dir))
    dataset_dir = os.path.abspath(os.path.join(base_dir, args.output_dir))

    generate_dataset(data_dir, dataset_dir, smoke_test=args.smoke_test, seq_len=args.seq_len, limit=args.limit)


if __name__ == "__main__":
    main()
