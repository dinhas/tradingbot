import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import logging

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from Alpha.src.model import AlphaSLModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DEVICE = torch.device("cpu")
SEQUENCE_LENGTH = 50

class AlphaSequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_seq = self.sequences[idx].copy()
        y_dir = self.labels[idx]
        return torch.from_numpy(x_seq), torch.tensor(y_dir, dtype=torch.float32)

def evaluate_model():
    dataset_dir = os.path.join(PROJECT_ROOT, "Alpha", "data", "training_set")
    model_path = os.path.join(PROJECT_ROOT, "Alpha", "models", "alpha_model.pth")
    test_parquet = os.path.join(dataset_dir, "test_dataset.parquet")

    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return

    # Load test data from sequences.npy and labels.npz for consistency with training
    sequences = np.load(os.path.join(dataset_dir, "sequences.npy"))
    labels_data = np.load(os.path.join(dataset_dir, "labels.npz"))
    directions = labels_data['direction']

    # Use the same split logic as in the generation script
    n_final = len(directions)
    val_end = int(n_final * 0.85)
    X_test = sequences[val_end:]
    y_test = directions[val_end:]

    test_dataset = AlphaSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    input_dim = X_test.shape[-1]
    model = AlphaSLModel(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for b_X, b_dir in test_loader:
            logits = model(b_X)
            preds = torch.argmax(logits, dim=1) - 1 # Map [0, 1, 2] -> [-1, 0, 1]
            all_preds.extend(preds.numpy())
            all_targets.extend(b_dir.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = np.mean(all_preds == all_targets)

    logger.info("\n[MODEL EVALUATION]")
    logger.info(f"Test Accuracy: {accuracy:.4f}")

    from collections import Counter
    pred_counts = Counter(all_preds.tolist())
    target_counts = Counter(all_targets.tolist())

    logger.info("\nPredicted Class Distribution:")
    for val in [-1.0, 0.0, 1.0]:
        cnt = pred_counts.get(val, 0)
        logger.info(f"Class {int(val)}: {cnt} ({cnt/len(all_preds)*100:.2f}%)")

    logger.info("\nTarget Class Distribution:")
    for val in [-1.0, 0.0, 1.0]:
        cnt = target_counts.get(val, 0)
        logger.info(f"Class {int(val)}: {cnt} ({cnt/len(all_targets)*100:.2f}%)")

if __name__ == "__main__":
    evaluate_model()
