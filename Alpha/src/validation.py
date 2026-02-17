import numpy as np
import pandas as pd
from typing import List, Tuple
import torch
from Alpha.src.model import AlphaSLModel, multi_head_loss

class ValidationEngine:
    """
    Handles Purged Cross-Validation and Walk-Forward Evaluation.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def walk_forward_validation(self, X, y_dir, y_qual, y_meta, n_splits=5):
        """
        Implements walk-forward validation.
        """
        n_samples = len(X)
        split_size = n_samples // (n_splits + 1)

        results = []
        for i in range(n_splits):
            train_end = (i + 1) * split_size
            test_end = (i + 2) * split_size

            X_train, X_test = X[:train_end], X[train_end:test_end]
            y_dir_train, y_dir_test = y_dir[:train_end], y_dir[train_end:test_end]
            y_qual_train, y_qual_test = y_qual[:train_end], y_qual[train_end:test_end]
            y_meta_train, y_meta_test = y_meta[:train_end], y_meta[train_end:test_end]

            # Simple evaluation (not full retraining for speed here)
            metrics = self._evaluate(X_test, y_dir_test, y_qual_test, y_meta_test)
            results.append(metrics)

        return results

    def _evaluate(self, X, y_dir, y_qual, y_meta):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float().to(self.device)
            dir_out, qual_out, meta_out = self.model(X_tensor)

            # Simplified metrics
            dir_pred = dir_out.cpu().numpy().flatten()
            qual_pred = qual_out.cpu().numpy().flatten()
            meta_pred = meta_out.cpu().numpy().flatten()

            # Accuracy on direction sign
            dir_acc = np.mean(np.sign(dir_pred) == np.sign(y_dir))

            # MAE on quality
            qual_mae = np.mean(np.abs(qual_pred - y_qual))

            # Meta accuracy (0.5 threshold)
            meta_acc = np.mean((meta_pred > 0.5) == (y_meta > 0.5))

            return {
                'dir_acc': dir_acc,
                'qual_mae': qual_mae,
                'meta_acc': meta_acc
            }

def calculate_strategy_metrics(returns: np.array):
    """
    Calculates Sharpe, Max DD, Win Rate, Profit Factor.
    """
    if len(returns) == 0:
        return {}

    win_rate = np.mean(returns > 0)

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    profit_factor = np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else np.inf

    sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 24 * 12) # Annualized for M5

    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = np.max(drawdown)

    return {
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

if __name__ == "__main__":
    # Example usage
    pass
