import os
import torch
import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlphaModel:
    """Single alpha model: P(buy) vs P(sell).

    Output: 1 = buy, 0 = sell. Apply sigmoid to get P(buy).
    Decision: P(buy) >= threshold → BUY, else SELL.
    """

    def __init__(self, model_path: str):
        self.checkpoint = torch.load(model_path, map_location=DEVICE)
        self.model = self._build_model(self.checkpoint)
        self.model.eval()

        self.feature_names = self.checkpoint.get("feature_names", [])
        self.sequence_length = self.checkpoint.get("sequence_length", 25)
        self.asset_names = self.checkpoint.get("asset_names", [])

        logger.info("Alpha model loaded: %s", model_path)

    def _build_model(self, checkpoint: dict):
        from .model import AlphaSLModel, AlphaSLModelV7
        cfg = checkpoint["model_config"]
        version = checkpoint.get("format_version", 6)
        if version == 7:
            model = AlphaSLModelV7(
                input_dim=cfg["input_dim"], lstm_units=cfg["lstm_units"],
                dense_units=cfg["dense_units"], dropout=cfg["dropout"],
                num_assets=cfg.get("num_assets", 4),
                asset_embedding_dim=cfg.get("asset_embedding_dim", 4),
                num_layers=cfg.get("num_layers", 3),
                num_heads=cfg.get("num_heads", 4),
                bidirectional=cfg.get("bidirectional", True),
            )
        else:
            model = AlphaSLModel(
                input_dim=cfg["input_dim"], lstm_units=cfg["lstm_units"],
                dense_units=cfg["dense_units"], dropout=cfg["dropout"],
                num_assets=cfg.get("num_assets", 4),
                asset_embedding_dim=cfg.get("asset_embedding_dim", 4),
                num_layers=cfg.get("num_layers", 3),
                num_heads=cfg.get("num_heads", 4),
                bidirectional=cfg.get("bidirectional", True),
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(DEVICE)

    @torch.no_grad()
    def predict(self, sequences: np.ndarray, asset_ids: np.ndarray,
                threshold: float = 0.5) -> Dict[str, np.ndarray]:
        """Run model on input sequences.

        Args:
            sequences: (N, seq_len, input_dim) float32
            asset_ids: (N,) int64 asset indices
            threshold: probability threshold for buy signal

        Returns:
            dict with keys:
              - action: (N,) int — 1=buy, 0=sell
              - buy_prob: (N,) float — P(buy)
              - confidence: (N,) float — probability of chosen action
        """
        sequences_t = torch.from_numpy(sequences).float().to(DEVICE)
        assets_t = torch.from_numpy(asset_ids.astype(np.int64)).to(DEVICE)

        out = self.model(sequences_t, assets_t, return_dict=True)
        buy_probs = torch.sigmoid(out["action_logits"].float()).squeeze(-1).cpu().numpy()

        actions = (buy_probs >= threshold).astype(np.int64)
        confidence = np.where(actions, buy_probs, 1 - buy_probs)

        return {
            "action": actions,
            "buy_prob": buy_probs,
            "confidence": confidence,
        }

    @torch.no_grad()
    def predict_latest(self, sequence: np.ndarray, asset_id: int,
                       threshold: float = 0.5) -> Dict[str, float]:
        """Single-sequence convenience wrapper. Returns dict of scalars."""
        seq = sequence[np.newaxis, ...]
        aid = np.array([asset_id], dtype=np.int64)
        result = self.predict(seq, aid, threshold=threshold)
        return {k: float(v[0]) for k, v in result.items()}

    def describe(self) -> dict:
        cfg = self.checkpoint.get("model_config", {})
        return {
            "format_version": self.checkpoint.get("format_version"),
            "input_dim": cfg.get("input_dim"),
            "output_semantics": self.checkpoint.get("output_semantics"),
            "sequence_length": self.sequence_length,
            "asset_names": self.asset_names,
        }
