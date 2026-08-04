import os
import sys
import logging
import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from Alpha.src.model import AlphaSLModelV7

# Add Filter/src to path for feature engine
_filter_src = Path(__file__).resolve().parent.parent.parent / "Filter" / "src"
if str(_filter_src) not in sys.path:
    sys.path.insert(0, str(_filter_src))

from Filter.src.feature_engine import FeatureEngine as FilterFeatureEngine


# --- Model Loader ---

class ModelLoader:
    """
    Loads and provides inference for Alpha (LSTM/V7) and Filter (RF ensemble) models.
    No risk model — SL/TP are fixed ATR multipliers.
    """

    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Alpha model
        self.alpha_model = None
        self.alpha_threshold = 0.60
        self.alpha_sequence_length = 25

        # RF Filter ensemble
        self.filter_ensemble = None
        self.filter_threshold = 0.72
        self.filter_fe = FilterFeatureEngine()

        # Default SL/TP multipliers (ATR-based)
        self.sl_multiplier = 2.0
        self.tp_multiplier = 4.0

    def load_all_models(self):
        """Loads alpha model and RF filter ensemble."""
        try:
            # 1. Alpha Model — single buy-vs-sell LSTM/V7
            alpha_path = self.project_root / "Alpha" / "models" / "alpha_model.pth"
            if not alpha_path.exists():
                alpha_path = self.project_root / "models" / "alpha" / "alpha_model.pth"
            if alpha_path.exists():
                self.logger.info(f"Loading Alpha model from {alpha_path}...")
                alpha_checkpoint = torch.load(alpha_path, map_location=self.device)
                self.alpha_model = self._build_alpha_model(alpha_checkpoint)
                self.alpha_model.eval()
                self.alpha_sequence_length = alpha_checkpoint.get("sequence_length", 25)
            else:
                self.logger.error(f"Alpha model not found at {alpha_path}. Cannot trade.")
                return False

            # 2. RF Filter Ensemble
            filter_path = self.project_root / "Filter" / "models" / "filter_rf_ensemble.joblib"
            if filter_path.exists():
                try:
                    self.logger.info(f"Loading RF Filter ensemble from {filter_path}...")
                    self.filter_ensemble = joblib.load(filter_path)
                    # Override threshold from ensemble if saved, else use default
                    saved_threshold = self.filter_ensemble.get("threshold", self.filter_threshold)
                    if saved_threshold:
                        self.filter_threshold = float(saved_threshold)
                    self.logger.info(f"Filter ensemble loaded. Threshold={self.filter_threshold:.3f}")
                except Exception as fe:
                    self.logger.warning(f"Could not load RF Filter ensemble from {filter_path} due to serialization/version incompatibility: {fe}. Filter gate will be disabled.")
                    self.filter_ensemble = None
            else:
                self.logger.warning(f"Filter ensemble not found at {filter_path}. Filter gate disabled.")

            self.logger.info(
                f"All models loaded. Alpha thresh={self.alpha_threshold}, "
                f"Filter thresh={self.filter_threshold}, "
                f"SL={self.sl_multiplier}x ATR, TP={self.tp_multiplier}x ATR"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False

    def _build_alpha_model(self, checkpoint):
        """Build V7 alpha model from checkpoint."""
        version = checkpoint.get("format_version", 7)
        cfg = checkpoint.get("model_config", {})
        if version != 7:
            self.logger.error(
                f"Unsupported model format_version={version}. Only V7 is supported."
            )
            return None
        model = AlphaSLModelV7(
            input_dim=cfg["input_dim"], lstm_units=cfg["lstm_units"],
            dense_units=cfg["dense_units"], dropout=cfg["dropout"],
            num_assets=cfg.get("num_assets", 4),
            asset_embedding_dim=cfg.get("asset_embedding_dim", 4),
            num_layers=cfg.get("num_layers", 3),
            num_heads=cfg.get("num_heads", 4),
            bidirectional=cfg.get("bidirectional", True),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.to(self.device)

    def get_alpha_signal(self, observation, threshold=None):
        """Run alpha model: P(buy) vs P(sell).

        Returns dict with:
          - action: 1=buy, -1=sell
          - buy_prob: raw P(buy)
          - confidence: probability of the chosen action
        """
        if self.alpha_model is None:
            raise RuntimeError("Alpha model not loaded.")

        if observation.ndim == 2:
            observation = observation.reshape(1, observation.shape[0], observation.shape[1])
        elif observation.ndim != 3:
            raise ValueError("Observation must have shape [seq, features] or [batch, seq, features].")

        if threshold is None:
            threshold = self.alpha_threshold

        obs_tensor = torch.from_numpy(observation.astype(np.float32)).to(self.device)
        asset_ids = torch.zeros(obs_tensor.shape[0], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self.alpha_model(obs_tensor, asset_ids, return_dict=True)
            buy_prob = torch.sigmoid(out["action_logits"].float()).squeeze(-1)

        buy_p = buy_prob.cpu().numpy()
        action = np.where(buy_p >= threshold, 1, -1)
        confidence = np.where(buy_p >= threshold, buy_p, 1 - buy_p)

        return {
            "direction": action.astype(float),
            "confidence": confidence.astype(float),
            "action": action.astype(int),
            "buy_prob": buy_p.astype(float),
        }

    def get_filter_signal(self, filter_features):
        """Run RF filter ensemble on 26-feature observation.

        Args:
            filter_features: numpy array of shape (26,) or (1, 26) — latest bar features.

        Returns dict with:
          - should_trade: bool — True if filter confidence >= threshold
          - confidence: float — RF ensemble probability
          - direction: int — +1 (buy-side confident) or -1 (sell-side confident)
        """
        if self.filter_ensemble is None:
            # No filter loaded — always pass
            return {"should_trade": True, "confidence": 1.0, "direction": 0}

        if filter_features.ndim == 1:
            filter_features = filter_features.reshape(1, -1)

        rf1 = self.filter_ensemble["rf1"]
        rf2 = self.filter_ensemble["rf2"]
        gb = self.filter_ensemble["gb"]
        meta = self.filter_ensemble["meta"]

        rf1_p = rf1.predict_proba(filter_features)[:, 1]
        rf2_p = rf2.predict_proba(filter_features)[:, 1]
        gb_p = gb.predict_proba(filter_features)[:, 1]

        meta_input = np.column_stack([rf1_p, rf2_p, gb_p])
        rf_prob = meta.predict_proba(meta_input)[:, 1]

        confidence = float(rf_prob[0])
        should_trade = confidence >= self.filter_threshold

        return {
            "should_trade": should_trade,
            "confidence": confidence,
            "direction": 0,  # direction determined by alpha model
        }

    def build_filter_features(self, data_dict):
        """Build the 26-feature vector for the RF filter from OHLCV data.

        Args:
            data_dict: dict of {asset: DataFrame} with OHLCV columns.

        Returns:
            numpy array of shape (n_rows, 26) from the FilterFeatureEngine.
        """
        try:
            _, normalized_df = self.filter_fe.preprocess_data(data_dict)
            # Use EURUSD as the representative asset for filter features
            # (filter is applied per-bar, direction is determined by alpha)
            asset = "EURUSD"
            if asset not in normalized_df.columns and f"{asset}_volatility" not in normalized_df.columns:
                # Fallback: return zeros
                return np.zeros((1, 26), dtype=np.float32)
            obs = self.filter_fe.get_observation_vectorized(normalized_df, asset)
            return obs
        except Exception as e:
            self.logger.error(f"Filter feature build error: {e}")
            return np.zeros((1, 26), dtype=np.float32)
