import os
import logging
import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from Alpha.src.model import AlphaSLModel

# --- Model Loader ---

class ModelLoader:
    """
    Loads and provides inference for the multi-head Alpha SL model.
    """
    def __init__(self):
        self.logger = logging.getLogger("LiveExecution")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha_model = None

    def load_all_models(self):
        """Loads the multi-head Alpha SL model."""
        try:
            # Alpha SL Model
            model_path = PROJECT_ROOT / "Alpha" / "models" / "alpha_model.pth"
            
            self.logger.info(f"Loading Alpha SL model from {model_path}...")
            self.alpha_model = AlphaSLModel(input_dim=40, head_a_type='tanh')
            self.alpha_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.alpha_model.to(self.device)
            self.alpha_model.eval()

            self.logger.info("Alpha SL model loaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False

    def get_alpha_predictions(self, observation):
        """
        Predicts direction score, quality, and meta probability.
        Returns: (direction_score, predicted_quality, meta_prob)
        """
        if self.alpha_model is None:
            raise RuntimeError("Alpha model not loaded.")

        # Ensure observation is 2D tensor
        if isinstance(observation, np.ndarray):
            if observation.ndim == 1:
                observation = observation.reshape(1, -1)
            obs_tensor = torch.from_numpy(observation).float().to(self.device)
        else:
            obs_tensor = observation.to(self.device)

        with torch.no_grad():
            direction, quality, meta = self.alpha_model(obs_tensor)

        return {
            'direction_score': float(direction.item()),
            'predicted_quality': float(quality.item()),
            'meta_probability': float(meta.item())
        }

    # Compatibility methods (to be deprecated if needed)
    def get_alpha_action(self, observation):
        preds = self.get_alpha_predictions(observation)
        return np.array([preds['direction_score']])

    def get_risk_action(self, observation):
        # In the new architecture, risk is derived from alpha predictions
        preds = self.get_alpha_predictions(observation)
        return {
            'sl_mult': 1.5, # Fixed as per new logic
            'tp_mult': 3.0 if preds['predicted_quality'] > 0.8 else (2.0 if preds['predicted_quality'] > 0.6 else 1.5),
            'size': preds['meta_probability'] * preds['predicted_quality']
        }
