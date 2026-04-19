"""Risk-layer feature engine.

Now inherits the cleaner Kalman-based and Robust Scaling logic 
directly from the Alpha FeatureEngine to maintain 100% parity.
"""

from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine

class FeatureEngine(AlphaFeatureEngine):
    """Risk-layer feature engine aligned with Alpha's causal Kalman logic."""
    pass
