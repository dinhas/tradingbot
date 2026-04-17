"""Risk feature engine.

This module intentionally re-uses Alpha's feature engineering stack so the
risk-model pipeline receives an identical feature set (including wavelet
smoothing behavior and normalization settings).
"""

from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine


class FeatureEngine(AlphaFeatureEngine):
    """Risk-layer feature engine that is 100% aligned with Alpha features."""

    pass
