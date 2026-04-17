"""Frozen risk feature engine.

For evaluation/backtesting we keep this frozen module pinned to the exact same
feature implementation as Alpha to guarantee deterministic parity.
"""

from Alpha.src.feature_engine import FeatureEngine as AlphaFeatureEngine


class FeatureEngine(AlphaFeatureEngine):
    """Frozen engine with exact Alpha feature parity."""

    pass
