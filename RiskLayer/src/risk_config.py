"""
Risk Configuration for Trading Environment
Centralized configuration for spreads, buffers, and risk parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import logging


@dataclass
class AssetSpreadProfile:
    """Per-asset spread configuration"""

    symbol: str
    base_spread_pips: float
    max_spread_pips: float
    pip_value: float  # 0.0001 for most pairs, 0.01 for JPY pairs and XAU

    # Session multipliers (1.0 = base spread)
    session_multipliers: Dict[str, float] = field(default_factory=dict)

    # Volatility scaling
    volatility_multiplier: float = 1.5  # Spread increase per 1 ATR move

    def get_spread(
        self, session: str, volatility_ratio: float = 1.0, is_news: bool = False
    ) -> float:
        """Calculate spread in price units"""
        session_mult = self.session_multipliers.get(session, 1.0)
        vol_mult = 1.0 + max(0, volatility_ratio - 1.0) * self.volatility_multiplier
        news_mult = 3.0 if is_news else 1.0

        spread_pips = self.base_spread_pips * session_mult * vol_mult * news_mult
        spread_pips = min(spread_pips, self.max_spread_pips)

        return spread_pips * self.pip_value


# Realistic spread profiles for each asset (ECN-like spreads)
DEFAULT_SPREAD_PROFILES: Dict[str, AssetSpreadProfile] = {
    "EURUSD": AssetSpreadProfile(
        symbol="EURUSD",
        base_spread_pips=0.1,
        max_spread_pips=2.0,
        pip_value=0.0001,
        session_multipliers={
            "asian": 1.5,
            "london": 1.0,
            "new_york": 1.0,
            "overlap_lon_ny": 0.8,
            "overlap_asia_lon": 1.2,
            "weekend": 5.0,
        },
    ),
    "GBPUSD": AssetSpreadProfile(
        symbol="GBPUSD",
        base_spread_pips=0.2,
        max_spread_pips=3.0,
        pip_value=0.0001,
        session_multipliers={
            "asian": 1.8,
            "london": 1.0,
            "new_york": 1.2,
            "overlap_lon_ny": 0.9,
            "overlap_asia_lon": 1.3,
            "weekend": 5.0,
        },
    ),
    "USDJPY": AssetSpreadProfile(
        symbol="USDJPY",
        base_spread_pips=0.2,
        max_spread_pips=3.0,
        pip_value=0.01,
        session_multipliers={
            "asian": 1.0,
            "london": 1.3,
            "new_york": 1.2,
            "overlap_lon_ny": 1.0,
            "overlap_asia_lon": 0.9,
            "weekend": 5.0,
        },
    ),
    "USDCHF": AssetSpreadProfile(
        symbol="USDCHF",
        base_spread_pips=0.3,
        max_spread_pips=4.0,
        pip_value=0.0001,
        session_multipliers={
            "asian": 2.0,
            "london": 1.2,
            "new_york": 1.3,
            "overlap_lon_ny": 1.0,
            "overlap_asia_lon": 1.5,
            "weekend": 6.0,
        },
    ),
    "XAUUSD": AssetSpreadProfile(
        symbol="XAUUSD",
        base_spread_pips=2.0,
        max_spread_pips=20.0,
        pip_value=0.01,
        session_multipliers={
            "asian": 2.0,
            "london": 1.0,
            "new_york": 1.0,
            "overlap_lon_ny": 0.8,
            "overlap_asia_lon": 1.5,
            "weekend": 10.0,
        },
    ),
}


@dataclass
class BreathingRoomConfig:
    """
    Buffer configuration for SL placement.
    Prevents immediate SL hits from spread and gives trades room to breathe.
    """

    min_spread_buffer_pips: float = 1.0  # Minimum buffer in pips
    atr_buffer_multiplier: float = 0.3  # Additional buffer as % of ATR
    min_atr_buffer: float = 0.0005  # Minimum absolute buffer

    def calculate_sl_buffer(self, spread: float, atr: float) -> float:
        """Calculate total buffer to add to SL distance"""
        # Buffer at least equal to spread (protect against immediate SL hit)
        spread_buffer = spread * 2.0  # Double spread for safety

        # Add ATR-based buffer for noise
        atr_buffer = atr * self.atr_buffer_multiplier

        # Return the larger of spread buffer or ATR buffer
        return max(spread_buffer, atr_buffer, self.min_atr_buffer)


@dataclass
class RiskConfig:
    """Master risk configuration"""

    spread_profiles: Dict[str, AssetSpreadProfile] = field(
        default_factory=lambda: DEFAULT_SPREAD_PROFILES
    )
    breathing_room: BreathingRoomConfig = field(default_factory=BreathingRoomConfig)

    # Slippage settings (for backtest realism)
    slippage_min_pips: float = 0.5
    slippage_max_pips: float = 1.5
    enable_slippage: bool = True

    # Risk limits
    max_position_pct: float = 0.50
    max_total_exposure: float = 0.60
    drawdown_limit: float = 0.25

    def get_session(self, hour_utc: int, is_weekend: bool = False) -> str:
        """Determine trading session from UTC hour"""
        if is_weekend:
            return "weekend"

        if 13 <= hour_utc < 16:
            return "overlap_lon_ny"
        elif 8 <= hour_utc < 9:
            return "overlap_asia_lon"
        elif 8 <= hour_utc < 16:
            return "london"
        elif 16 <= hour_utc < 21:
            return "new_york"
        else:
            return "asian"

    def get_spread(
        self,
        asset: str,
        price: float,
        atr: float,
        hour_utc: int = 12,
        is_weekend: bool = False,
        is_news: bool = False,
    ) -> float:
        """
        Get realistic spread for an asset.

        Args:
            asset: Asset symbol (e.g., 'EURUSD')
            price: Current price (used for volatility calculation)
            atr: Current ATR value
            hour_utc: Current hour in UTC (for session detection)
            is_weekend: Whether it's weekend
            is_news: Whether there's high-impact news

        Returns:
            Spread in price units (not pips)
        """
        session = self.get_session(hour_utc, is_weekend)

        # Calculate volatility ratio
        typical_atr = price * 0.001  # ~0.1% typical
        volatility_ratio = atr / typical_atr if typical_atr > 0 else 1.0

        profile = self.spread_profiles.get(asset)
        if profile:
            return profile.get_spread(session, volatility_ratio, is_news)
        else:
            # Fallback: 2 pips for unknown assets
            return price * 0.0002

    @classmethod
    def load_from_yaml(cls, path: str) -> "RiskConfig":
        """Load configuration from YAML file"""
        try:
            import yaml
        except ImportError:
            logging.warning("PyYAML not installed, using defaults")
            return cls()

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            logging.warning(f"Config file not found: {path}, using defaults")
            return cls()

        # Parse spread profiles
        spread_profiles = {}
        for symbol, cfg in data.get("spreads", {}).items():
            spread_profiles[symbol] = AssetSpreadProfile(
                symbol=symbol,
                base_spread_pips=cfg.get("base_pips", 0.2),
                max_spread_pips=cfg.get("max_pips", 5.0),
                pip_value=cfg.get("pip_value", 0.0001),
                session_multipliers=cfg.get("sessions", {}),
            )

        # Parse breathing room config
        br_cfg = data.get("breathing_room", {})
        breathing_room = BreathingRoomConfig(
            min_spread_buffer_pips=br_cfg.get("min_spread_buffer_pips", 1.0),
            atr_buffer_multiplier=br_cfg.get("atr_buffer_multiplier", 0.3),
            min_atr_buffer=br_cfg.get("min_atr_buffer", 0.0005),
        )

        # Parse slippage and risk limits
        slip_cfg = data.get("slippage", {})
        risk_cfg = data.get("risk_limits", {})

        return cls(
            spread_profiles=spread_profiles
            if spread_profiles
            else DEFAULT_SPREAD_PROFILES,
            breathing_room=breathing_room,
            slippage_min_pips=slip_cfg.get("min_pips", 0.5),
            slippage_max_pips=slip_cfg.get("max_pips", 1.5),
            enable_slippage=slip_cfg.get("enabled", True),
            max_position_pct=risk_cfg.get("max_position_pct", 0.50),
            max_total_exposure=risk_cfg.get("max_total_exposure", 0.60),
            drawdown_limit=risk_cfg.get("drawdown_limit", 0.25),
        )


# Singleton instance for easy import
DEFAULT_RISK_CONFIG = RiskConfig()
