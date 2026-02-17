import numpy as np
import logging

class RiskEngine:
    """
    Probabilistic Risk Engine for capital allocation and trade management.
    Redesigned to eliminate lookahead and use multi-head Alpha model outputs.
    """
    def __init__(self, config=None):
        self.logger = logging.getLogger("RiskEngine")

        # Default Configuration
        self.cfg = {
            'meta_threshold': 0.70,
            'direction_threshold': 0.10,
            'quality_threshold': 0.30,
            'k_sl': 1.5,
            'base_risk_percent': 0.01, # 1% of balance
            'max_risk_cap': 0.05,      # 5% max risk
            'min_required_rr': 1.0,
            'max_leverage': 100.0
        }

        if config:
            # Update with provided config (case-insensitive keys)
            self.cfg.update({k.lower(): v for k, v in config.items() if k.lower() in self.cfg})

    def get_trade_decision(self, asset_name, entry_price, atr, direction_score, quality_score, meta_prob, balance, digits):
        """
        Calculates position sizing and SL/TP prices based on probabilistic edge.
        Returns: Dict containing decision and parameters, or None if rejected.
        """
        # 1. Trade Filtering (Phase 3)
        if meta_prob < self.cfg['meta_threshold']:
            return {'action': 0, 'reason': f'Meta Filter ({meta_prob:.3f} < {self.cfg["meta_threshold"]})'}

        if abs(direction_score) < self.cfg['direction_threshold']:
            return {'action': 0, 'reason': f'Direction Filter ({abs(direction_score):.3f} < {self.cfg["direction_threshold"]})'}

        if quality_score < self.cfg['quality_threshold']:
            return {'action': 0, 'reason': f'Quality Filter ({quality_score:.3f} < {self.cfg["quality_threshold"]})'}

        # 2. Stop Loss (Phase 4)
        # atr is assumed to be in the same scale as entry_price (scaled by 100,000 in this system)
        sl_distance_scaled = self.cfg['k_sl'] * atr

        # 3. Dynamic Take Profit (Phase 5)
        if quality_score > 0.8:
            rr = 3.0
        elif quality_score > 0.6:
            rr = 2.0
        else:
            rr = 1.5

        tp_distance_scaled = rr * sl_distance_scaled

        # 4. Probabilistic Position Sizing (Phase 6)
        base_risk = self.cfg['base_risk_percent'] * balance

        # Edge Factor calculation: Confidence * Strength * Conviction
        edge_strength = meta_prob * quality_score * abs(direction_score)
        allocated_risk = base_risk * edge_strength

        # Cap risk to safety limit
        max_risk = self.cfg['max_risk_cap'] * balance
        allocated_risk = min(allocated_risk, max_risk)

        # position_size = allocated_risk / sl_distance (in real price units)
        # sl_distance_real = (sl_distance_scaled * 100,000) / 10^digits
        sl_distance_real = (sl_distance_scaled * 100000) / (10**digits)

        if sl_distance_real <= 0:
            return {'action': 0, 'reason': 'Zero SL Distance'}

        # RR check against minimum requirement
        if rr < self.cfg['min_required_rr']:
            return {'action': 0, 'reason': f'RR Filter ({rr:.2f} < {self.cfg["min_required_rr"]})'}

        # Final parameters
        direction = 1 if direction_score > 0 else -1

        # Convert to lots
        contract_size = 100 if asset_name == 'XAUUSD' else 100000
        is_usd_quote = asset_name in ['EURUSD', 'GBPUSD', 'XAUUSD']

        # For non-USD quote pairs, we must convert risk to quote currency first
        # real_price = (entry_price * 100,000) / 10^digits
        real_price = (entry_price * 100000) / (10**digits)
        risk_in_quote = allocated_risk * (1.0 if is_usd_quote else real_price)
        position_units = risk_in_quote / sl_distance_real

        lots = position_units / contract_size
        lots = np.clip(lots, 0.01, 50.0) # Reasonable safety cap

        # API Point calculations
        relative_sl = int(round(sl_distance_scaled * 100000 / (10**(5 - digits))))
        relative_tp = int(round(tp_distance_scaled * 100000 / (10**(5 - digits))))

        sl_price = round(real_price - (direction * relative_sl / (10**digits)), digits)
        tp_price = round(real_price + (direction * relative_tp / (10**digits)), digits)

        return {
            'action': 1 if direction == 1 else 2, # 1: BUY, 2: SELL
            'lots': float(lots),
            'sl': float(sl_price),
            'tp': float(tp_price),
            'relative_sl': relative_sl,
            'relative_tp': relative_tp,
            'rr': rr,
            'edge_strength': edge_strength,
            'allocated_risk': allocated_risk,
            'direction_score': direction_score,
            'quality_score': quality_score,
            'meta_prob': meta_prob
        }
