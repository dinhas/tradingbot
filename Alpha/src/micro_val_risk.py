import numpy as np
import pandas as pd
import logging
import sys
import os

# Add paths
sys.path.append(os.path.abspath("."))

# Mocking parts of the system for simulation
class MockModelLoader:
    def get_alpha_action(self, obs):
        # Return random but plausible outputs
        return {
            'direction': np.array([np.random.uniform(-1, 1)]),
            'quality': np.array([np.random.uniform(0, 1)]),
            'meta': np.array([np.random.uniform(0, 1)])
        }

class MockFeatureManager:
    def __init__(self):
        self.assets = ['EURUSD']
        self.history = {
            'EURUSD': pd.DataFrame(
                index=pd.date_range("2023-01-01", periods=1000, freq="5min"),
                data={'close': np.random.uniform(1.1000, 1.1100, 1000)}
            )
        }
    def get_alpha_observation(self, asset, portfolio):
        return np.random.randn(40)
    def get_atr(self, asset):
        return 0.0010

class MockOrchestrator:
    def __init__(self):
        self.fm = MockFeatureManager()
        self.ml = MockModelLoader()
        self.portfolio_state = {'balance': 10000.0}
        self.symbol_digits = {'EURUSD': 5}
        self.logger = logging.getLogger("Simulation")
        self.live_trade_count = 0
        self.risk_cfg = {
            'meta_threshold': 0.70,
            'direction_threshold': 0.10,
            'quality_threshold': 0.30,
            'k_sl': 1.5,
            'base_risk_percent': 0.01,
            'max_risk_cap': 0.05,
            'min_required_rr': 1.0,
            'max_leverage': 100.0
        }

    def _get_symbol_name(self, sid): return "EURUSD"

    # Copy-pasted logic from Orchestrator (simplified for sim)
    def simulate_inference(self):
        asset_name = "EURUSD"
        latest_bar = self.fm.history[asset_name].iloc[-1]
        alpha_obs = self.fm.get_alpha_observation(asset_name, self.portfolio_state)
        alpha_out = self.ml.get_alpha_action(alpha_obs)

        direction_score = float(alpha_out['direction'][0])
        quality_score = float(alpha_out['quality'][0])
        meta_prob = float(alpha_out['meta'][0])

        # Filtering
        if meta_prob < self.risk_cfg['meta_threshold'] or \
           abs(direction_score) < self.risk_cfg['direction_threshold'] or \
           quality_score < self.risk_cfg['quality_threshold']:
            return None, "Filtered"

        # Risk Calculation
        digits = 5
        scaled_price = latest_bar['close']
        real_price = round(scaled_price * 100000 / (10**digits), digits)
        atr_scaled = self.fm.get_atr(asset_name)
        sl_distance_scaled = self.risk_cfg['k_sl'] * atr_scaled

        if quality_score > 0.8: rr = 3.0
        elif quality_score > 0.6: rr = 2.0
        else: rr = 1.5

        edge_strength = meta_prob * quality_score * abs(direction_score)
        allocated_risk = self.risk_cfg['base_risk_percent'] * self.portfolio_state['balance'] * edge_strength
        allocated_risk = min(allocated_risk, self.risk_cfg['max_risk_cap'] * self.portfolio_state['balance'])

        sl_distance_real = (sl_distance_scaled * 100000) / (10**digits)

        # Convert to units, then to lots
        contract_size = 100000
        is_usd_quote = True # Mocking EURUSD

        risk_in_quote = allocated_risk * (1.0 if is_usd_quote else real_price)
        position_units = risk_in_quote / sl_distance_real

        lots = position_units / contract_size
        lots = np.clip(lots, 0.01, 50.0)

        return {
            'rr': rr,
            'edge_strength': edge_strength,
            'lots': lots,
            'allocated_risk': allocated_risk
        }, "Accepted"

def run_simulation(n_trials=200):
    sim = MockOrchestrator()
    results = []
    rejections = 0

    for _ in range(n_trials):
        res, status = sim.simulate_inference()
        if res:
            results.append(res)
        else:
            rejections += 1

    # Calculate metrics
    avg_rr = np.mean([r['rr'] for r in results]) if results else 0
    rejection_rate = rejections / n_trials
    mean_lots = np.mean([r['lots'] for r in results]) if results else 0
    edge_strengths = [r['edge_strength'] for r in results]

    print(f"--- Simulation Results ({n_trials} trials) ---")
    print(f"Trade Rejection Rate: {rejection_rate:.1%}")
    print(f"Average RR: {avg_rr:.2f}")
    print(f"Mean Position Size (Lots): {mean_lots:.2f}")
    if edge_strengths:
        print(f"Edge Strength: Mean={np.mean(edge_strengths):.4f}, Max={np.max(edge_strengths):.4f}, Min={np.min(edge_strengths):.4f}")
    print("------------------------------------------")

if __name__ == "__main__":
    run_simulation(200)
