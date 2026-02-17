import numpy as np
import pandas as pd
import logging
import sys
import os

# Add paths
sys.path.append(os.path.abspath("."))

from RiskLayer.src.risk_engine import RiskEngine

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
        self.risk_engine = RiskEngine()

    def _get_symbol_name(self, sid): return "EURUSD"

    def simulate_inference(self):
        asset_name = "EURUSD"
        latest_bar = self.fm.history[asset_name].iloc[-1]
        alpha_obs = self.fm.get_alpha_observation(asset_name, self.portfolio_state)
        alpha_out = self.ml.get_alpha_action(alpha_obs)

        direction_score = float(alpha_out['direction'][0])
        quality_score = float(alpha_out['quality'][0])
        meta_prob = float(alpha_out['meta'][0])

        digits = 5
        atr = self.fm.get_atr(asset_name)
        balance = self.portfolio_state['balance']

        decision = self.risk_engine.get_trade_decision(
            asset_name,
            latest_bar['close'],
            atr,
            direction_score,
            quality_score,
            meta_prob,
            balance,
            digits
        )

        if not decision or decision['action'] == 0:
            return None, "Rejected"

        return decision, "Accepted"

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
