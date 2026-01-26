import unittest
import numpy as np
import pandas as pd
import os
import sys

# Add project root to path to allow direct import of modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from RiskLayer.src.risk_env import RiskManagementEnv
from Shared.execution import ExecutionEngine, TradeConfig

class TestRiskEnvironment(unittest.TestCase):
    """Comprehensive test suite for the RiskManagementEnv."""

    @classmethod
    def setUpClass(cls):
        """Create a mock dataset for all tests to use."""
        cls.dataset_path = 'tests/risk_layer/mock_risk_dataset.parquet'
        cls.cache_dir = 'tests/risk_layer/mock_risk_dataset_cache'

        # Clean up old cache if it exists
        if os.path.exists(cls.cache_dir):
            import shutil
            shutil.rmtree(cls.cache_dir)

        # Create a diverse mock dataset
        data = {
            'features': [np.random.rand(40).tolist() for _ in range(105)],
            'entry_price': np.linspace(1.1000, 1.1200, 105),
            'atr': np.full(105, 0.0010),
            'direction': ([1, -1] * 53)[:105], # Alternating long/short
            'max_profit_pct': np.full(105, 0.01), # 1% potential profit
            'max_loss_pct': np.full(105, -0.01), # 1% potential loss
            'close_1000_price': np.linspace(1.1050, 1.1250, 105),
            'pair': ['EURUSD'] * 105,
        }
        mock_df = pd.DataFrame(data)
        mock_df.to_parquet(cls.dataset_path)

    @classmethod
    def tearDownClass(cls):
        """Clean up the mock dataset and cache after all tests."""
        if os.path.exists(cls.dataset_path):
            os.remove(cls.dataset_path)
        if os.path.exists(cls.cache_dir):
            import shutil
            shutil.rmtree(cls.cache_dir)

    def setUp(self):
        """Set up a fresh environment instance for each test."""
        self.env = RiskManagementEnv(dataset_path=self.dataset_path, initial_equity=10000.0)
        self.engine = self.env.engine

    # ----------------------------------------
    # A. Transaction Cost Tests
    # ----------------------------------------

    def test_bug_01_commission_fee_is_zero_in_code(self):
        """
        BUG-01: Validates that the commission fee is 0, contradicting documentation.
        """
        config_fee = self.engine.config.TRADING_COST_PCT
        self.assertEqual(config_fee, 0.0, "BUG-01: TRADING_COST_PCT in TradeConfig should be 0.0 as per code, but test failed.")

        self.env.reset()
        action = np.array([0.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = self.env.step(action)

        documented_fee_pct = 0.0002
        entry_price = self.env.entry_prices[0]
        lots = info['lots']
        contract_size = self.env.contract_sizes[0]
        position_value = lots * contract_size * entry_price
        round_trip_fee = position_value * documented_fee_pct

        self.assertNotIn('fee', info, "BUG-01: A 'fee' key should not be in the info dict if fees are not calculated.")
        print(f"\\nBUG-01 INFO: PnL from step: {info['pnl']:.2f}, Calculated fee would be: {round_trip_fee:.2f}. The PnL does not account for this fee.")

    def test_bug_02_spread_formula_is_unconventional(self):
        """
        BUG-02: Documents that the spread calculation formula is unconventional, but the resulting value is realistic.
        """
        mid_price = 1.10000
        atr = 0.00100
        calculated_spread = self.engine.get_spread(mid_price=mid_price, atr=atr)
        calculated_pips = calculated_spread / 0.0001

        min_realistic_pips = 0.5
        max_realistic_pips = 3.0

        self.assertTrue(min_realistic_pips <= calculated_pips <= max_realistic_pips,
                        f"BUG-02 Evidence: Calculated spread is {calculated_pips:.2f} pips, which is within the realistic range of [{min_realistic_pips}, {max_realistic_pips}].")
        print(f"\\nBUG-02 INFO: Calculated spread is {calculated_pips:.2f} pips. This is a realistic value, but the formula component `* mid_price` is unconventional.")

    def test_slippage_is_disabled_by_default(self):
        """
        CONFIRMATION: Validates that slippage is disabled in the environment.
        """
        self.assertFalse(self.env.ENABLE_SLIPPAGE)
        mid_price = 1.1000
        direction = 1
        atr = 0.0010
        entry_price_with_slippage_disabled = self.engine.get_entry_price(mid_price, direction, atr, enable_slippage=False)
        spread = self.engine.get_spread(mid_price, atr)
        expected_price = mid_price + spread
        self.assertAlmostEqual(entry_price_with_slippage_disabled, expected_price, 5)

    # ----------------------------------------
    # B. Environment Mechanics Tests
    # ----------------------------------------

    def test_observation_space_dimensions(self):
        """CONFIRMATION: Checks if the observation space has the correct shape of 65."""
        obs, _ = self.env.reset()
        self.assertEqual(obs.shape, (65,))
        action = self.env.action_space.sample()
        obs, _, _, _, _ = self.env.step(action)
        self.assertEqual(obs.shape, (65,))

    def test_bug_03_action_scaling_inconsistency(self):
        """
        BUG-03: Validates that action scaling in `risk_env.py` is inconsistent with `Shared/execution.py`.
        """
        # Test with a neutral action
        action = np.array([0.0, 0.0, 0.0])

        # Get the SL multiplier from the environment's internal logic
        sl_mult_env = (action[0] + 1) / 2 * (3.0 - 0.5) + 0.5
        tp_mult_env = (action[1] + 1) / 2 * (10.0 - 1.0) + 1.0

        # Get the SL multiplier from the shared engine's logic
        sl_mult_engine, tp_mult_engine = self.engine.decode_action(action)

        # Assert that the two implementations are not the same
        self.assertNotAlmostEqual(sl_mult_env, sl_mult_engine,
                                  msg=(f"BUG-03: `risk_env` SL multiplier ({sl_mult_env}) differs from "
                                       f"`Shared/execution.py` SL multiplier ({sl_mult_engine})."))

        self.assertNotAlmostEqual(tp_mult_env, tp_mult_engine,
                                  msg=(f"BUG-03: `risk_env` TP multiplier ({tp_mult_env}) differs from "
                                       f"`Shared/execution.py` TP multiplier ({tp_mult_engine})."))

        print(f"\\nBUG-03 INFO: `risk_env` SL multiplier: {sl_mult_env:.2f}, "
              f"`Shared/execution.py` SL multiplier: {sl_mult_engine:.2f}. The logic has diverged.")

    def test_reset_functionality(self):
        """CONFIRMATION: Ensures reset() properly reinitializes the environment state."""
        self.env.reset()
        for _ in range(5): self.env.step(self.env.action_space.sample())
        self.env.reset()
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.equity, self.env.initial_equity_base)

    def test_episode_truncation(self):
        """CONFIRMATION: Tests if the episode is truncated after EPISODE_LENGTH steps."""
        self.env.reset()
        for i in range(self.env.EPISODE_LENGTH):
            _, _, _, truncated, _ = self.env.step(self.env.action_space.sample())
        self.assertTrue(truncated)

    def test_bug_04_episode_termination_on_drawdown(self):
        """
        BUG-04: Validates that the drawdown termination reward logic is complex.
        """
        self.env.reset()
        self.env.equity = self.env.initial_equity_base * 0.2
        _, reward, terminated, _, info = self.env.step(self.env.action_space.sample())
        self.assertTrue(terminated)
        expected_reward = np.clip(info['efficiency'] + info['bullet'] - 20.0, -100.0, 100.0)
        self.assertEqual(reward, expected_reward)
        print(f"\\nBUG-04 INFO: Termination reward was {reward:.2f}. The -20 penalty is combined with other rewards, not an absolute value.")

    # ----------------------------------------
    # C. Financial Logic Tests
    # ----------------------------------------

    def test_portfolio_value_updates_correctly(self):
        """CONFIRMATION: Verifies that equity is correctly updated with PnL after a trade."""
        self.env.reset()
        initial_equity = self.env.equity
        _, _, _, _, info = self.env.step(self.env.action_space.sample())
        pnl = info['pnl']
        final_equity = self.env.equity
        self.assertAlmostEqual(final_equity, initial_equity + pnl, 2)

    def test_bug_05_position_sizing_logic_divergence(self):
        """
        BUG-05: Validates that position sizing in `risk_env.py` is inconsistent with `Shared/execution.py`.
        """
        self.env.reset()
        action = np.array([0.0, 0.0, 1.0]) # Max risk action -> 10%

        # Logic from risk_env.py
        risk_pct_env = 0.10
        risk_amount_cash_env = self.env.equity * risk_pct_env

        # Logic from Shared/execution.py (adaptive risk)
        initial_equity = self.env.initial_equity_base
        current_equity = self.env.equity
        risk_multiplier = initial_equity / max(current_equity, initial_equity * 0.3)
        effective_risk_pct_engine = min(self.engine.config.DEFAULT_RISK_PCT * risk_multiplier, 0.50)
        risk_amount_cash_engine = current_equity * effective_risk_pct_engine

        self.assertNotAlmostEqual(risk_amount_cash_env, risk_amount_cash_engine, 2)
        print(f"\\nBUG-05 INFO: `risk_env` calculates position size based on {risk_pct_env*100:.1f}% risk from action. `Shared/execution.py` would have used {effective_risk_pct_engine*100:.1f}% adaptive risk.")

    def test_pnl_calculation_correctness(self):
        """CONFIRMATION: Verifies PnL calculation against a manual calculation for an SL exit."""
        self.env.reset()

        # This action will result in an SL hit with the mock data
        action = np.array([0.0, 1.0, 0.0])

        obs, reward, term, trunc, info = self.env.step(action)
        self.assertEqual(info['exit'], 'SL', "Test setup assumption failed: trade should have exited on SL.")

        # Manual PnL Calculation
        idx = self.env.episode_start_idx + self.env.current_step - 1

        mid_price_entry = self.env.entry_prices[idx]
        atr = self.env.atrs[idx]
        direction = self.env.directions[idx]

        # Recreate the exact entry price
        entry_price_exec = self.engine.get_entry_price(mid_price_entry, direction, atr, enable_slippage=False)

        # Recreate the exact SL price used for the exit
        # sl_mult is 1.75 for action[0] = 0.0
        sl_mult = (action[0] + 1) / 2 * (3.0 - 0.5) + 0.5
        sl_dist_price = sl_mult * atr
        exit_price_exec = entry_price_exec - (direction * sl_dist_price)

        # Use the engine's own PnL function for an apples-to-apples comparison
        manual_pnl = self.engine.calculate_pnl(
            entry_price=entry_price_exec,
            exit_price=exit_price_exec,
            lots=info['lots'],
            direction=direction,
            is_usd_quote=True,
            contract_size=self.engine.config.CONTRACT_SIZE
        )

        self.assertAlmostEqual(info['pnl'], manual_pnl, 2, "PnL calculation for SL exit should match manual calculation.")

    # ----------------------------------------
    # D. Data Integrity Tests
    # ----------------------------------------

    def test_data_loading_and_caching(self):
        """CONFIRMATION: Ensures data is loaded and the caching mechanism works."""
        self.assertTrue(os.path.exists(self.cache_dir))
        self.assertGreater(len(self.env.entry_prices), 0)

        # Re-initializing should be faster as it uses the cache
        import time
        start_time = time.time()
        RiskManagementEnv(dataset_path=self.dataset_path)
        end_time = time.time()
        self.assertLess(end_time - start_time, 0.1) # Should be very fast

    # ----------------------------------------
    # E. Edge Case Tests
    # ----------------------------------------

    def test_bug_06_zero_atr_scenario(self):
        """BUG-06: Test if environment handles zero ATR gracefully to avoid division by zero."""
        # Override ATR value in the dataset array for the current step
        self.env.reset()
        current_idx = self.env.episode_start_idx + self.env.current_step

        # Create a writable copy of the memory-mapped array for the test
        writable_atrs = np.copy(self.env.atrs)
        writable_atrs[current_idx] = 0.0
        self.env.atrs = writable_atrs

        try:
            _, _, _, _, info = self.env.step(self.env.action_space.sample())
            self.assertGreater(info['lots'], 0, "Lots should be greater than zero.")

        except ZeroDivisionError:
            self.fail("BUG-06: A ZeroDivisionError was raised when ATR is 0.0.")
        print(f"\\nBUG-06 INFO: Zero ATR was handled without crashing. Lots: {info['lots']}")


if __name__ == '__main__':
    unittest.main()
