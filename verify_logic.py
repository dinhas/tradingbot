import numpy as np

# Mocking the slippage logic from backtest_combined.py
def test_slippage_logic():
    price_raw = 1.1000
    direction = 1 # Long
    slippage = 0.0001
    price = price_raw + (direction * -1 * slippage)
    print(f"Long Entry: Raw={price_raw}, Price={price}, Diff={price-price_raw}")

    direction = -1 # Short
    price = price_raw + (direction * -1 * slippage)
    print(f"Short Entry: Raw={price_raw}, Price={price}, Diff={price-price_raw}")

# Mocking resolve_trade_fast from risk_ppo_env.py
def test_resolve_trade_fast_logic():
    # Long trade
    entry_mid = 1.1000
    spread = 0.0002
    sl_mid = 1.0990
    direction = 1

    actual_entry = entry_mid + spread
    print(f"RiskEnv Long Entry: {actual_entry} (Mid + Spread)")

    # Check if SL hit
    low_mid = 1.0991
    sl_hit = low_mid <= sl_mid
    print(f"RiskEnv SL Hit (Low Mid={low_mid}, SL Mid={sl_mid}): {sl_hit}")
    # Realistically, Bid = low_mid - spread/2 = 1.0991 - 0.0001 = 1.0990.
    # SL should be hit! But code says False.

test_slippage_logic()
test_resolve_trade_fast_logic()
