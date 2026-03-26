import numpy as np
from numba import njit

@njit
def resolve_trade_fast(opens, highs, lows, closes, entry_mid, sl_price, tp_price, direction, spread):
    half_spread = spread / 2.0
    if direction == 1: actual_entry = entry_mid + half_spread
    else: actual_entry = entry_mid - half_spread

    sl_dist = abs(actual_entry - sl_price)
    if sl_dist < 1e-10: sl_dist = 1e-10

    for i in range(len(opens)):
        o, h, l = opens[i], highs[i], lows[i]
        if direction == 1:
            bid_o, bid_h, bid_l = o - half_spread, h - half_spread, l - half_spread
            if bid_o <= sl_price: return 0, (bid_o - actual_entry) / sl_dist, i + 1
            if bid_o >= tp_price: return 1, (bid_o - actual_entry) / sl_dist, i + 1
            if bid_l <= sl_price: return 0, (sl_price - actual_entry) / sl_dist, i + 1
            if bid_h >= tp_price: return 1, (tp_price - actual_entry) / sl_dist, i + 1
        else:
            ask_o, ask_h, ask_l = o + half_spread, h + half_spread, l + half_spread
            if ask_o >= sl_price: return 0, (actual_entry - ask_o) / sl_dist, i + 1
            if ask_o <= tp_price: return 1, (actual_entry - ask_o) / sl_dist, i + 1
            if ask_h >= sl_price: return 0, (actual_entry - sl_price) / sl_dist, i + 1
            if ask_l <= tp_price: return 1, (actual_entry - tp_price) / sl_dist, i + 1
    return 2, 0.0, len(opens)

# Test Long
entry_mid = 1.1000
spread = 0.0002
sl = 1.0990
tp = 1.1020
# Correct Ask Entry: 1.1001. SL Dist: 0.0011.
# Candle Low Mid 1.0991 -> Bid Low 1.0990. Should hit SL.
res_code, res_r, res_len = resolve_trade_fast(
    np.array([1.1000], dtype=np.float32),
    np.array([1.1010], dtype=np.float32),
    np.array([1.0991], dtype=np.float32),
    np.array([1.1000], dtype=np.float32),
    entry_mid, sl, tp, 1, spread
)
print(f"Long SL Hit: {res_code == 0}, R: {res_r}")

# Test Short
# Correct Bid Entry: 1.0999. SL 1.1010. SL Dist: 0.0011.
# Candle High Mid 1.1009 -> Ask High 1.1010. Should hit SL.
res_code, res_r, res_len = resolve_trade_fast(
    np.array([1.1000], dtype=np.float32),
    np.array([1.1009], dtype=np.float32),
    np.array([1.0990], dtype=np.float32),
    np.array([1.1000], dtype=np.float32),
    entry_mid, 1.1010, 1.0980, -1, spread
)
print(f"Short SL Hit: {res_code == 0}, R: {res_r}")
