NORMALIZATION_WINDOW = 50
MIN_HISTORY_CANDLES = 200
MAX_HISTORY_BUFFER = 300

ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "XAUUSD"]

FEATURE_LOOKBACKS = {
    "returns": 12,
    "atr": 14,
    "atr_ma": 20,
    "bollinger": 20,
    "ema_fast": 9,
    "ema_slow": 21,
    "rsi": 14,
    "volume_ma": 20,
    "swing_structure": 40,
    "correlation": 50,
}
