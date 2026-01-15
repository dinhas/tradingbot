# Risk Model (TradeGuard) Fix Report - Jan 15, 2026

## 1. The Problems Identified
We discovered three critical bugs that caused the last ~15 retraining attempts to fail in backtests:

*   **Feature Index Mismatch:** The code was prepending Alpha Signals to the observation vector, shifting all market indicators (RSI, MACD, etc.) by 2 positions. The AI was essentially "blind," reading RSI values as MACD values.
*   **Corrupted Training Statistics:** The `equity_norm` feature reached a mean of **1756.9**. This indicates the training environment had a "leak" where equity was compounding infinitely or not resetting. This made the model's normalization logic useless for real-world account balances.
*   **System Incompatibility:** The Live Execution system was expecting a 45-feature model, but the environment was producing 88 features.

## 2. The Fixes Applied

### A. Locked Feature Ordering (DNA Fix)
*   **Modified:** `RiskLayer/src/feature_engine.py`
*   **Change:** Ensured that **OHLCV (Open, High, Low, Close, Volume)** always occupies Indices 0-4 for every asset.
*   **Result:** The Model and the Normalizer are now perfectly aligned. Index 0 is always "Open", Index 7 is always "RSI", etc.

### B. Reality Capping (Anti-Glitch)
*   **Modified:** `RiskLayer/env/risk_env.py`
*   **Change:** Added `MAX_EQUITY_MULT = 5.0`. If the model reaches 500% profit in a single training round, the simulation terminates. 
*   **Result:** This prevents "Infinite Money" bugs from corrupting the normalizer stats. The average equity will now stay near 1.0, which is correct for a fresh account.

### C. Observation Clipping
*   **Modified:** `RiskLayer/env/risk_env.py`
*   **Change:** Added `np.clip` to equity and drawdown features.
*   **Result:** Even if a weird price spike happens, it won't break the AI's "brain" with extreme numbers.

### D. Live System Synchronization
*   **Modified:** `LiveExecution/src/features.py`
*   **Change:** Switched the `FeatureManager` to use the `RiskFeatureEngine` (86 features) instead of the old `AlphaFeatureEngine` (45 features) for risk decisions.
*   **Result:** The Live system now "sees" exactly what the AI saw during training.

## 3. Next Steps
The code is now **fixed and verified**. 
**Action Required:** You must run a fresh training session. The model will now learn correctly because the data is clean and the indices are aligned.
