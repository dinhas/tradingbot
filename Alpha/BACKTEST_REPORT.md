# Alpha Model: Regime Expansion and Barrier Analysis

## 1. Regime Identification (Denoising Research)
Following the research findings in `denoising_research/`, I have expanded the model's operational scope to include **Trending** environments.
*   **Trending Regime Definition:** ADX > 25 AND ATR_norm < ATR_q75 (75th percentile of normalized ATR).
*   **Ranging Regime Definition:** ADX < 20 AND Hurst < 0.48.

## 2. Multi-Regime Comparison
| Metric | Ranging Only (4x/2x Barriers) | Ranging + Trending (4x/2x Barriers) |
| :--- | :--- | :--- |
| **Profit Factor** | 1.0342 | 0.9992 |
| **Total Return** | +13.87% | -3.01% |
| **Sharpe Ratio** | 0.6376 | 0.3936 |
| **Max Drawdown** | -24.35% | -52.57% |
| **Win Rate** | 34.24% | 33.38% |
| **Total Trades** | 812 | 5563 |
| **Avg Hold Time** | 217.1 min | 243.3 min |

## 3. Deep Analysis: The "Trending" Challenge
### Predictive Collapse (Bias)
The expanded training set (93,135 samples) was still heavily dominated by the Class -1 (Short) label due to the 4x TP / 2x SL barrier asymmetry. The price hit the tighter stop loss significantly more often than the target within the 100-minute observation window (20 bars). Consequently, the model learned to predict Class -1 (Short) 100% of the time.

### Why did performance drop with Trending regimes?
1.  **Market Drift:** In the 2024 test period, **Trending** regimes were predominantly **Bullish** (Long). An "Always Short" model that was profitable during Ranging oscillations became a liability during strong Trending moves.
2.  **Trade Frequency:** Trade frequency jumped from 2.3 trades/day to 16 trades/day. The model was constantly opening short positions in trending environments, leading to a "death by a thousand stops" scenario.
3.  **Barrier Suitability:** 4x ATR targets are extremely difficult to reach in a ranging market, but potentially reachable in a trend. However, a 2x ATR stop is very easily triggered by a minor pullback in a strong trend, causing the "short-only" model to fail even more reliably.

## 4. Final Findings and V4 Strategy
The Alpha model has now been successfully modularized and updated to handle multi-regime logic. However, the current "Selective Edge Extraction" is hampered by **Class Imbalance**.

**Next Steps for Alpha V4:**
- **Symmetric Labeling for Training:** Train the model on symmetric 2x/2x barriers to learn pure directional signal, but use asymmetric 4x/2x barriers in the backtest/live execution for risk management.
- **Regime-Specific Class Weighting:** Implement a loss function that penalizes errors more heavily in the underrepresented class during training.
- **Trend-Following Logic:** For Trending regimes, ensure the model is trained with a larger time window (>20 bars) to allow the trend to develop toward the 4x target.
