# Alpha Model: Barrier Sensitivity Analysis and Backtest Report

## 1. Parameter Comparison
| Metric | Baseline (TP: 1.5, SL: 1.5) | Optimized (TP: 4.0, SL: 2.0) |
| :--- | :--- | :--- |
| **Profit Factor** | 0.9874 | 1.0342 |
| **Total Return** | -5.39% | +13.87% |
| **Sharpe Ratio** | -0.0821 | 0.6376 |
| **Max Drawdown** | -40.73% | -24.35% |
| **Win Rate** | 38.64% | 34.24% |
| **Total Trades** | 885 | 812 |
| **Avg Hold Time** | 174.7 min | 217.1 min |

## 2. Training Analysis
*   **Dataset Generation:** The 4.0/2.0 barrier configuration significantly reduced the training set size (7,173 -> 3,027) because fewer price movements reached the 4x ATR target within the 20-bar window in a RANGING regime.
*   **Class Imbalance:**
    *   Baseline: 100% Long Bias (Class 1)
    *   Optimized: 100% Short Bias (Class -1)
*   **Convergence:** Both models achieved near-zero loss, indicating they successfully "learned" to predict the majority class perfectly, rather than capturing subtle alpha signals.

## 3. Deep Analysis: Why did the Optimized Model perform better?
Despite the predictive collapse (Short Bias), the **Optimized (4.0/2.0)** model outperformed the baseline primarily due to the asymmetry of the barriers:
1.  **Risk/Reward Ratio:** By setting TP to 2x the SL distance, even a low win rate (34%) generated a positive expected value.
2.  **Regime Characteristics:** In the 2024 ranging markets, a persistent short-side drift (or specific asset behavior) allowed a "Always Short" strategy to be profitable when gated by the ADX/Hurst filters.
3.  **Noise Tolerance:** Larger barriers are less susceptible to 5-minute market noise "whipsawing" the position before the logic can play out.

## 4. Recommendations for V4
- **Synthetic Balancing:** Implement SMOTE or Class Weighting during training to force the model to learn both Buy and Sell signals.
- **Dynamic Exit Logic:** Instead of fixed barriers, use the Kalman-smoothed ATR to adjust barriers per-bar as volatility evolves.
- **Contrarian Labeling:** In RANGING regimes, explore labeling based on mean-reversion (e.g., selling at upper BB and buying at lower BB) to capture the oscillation edge.
