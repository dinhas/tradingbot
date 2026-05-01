# Alpha Model: Training and Backtest Report

## 1. Training Report
*   **Total Training Time:** 2 minutes 17 seconds (20 epochs)
*   **Device:** CPU (optimized with 4 workers)
*   **Training Loss performance:**
    *   **Initial Epoch:** ~0.11
    *   **Final Epoch (20):** Train Loss: 0.0632 | Val Loss: 0.0638
*   **Convergence:** The model showed steady convergence over 20 epochs, though the loss plateaued around epoch 15.

## 2. Market Regime Analysis
*   **Regime used for training:** **RANGING**
*   **Regime Definition:** ADX < 20 AND Hurst Exponent < 0.48.
*   **Logic:** The Alpha model is specifically designed to extract edge from mean-reverting environments. Trending and Breakout periods are filtered out during training and backtesting.

## 3. Backtest Metrics (OOS 2024)
*   **Test Period:** 2023-12-31 to 2024-12-13
*   **Confidence Threshold:** 0.50
*   **Profit Factor:** 0.9874
*   **Total Return:** -5.39%
*   **Sharpe Ratio:** -0.0821
*   **Max Drawdown:** -40.73%
*   **Win Rate:** 38.64%
*   **Total Trades:** 885
*   **Avg Hold Time:** 174.7 minutes
*   **Exit Summary:**
    *   **Stop Loss (SL):** 517
    *   **Take Profit (TP):** 232
    *   **Signal Flip:** 134

## 4. Deep Analysis
### Model Bias Observation
Evaluation of the model on the test set revealed a significant **Long Bias** (100% Class 1 predictions). This is a common failure mode in LSTM training on noisy financial data where the model finds a local minimum by predicting the majority class or the mean direction of the training set.

### Backtest Performance
Despite the bias, the backtester executed 885 trades because the **RANGING** regime filter and the **0.50 threshold** were active. The poor profit factor (0.98) and high SL hit rate suggest that:
1.  **Label Quality:** The Triple Barrier labels in ranging markets may be too tight, causing the model to hit SL before reaching the TP barrier.
2.  **Feature Signal:** The 5 Kalman-smoothed features might be over-smoothed, losing the micro-patterns necessary for 5-minute prediction.
3.  **Regime Stability:** Even within the RANGING regime, 2024 saw significant "false ranges" that eventually broke into trends, hitting the SL.

## 5. Recommendations
- **Increase Epochs:** The 20 epochs used might be insufficient for the LSTM to learn complex temporal dependencies.
- **Rebalance Training Set:** Although undersampling was used, the model still collapsed to a single class. Advanced synthetic oversampling (SMOTE) or tighter label filtering might be required.
- **Dynamic Thresholding:** Adjusting the 0.50 threshold per asset based on rolling performance.
