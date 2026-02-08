# Alpha Model V2: Analysis & Proposal

## 1. Feature Analysis
The feature set (`FeatureEngine`) is robust and well-normalized.
- **Inputs:** 40 features (25 per-asset, 15 global).
- **Strengths:** Includes regime detection (`risk_on`, `volatility`) and cross-asset correlations (`corr_basket`), which are crucial for a multi-asset bot.
- **Scaling:** Robust scaling (Median/IQR) is excellent for financial data outliers.

## 2. Reward System Analysis
The "Peek & Label" reward (`TradingEnv`) is a high-variance, high-quality signal.
- **Mechanism:** It looks ahead 1000 steps to determine if SL or TP is hit.
- **Implication:** The reward is "immediate" in time-step assignment (you get it when you enter), but "future" in reality.
- **Optimization:** This turns the RL problem partly into a contextual bandit problem (State -> Prediction of Trade Outcome).
- **Gamma:** A high gamma (0.99) is less necessary because the "future reward" is already compressed into the current step. A slightly lower gamma (0.95) helps the agent focus on this immediate high-quality signal while still respecting the "survival" aspect (drawdown penalty).

## 3. Architecture Proposal
Given the complexity of mapping 40 features to a precise "Buy/Sell" decision that predicts 1000 steps ahead, the previous network (`[256, 256, 128]`) might be slightly underpowered or prone to underfitting the complex "Regime -> Outcome" mapping.

**New Architecture:** `[512, 256, 256]`
- **Why?** The first layer needs to expand the 40 features into a high-dimensional space to disentangle the "Market Regime" from the "Asset Signal".
- **Depth:** 3 layers allow for hierarchical feature extraction.

## 4. Hyperparameter Tuning
- **Learning Rate:** Reduced to `1e-4` to handle the larger network and prevent "catastrophic forgetting" of rare but important regime patterns.
- **Batch Size/Steps:** Increased to `4096/512` to provide smoother gradient estimates, reducing the noise from the stochastic nature of the market data.
- **Entropy:** Lowered to `0.01` because the "Peek & Label" signal is strong. We want the agent to converge on the "correct" prediction once learned, rather than staying too random.

## 5. Risk Management
- **Persistence:** The training loop uses `enable_persistence=True`. This mimics a continuous deployment.
- **Recommendation:** Ensure the dataset covers different market conditions (Bull, Bear, Range). The current usage of 5 environments on 5 different assets provides good diversity.
