# TradeGuard: Meta-Labeling Filter Layer

TradeGuard is a secondary "Quality Assurance" machine learning layer positioned at the end of the trading pipeline. It analyzes the decisions made by the Alpha model (Direction) and the Risk model (Sizing/SL/TP) to predict the probability of a trade being profitable.

## 🧠 Core Concept

The Alpha model identifies opportunities, and the Risk model structures the trade. However, certain market regimes or volatile conditions may lead to high-risk setups that both models "agree" on but are historically losers. TradeGuard learns these patterns and acts as a final gatekeeper.

## 📁 Structure

- `generate_dataset.py`: The "Data Factory". Runs Alpha + Risk models through 2016-2024 data to record performance.
- `train_guard.py`: The "Trainer". Uses LightGBM to build a classifier based on harvested data.
- `guard_model.py`: The "Inference Engine". A class used by the live system to get real-time win probabilities.
- `data/`: Directory for buffered Parquet chunks (Memory Optimized).
- `models/`: Directory for saved LightGBM models.

## 🛡️ Portfolio & Management Rules

The dataset generator enforces specific management rules (applied universally to all assets) to ensure realistic training:

1.  **No Pyramiding (Universal):** If a position is already open for an asset (e.g., Buy EURUSD), any new signal in the **same direction** (Buy EURUSD) is **ignored**. We only hold one trade per asset at a time.
2.  **Reversal Logic (Universal):** If a position is open (e.g., Buy EURUSD) and an **opposite signal** arrives (Sell EURUSD), the current position is closed, and the new position is opened immediately.
3.  **Memory Management:** Data is streamed in 5,000-row chunks to Parquet files to prevent RAM exhaustion during multi-year backtests.

## 🚀 Execution Plan

### 1. Data Generation
Run the following to simulate the Alpha+Risk system over history. This generates the "experience" needed to train the Guard.
```bash
python Guard/generate_dataset.py
```

### 2. Model Training
Train the LightGBM classifier. It will automatically detect categorical asset IDs and optimize for trade precision.
```bash
python Guard/train_guard.py
```

### 3. Integration
Incorporate the Guard into the final execution logic:
```python
from Guard import TradeGuard

guard = TradeGuard()
prob = guard.predict_proba(market_features, sl, tp, risk, "XAUUSD")

if prob > 0.55:
    # Execute Trade
```

## 📊 Model Details
- **Framework:** LightGBM
- **Inputs:** 140 Market Features + 3 Risk Parameters + Asset ID.
- **Objective:** Binary Classification (Win = 1, Loss = 0).
- **Optimization:** Focuses on minimizing False Positives (preventing bad trades).