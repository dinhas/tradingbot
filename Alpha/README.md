# Alpha Model — Training & Backtest Report

## Overview

This project trains an LSTM-based model on 4 forex pairs (EURUSD, GBPUSD, USDJPY, USDCHF) to predict trade entry signals. The model was originally designed as a 3-class direction classifier, then redesigned into a two-head architecture (tradeability gate + conditional direction) to improve stability and reduce overtrading.

All backtests run on 5000 steps with $10,000 initial equity, 1% risk per trade, spread-based costs.

---

## Original Model (v1) — 3-Class Direction Classifier

**Architecture:** LSTM(64) → Dense(32) → 3-class softmax (short/flat/long)

**Labels:** Barrier-based direction labels (short if price drops below lower barrier, long if above upper, flat otherwise). No explicit tradeability filter — model always trades.

**Training:** 20 epochs, sequence length 50, ~20k sequences. Unstable — inf/nan gradients, oscillating losses, calibration drifted from 0.333 → 0.564.

**Backtest result (5000 steps):**

| Metric | Value |
|---|---|
| Profit Factor | 0.26 |
| Total Return | -98.3% |
| Max Drawdown | -98.3% |
| Total Trades | 4,413 |
| Win Rate | 13.0% |
| Final Equity | $169 |

**Asset breakdown:**

| Asset | Trades | Win Rate | PF | PnL |
|---|---|---|---|---|
| EURUSD | 898 | 17.8% | 0.38 | -$1,385 |
| GBPUSD | 1,119 | 10.9% | 0.18 | -$2,616 |
| USDCHF | 1,296 | 5.9% | 0.09 | -$3,841 |
| USDJPY | 1,100 | 19.5% | 0.46 | -$1,989 |

**Root cause:** Model overtraded (179 trades/day), labels encoded barrier events not net PnL, no tradeability gate, poor calibration.

---

## Redesigned Model (v2) — Two-Head Architecture

### Label Redesign

Old labels were barrier-based (short if price < lower barrier, long if > upper, flat otherwise). New labels emit:

- `tradeable` (bool): Whether a trade is worth taking, based on net-R simulation with `min_edge_r=0.10`
- `direction` (0=short, 1=long): Conditional direction given tradeable
- `net_r` (float): Simulated net reward after spread and barriers
- `valid` (bool): Whether the sample passes all filters

The tradeability gate filters out low-edge setups before the model ever sees them.

### Model Architecture

```
Input (50, 14) → LSTM(64, dropout=0.3) → Dense(32)
    ├── trade_head: Dense(1) → sigmoid     [is this tradeable?]
    └── direction_head: Dense(2) → softmax  [short or long?]
```

**Loss:** `0.65 * BCE(trade) + 0.35 * CE(direction, masked to tradeable samples only)`

### Training

- Trained on Kaggle (GPU)
- 719,864 sequences (14 features), 4 assets
- 25 epochs, batch size 64, LR 1e-4 with cosine decay
- AMP disabled, gradient clipping at 1.0
- Best val loss: 0.69485 at epoch 15

**Holdout evaluation (111k test samples):**

| Metric | Value |
|---|---|
| Trade precision | 42.97% |
| Trade base rate | 29.38% |
| Trade recall | 22.92% |
| F1 | 0.299 |
| Direction accuracy | 50.17% (random) |
| Overall accuracy | 68.42% |

**Key finding:** The direction head learned nothing useful (50% = random). All signal comes from the tradeability gate.

**Feature importance (F-scores):**

| Feature | F-score |
|---|---|
| atr_norm | 41,736 |
| volatility | 33,291 |
| regime | 31,183 |
| is_tradeable | 31,183 |
| bb_width | 19,867 |
| hour | 7,653 |
| rsi / htf_rsi / rsi_momentum | 60–115 |

### Calibration

Temperature scaling fit on validation split:

| Metric | Uncalibrated | Calibrated (T=0.7) |
|---|---|---|
| Brier score | 0.1887 | 0.1848 |
| ECE | 0.1097 | 0.0799 |

Calibration file: `Alpha/models/alpha_calibration.json`

---

## Backtest Results — All Runs

All runs use the same backtest engine: 5000 steps, $10,000 initial equity, 1% risk/trade, spread costs.

### Summary Table

| # | Run | Model | Calibration | Threshold | PF | Return | Max DD | Trades | Win Rate | Final Equity |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 072547 | v1 (old) | — | — | 0.26 | -98.3% | -98.3% | 4,413 | 13.0% | $169 |
| 2 | 093437 | v2 (new) | None | 0.55 | 0.52 | -36.7% | -37.9% | 425 | 21.6% | $6,328 |
| 3 | 101816 | v2 (new) | T=0.7 | 0.55 | 0.49 | -46.3% | -47.3% | 547 | 20.8% | $5,368 |
| 4 | 102034 | v2 (new) | T=0.7 | 0.65 | 0.66 | -6.7% | -9.8% | 79 | 26.6% | $9,334 |
| 5 | 102410 | v2 (new) | T=0.7 | 0.70 | 1.43 | +1.9% | -1.6% | 17 | 41.2% | $10,189 |

### Per-Asset Breakdown — Run 1 (Old Model)

| Asset | Trades | Win Rate | PF | PnL |
|---|---|---|---|---|
| EURUSD | 898 | 17.8% | 0.38 | -$1,385 |
| GBPUSD | 1,119 | 10.9% | 0.18 | -$2,616 |
| USDCHF | 1,296 | 5.9% | 0.09 | -$3,841 |
| USDJPY | 1,100 | 19.5% | 0.46 | -$1,989 |

### Per-Asset Breakdown — Run 2 (New Model, Uncalibrated, 0.55)

| Asset | Trades | Win Rate | PF | PnL |
|---|---|---|---|---|
| EURUSD | 114 | 25.4% | 0.66 | -$510 |
| GBPUSD | 67 | 25.4% | 0.65 | -$374 |
| USDJPY | 109 | 23.9% | 0.58 | -$1,006 |
| USDCHF | 135 | 14.8% | 0.34 | -$1,789 |

### Per-Asset Breakdown — Run 3 (New Model, Calibrated T=0.7, 0.55)

| Asset | Trades | Win Rate | PF | PnL |
|---|---|---|---|---|
| EURUSD | 146 | 24.7% | 0.60 | -$757 |
| GBPUSD | 93 | 20.4% | 0.50 | -$760 |
| USDJPY | 135 | 28.1% | 0.70 | -$741 |
| USDCHF | 173 | 12.1% | 0.27 | -$2,381 |

### Per-Asset Breakdown — Run 4 (New Model, Calibrated T=0.7, 0.65)

| Asset | Trades | Win Rate | PF | PnL |
|---|---|---|---|---|
| GBPUSD | 16 | 31.3% | 0.87 | -$43 |
| USDJPY | 37 | 37.8% | 0.97 | -$25 |
| USDCHF | 26 | 7.7% | 0.17 | -$597 |
| EURUSD | 0 | — | — | $0 |

### Per-Asset Breakdown — Run 5 (New Model, Calibrated T=0.7, 0.70)

| Asset | Trades | Win Rate | PF | PnL |
|---|---|---|---|---|
| USDJPY | 17 | 41.2% | 1.43 | +$189 |

---

## Key Findings

1. **v2 is massively better than v1** — Trades dropped from 4,413 → 425 (uncalibrated), drawdown from -98% → -38%, PF from 0.26 → 0.52. The two-head architecture and tradeability labels prevented catastrophic overtrading.

2. **Direction head is useless** — 50% accuracy on holdout (random). It never learned signal. All positive returns come from the tradeability gate selecting slightly better-than-random entries.

3. **Tradeability gate has weak but real signal** — Precision 43% vs 29% base rate. When selecting only the top-confidence bucket (≥0.70), it reaches 41% win rate on 17 trades.

4. **Calibration helps but requires threshold retuning** — Temperature T=0.7 shifted probability mass down, so the default 0.55 threshold now admits too many low-confidence trades. Raising to 0.70 extracts the useful signal.

5. **Statistically fragile** — The 17 trades at 0.70 threshold are too few to distinguish skill from luck. All trades were on a single asset (USDJPY).

6. **USDCHF is consistently the worst performer** — Worst win rate and PF across all runs. The model struggles most with this pair.

---

## Files & Artifacts

| Path | Description |
|---|---|
| `Alpha/src/labeling.py` | Labeler: net-R simulation, tradeable/direction/net_r outputs |
| `Alpha/src/model.py` | Two-head LSTM model, `trade_direction_loss()` |
| `Alpha/run_pipeline.py` | Training loop, dataset generation (disk-backed memmap), holdout eval |
| `Alpha/src/calibration.py` | Temperature scaling, Brier, ECE, reliability |
| `Alpha/calibrate.py` | Standalone calibration script |
| `Alpha/models/alpha_model.pth` | Trained two-head model (Kaggle, 25 epochs) |
| `Alpha/models/alpha_calibration.json` | Calibration params (T=0.7) |
| `Alpha/models/alpha_calibration.report.json` | Calibration reliability report |
| `Alpha/data/training_set/labels.npz` | New schema labels (719,864 rows) |
| `Alpha/data/training_set/labels_old_schema.npz` | Preserved old labels |
| `Alpha/data/training_set/sequences.npy` | Feature matrix (719,864 × 50 × 14) |
| `Alpha/diagnostics/run_20260724_032019/` | Training diagnostics, epoch curves |
| `backtest/alpha_lstm_backtest.py` | Vectorized backtest engine |
| `backtest/results/` | All backtest metrics, trades, and asset breakdowns |
