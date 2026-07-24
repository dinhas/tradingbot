# Exit Policy Inconsistency Fixed — Backtest Results

Removed the **Signal Flip** exit path from `alpha_lstm_backtest.py` so the backtest now matches the Labeler's triple-barrier policy: exits only via SL, TP, or 6-bar timeout. No more early exits when the direction signal flips or confidence drops.

## Updated Comparison Table

| # | Run | Threshold | Calibration | Signal Flip | PF | Return | Max DD | Trades | Win Rate | Final Equity |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 093437 | 0.55 | None | Yes | 0.52 | -36.7% | -37.9% | 425 | 21.6% | $6,328 |
| 3 | 101816 | 0.55 | T=0.7 | Yes | 0.49 | -46.3% | -47.3% | 547 | 20.8% | $5,368 |
| new | 110311 | 0.60 | T=0.7 | No | 0.59 | -23.7% | -25.5% | 280 | 23.6% | $7,634 |
| 4 | 102034 | 0.65 | T=0.7 | Yes | 0.66 | -6.7% | -9.8% | 79 | 26.6% | $9,334 |
| new | 110509 | 0.70 | T=0.7 | No | 1.31 | +1.5% | -1.6% | 17 | 41.2% | $10,151 |
| 5 | 102410 | 0.70 | T=0.7 | Yes | 1.43 | +1.9% | -1.6% | 17 | 41.2% | $10,189 |

## Per-Asset Breakdown (0.60, no Signal Flip)

| Asset | Trades | Win Rate | PF | PnL |
|---|---|---|---|---|
| EURUSD | 74 | 28.4% | 0.79 | -$221 |
| GBPUSD | 43 | 27.9% | 0.77 | -$176 |
| USDCHF | 72 | 16.7% | 0.43 | -$954 |
| USDJPY | 91 | 23.1% | 0.55 | -$1,023 |

## Per-Asset Breakdown (0.70, no Signal Flip)

| Asset | Trades | Win Rate | PF | PnL |
|---|---|---|---|---|
| USDJPY | 17 | 41.2% | 1.31 | +$151 |

## Key Observations

1. **Removing Signal Flip is not retroactively harmful** — monotonic improvement with threshold is preserved. The policy is now consistent between training labels and backtest evaluation.

2. **The 0.70 result is still 17 USDJPY trades on a single day** (2026-01-25). Statistical fragility unchanged.

3. **USDCHF remains the worst asset** — 16.7% win rate, PF 0.43, -$954 PnL. Dragging down all multi-asset runs.

4. **The direction head is still the bottleneck** — 50.2% holdout accuracy means ~half of the 280 trades at 0.60 threshold go the wrong way. The gate correctly identifies tradeable bars above random (precision 43% vs 29% base rate), but cannot choose the correct side.

5. **Next step** — Replace the oracle-gate + conditional-direction architecture with action-specific long/short net-R regression heads, so trade selection is directly coupled with the direction being executed.
