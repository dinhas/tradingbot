# AGENTS.md — TradeGuard AI v2.7

## Quick Commands

```bash
# Live execution (starts Twisted reactor, requires .env)
python main.py

# Alpha model training (GPU recommended)
cd Alpha && python run_pipeline.py --smoke-test  # quick sanity check
cd Alpha && python run_pipeline.py               # full training

# Backtesting
python backtest/backtest_combined.py --initial-equity 10

# Smoke test (unit test for component instantiation)
python scripts/smoke_test.py

# Feature validation (checks macro features, no look-ahead)
python scripts/validate_features.py
```

## Architecture

Two-layer signal chain: **Alpha (buy/sell direction)** → **Risk (SL/TP/position sizing)**

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `Alpha/` | Signal generation (LSTM + attention) | `run_pipeline.py`, `src/model.py`, `src/feature_engine.py` |
| `RiskLayer/` | Risk management (SL/TP multipliers) | `model.py`, `train.py`, `src/feature_engine.py` |
| `LiveExecution/` | Production orchestrator (Twisted) | `main.py`, `src/orchestrator.py`, `src/models.py` |
| `backtest/` | Backtesting scripts | `backtest_combined.py` |
| `Filter/` | Optional signal filter (RF classifier) | `run_pipeline.py` |
| `models/` | Shared model storage | `checkpoints/`, `risk/` |

## Gotchas

- **RiskLayer casing**: Directory is `RiskLayer/` but imports use `Risklayer.src.*` (lowercase 'l'). The `cerebrium.toml` creates a symlink: `ln -s /cortex/Risklayer /cortex/RiskLayer`. Do not rename the directory.
- **FilterModel missing**: `LiveExecution/src/models.py:12` imports `FilterModel.src.model.FilterClassifier` but no `FilterModel/` directory exists. The code handles this gracefully (logs warning, disables filter). If you need the filter, it lives in `Filter/`.
- **Thresholds loaded from disk**: `LiveExecution/src/config.py:get_thresholds()` reads `backtest/results/optimal_thresholds.json` if it exists, otherwise uses hardcoded defaults. Changing backtest thresholds affects live execution.
- **Alpha model versions**: Checkpoint `format_version` field determines V6 vs V7 architecture. V7 uses GRN + Variable Selection Network. `run_pipeline.py` defaults to V7.
- **sys.path manipulation**: Multiple files add project root to `sys.path` at import time. Do not rely on package-relative imports; use absolute paths from project root.
- **.env required for live**: `CT_APP_ID`, `CT_APP_SECRET`, `CT_ACCOUNT_ID`, `CT_ACCESS_TOKEN` must be set. See `.env.example`.
- **Demo by default**: `CT_HOST_TYPE` defaults to `demo`. Set to `live` explicitly for real trading.

## Training Details

- Alpha training uses purged time-series split (no lookahead): `PURGE_TD` and `EMBARGO_TD` in `run_pipeline.py`
- Dataset generates trade-only sequences (buy vs sell, hold excluded)
- Feature selection: features with ANOVA F-score < 1.0 are zeroed out
- Multi-GPU supported via `DataParallel` when `torch.cuda.device_count() > 1`
- Diagnostics zipped automatically after each training run into `Alpha/diagnostics/`

## Key Constants

- Assets: `EURUSD`, `GBPUSD`, `USDJPY`, `USDCHF` (defined in `shared_constants.py`)
- Sequence length: 25 bars (M5 candles)
- Confidence filter threshold: 0.30 (Risk Layer blocks trades below this)
- Max 1 position per asset in live execution
