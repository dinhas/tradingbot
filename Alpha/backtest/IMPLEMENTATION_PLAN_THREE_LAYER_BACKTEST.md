# Three-Layer Backtest Implementation Plan
**Date**: December 21, 2025  
**Purpose**: Backtest complete trading system: Alpha → Risk → TradeGuard  
**Base File**: `backtest_combined.py` (Alpha + Risk, 743 lines)  
**New File**: `backtest_full_system.py` (Alpha + Risk + TradeGuard)

---

## Executive Summary

This plan outlines the development of a comprehensive backtesting framework that integrates all three layers of the trading system:

1. **Alpha Layer**: Generates directional signals (long/short/flat)
2. **Risk Layer**: Determines stop-loss, take-profit, and position sizing
3. **TradeGuard Layer**: Meta-labeling filter that approves or blocks trades

The backtest will maintain **complete audit trail** of:
- Executed trades (approved by TradeGuard)
- Blocked trades (rejected by TradeGuard)
- TradeGuard confidence scores and reasoning
- Comparative analysis (what would have happened if we ignored TradeGuard)

---

## File Structure

### New File Location
```
e:\tradingbot\Alpha\backtest\backtest_full_system.py
```

### Relationship to Existing Files
```
Alpha/backtest/
├── backtest.py              # Alpha-only backtest (baseline)
├── backtest_combined.py     # Alpha + Risk backtest (DO NOT EDIT)
└── backtest_full_system.py  # Alpha + Risk + TradeGuard (NEW)
```

---

## Data Architecture

### Model Paths (Inputs)

#### 1. Alpha Model
- **Path**: `checkpoints/8.03.zip` (or similar)
- **Input**: 140-feature market observation
- **Output**: 5 values (one per asset) in range [-1, 1]
  - `> 0.33` → Long signal
  - `< -0.33` → Short signal
  - `[-0.33, 0.33]` → Flat/neutral

#### 2. Risk Model
- **Path**: `RiskLayer/models/risk_model_final.zip`
- **Input**: 165-feature observation
  - Market state (140 features)
  - Account state (5 features)
  - History (20 features)
- **Output**: 3 values in range [-1, 1]
  - `action[0]` → Stop-loss multiplier (0.2-2.0 ATR)
  - `action[1]` → Take-profit multiplier (0.5-4.0 ATR)
  - `action[2]` → Risk percentage (0-100% of max risk)

#### 3. TradeGuard Model
- **Path**: `TradeGuard/tradeguard_results/guard_model.txt`
- **Type**: LightGBM Booster (NOT Stable Baselines3 PPO)
- **Input**: Feature vector (determined from training dataset)
- **Output**: Probability score [0.0, 1.0]
- **Threshold**: From `TradeGuard/tradeguard_results/model_metadata.json`

### Data Directory
- **Path**: `Alpha/backtest/data` (existing, used by combined backtest)
- **Format**: Parquet files with OHLCV data for all 5 assets
- **Assets**: EURUSD, GBPUSD, USDJPY, USDCHF, XAUUSD

---

## TradeGuard Integration Analysis

### Model Loading Difference
**CRITICAL**: TradeGuard uses **LightGBM**, NOT PPO!

```python
# INCORRECT (PPO approach from Alpha/Risk):
# tradeguard_model = PPO.load(model_path)

# CORRECT (LightGBM approach):
import lightgbm as lgb
tradeguard_model = lgb.Booster(model_file='TradeGuard/tradeguard_results/guard_model.txt')

# Load threshold from metadata
import json
with open('TradeGuard/tradeguard_results/model_metadata.json', 'r') as f:
    metadata = json.load(f)
    threshold = metadata['threshold']  # e.g., 0.5
```

### Feature Construction for TradeGuard

#### Required Input Features (from `generate_dataset.py`)

The TradeGuard model expects features derived from:

##### Group A: Alpha Model Confidence (5 features)
```python
# From Alpha model's raw output before thresholding
alpha_confidence = {
    'eurusd_confidence': abs(alpha_action[0]),  # Distance from 0
    'gbpusd_confidence': abs(alpha_action[1]),
    'usdjpy_confidence': abs(alpha_action[2]),
    'usdchf_confidence': abs(alpha_action[3]),
    'xauusd_confidence': abs(alpha_action[4])
}
```

##### Group B: Market Context (Technical Indicators)
From current market observation (already in Alpha env):
- ATR (Average True Range)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators
- **Source**: `precomputed_features` from dataset generation

##### Group C: Risk Model Outputs (3 features)
```python
risk_features = {
    'sl_multiplier': sl_mult,  # 0.2-2.0
    'tp_multiplier': tp_mult,  # 0.5-4.0
    'risk_raw': risk_raw       # 0.0-1.0
}
```

##### Group D: Trade Characteristics (Derived)
```python
trade_characteristics = {
    'rr_ratio': tp_mult / sl_mult,  # Risk-reward ratio
    'position_size_pct': size_pct,  # From calculate_position_size()
    'direction': direction,          # 1 or -1
    'equity_ratio': equity / initial_equity,
    'drawdown': 1.0 - (equity / peak_equity)
}
```

##### Group E: Signal Persistence & Reversal (2 features)
```python
# From _calculate_persistence_reversal method
persistence_features = {
    'persistence_score': 0.0,  # How long same direction
    'reversal_score': 0.0      # Recent direction changes
}
```

##### Group F: Cross-Asset Correlation Signals (4 features)
```python
# Simultaneous signals on correlated pairs
correlation_features = {
    'eurusd_gbpusd_alignment': direction_eurusd * direction_gbpusd,
    'usdjpy_usdchf_alignment': direction_usdjpy * direction_usdchf,
    'n_assets_long': sum(d == 1 for d in directions.values()),
    'n_assets_short': sum(d == -1 for d in directions.values())
}
```

**TOTAL FEATURES**: ~30-50 (exact count from training dataset schema)

#### Feature Extraction Strategy

**Option 1: Use Existing Preprocessing Pipeline** (RECOMMENDED)
```python
# Import from TradeGuard's generate_dataset.py
from TradeGuard.src.generate_dataset import LightweightDatasetEnv

# The env's _record_signal method builds the feature vector
# We can extract that logic into a standalone function
```

**Option 2: Rebuild Feature Vector Manually**
- Risk: Feature drift if implementation differs from training
- Mitigation: Validate against training dataset schema

---

## Implementation Architecture

### Class Structure

```python
class ThreeLayerBacktest:
    """
    Full system backtest: Alpha → Risk → TradeGuard
    
    Inherits structure from CombinedBacktest but adds TradeGuard filtering.
    """
    
    def __init__(self, alpha_model, risk_model, tradeguard_model, 
                 tradeguard_threshold, data_dir, initial_equity=10):
        # Similar to CombinedBacktest.__init__
        # + Load TradeGuard model and threshold
        
    def build_tradeguard_features(self, asset, direction, alpha_action, 
                                   risk_action, sl_mult, tp_mult, risk_raw, 
                                   size_pct, price, atr):
        """
        Construct feature vector for TradeGuard model.
        
        Returns:
            np.ndarray: Feature vector matching training schema
        """
        pass
    
    def evaluate_tradeguard(self, features):
        """
        Get TradeGuard probability and decision.
        
        Returns:
            dict: {
                'probability': float,
                'decision': bool,  # True = approve, False = block
                'threshold': float
            }
        """
        pass
    
    def run_backtest(self, episodes=1):
        """
        Main backtest loop with three-layer decision flow:
        1. Alpha generates direction
        2. Risk generates SL/TP/sizing
        3. TradeGuard approves or blocks
        """
        pass
```

### Decision Flow in `run_backtest()`

```
For each timestep:
│
├─── Get Alpha model prediction
│    └─── Parse to directional signals (5 assets)
│
├─── For each asset with direction != 0:
│    │
│    ├─── Build Risk observation (165 features)
│    ├─── Get Risk model prediction (SL/TP/sizing)
│    ├─── Calculate position size
│    │
│    ├─── Build TradeGuard features
│    ├─── Get TradeGuard probability
│    │
│    ├─── IF probability >= threshold:
│    │    ├─── APPROVED → Execute trade
│    │    └─── Log: executed_trades
│    │
│    └─── ELSE:
│         ├─── BLOCKED → Simulate virtual outcome
│         └─── Log: blocked_trades (with theoretical PnL)
│
└─── Advance timestep, update positions, calculate PnL
```

---

## Data Structures

### 1. Executed Trades Log

```python
executed_trades = [{
    'timestamp': str,
    'asset': str,
    'direction': int,        # 1 or -1
    'entry_price': float,
    'position_size': float,
    'lots': float,
    
    # Risk Layer parameters
    'sl_price': float,
    'tp_price': float,
    'sl_mult': float,
    'tp_mult': float,
    'risk_raw': float,
    
    # TradeGuard parameters
    'tradeguard_probability': float,
    'tradeguard_threshold': float,
    'tradeguard_confidence': float,  # probability - threshold (margin)
    
    # Outcome (filled on trade close)
    'exit_price': float,
    'exit_timestamp': str,
    'net_pnl': float,
    'gross_pnl': float,
    'outcome': str,  # 'WIN' | 'LOSS' | 'BREAKEVEN'
    'exit_reason': str,  # 'SL_HIT' | 'TP_HIT' | 'DIRECTION_CHANGE' | 'EOD'
    'hold_time_minutes': float
}]
```

### 2. Blocked Trades Log

```python
blocked_trades = [{
    'timestamp': str,
    'asset': str,
    'direction': int,
    'entry_price': float,
    
    # Why it was proposed
    'alpha_raw_signal': float,
    'alpha_confidence': float,
    
    # Risk parameters (what would have been)
    'sl_mult': float,
    'tp_mult': float,
    'risk_raw': float,
    'proposed_size': float,
    'proposed_lots': float,
    
    # TradeGuard decision
    'tradeguard_probability': float,
    'tradeguard_threshold': float,
    'tradeguard_margin': float,  # How far below threshold
    
    # Virtual simulation (what WOULD have happened)
    'theoretical_entry_price': float,
    'theoretical_sl_price': float,
    'theoretical_tp_price': float,
    'theoretical_pnl': float,     # Simulated outcome
    'theoretical_outcome': str,   # 'WIN' | 'LOSS' | 'NEUTRAL'
    'simulation_exit_step': int,  # How many steps to SL/TP
    
    # Classification
    'block_quality': str,  # 'GOOD_BLOCK' (avoided loss) | 'BAD_BLOCK' (missed profit)
}]
```

### 3. TradeGuard Analysis Summary

```python
tradeguard_summary = {
    # Overall statistics
    'total_signals': int,           # Alpha signals generated
    'approved_trades': int,         # Passed TradeGuard
    'blocked_trades': int,          # Rejected by TradeGuard
    'approval_rate': float,         # approved / total_signals
    
    # Approval quality
    'approved_wins': int,
    'approved_losses': int,
    'approved_win_rate': float,
    'approved_pnl': float,
    
    # Blocking quality
    'good_blocks': int,             # Blocked losses (avoided)
    'bad_blocks': int,              # Blocked wins (missed)
    'block_accuracy': float,        # good_blocks / blocked_trades
    'avoided_loss_value': float,    # Sum of theoretical losses avoided
    'missed_profit_value': float,   # Sum of theoretical profits missed
    'net_value_add': float,         # avoided_loss - missed_profit
    
    # TradeGuard confidence distribution
    'avg_approved_probability': float,
    'avg_blocked_probability': float,
    'probability_separation': float,  # How well it discriminates
    
    # Per-asset breakdown
    'per_asset_approval_rate': dict,
    'per_asset_block_quality': dict
}
```

---

## Output Files

All outputs to: `Alpha/backtest/results/`

### 1. Metrics JSON
**Filename**: `metrics_full_system_{timestamp}.json`

```json
{
    "total_return": 0.25,
    "sharpe_ratio": 1.5,
    "max_drawdown": -0.15,
    "profit_factor": 1.8,
    "win_rate": 0.52,
    
    "tradeguard_metrics": {
        "approval_rate": 0.65,
        "block_accuracy": 0.70,
        "net_value_add": 150.0,
        "avoided_loss_value": 200.0,
        "missed_profit_value": 50.0
    },
    
    "comparison": {
        "with_tradeguard": {
            "total_return": 0.25,
            "sharpe_ratio": 1.5,
            "total_trades": 650
        },
        "without_tradeguard": {
            "total_return": 0.18,
            "sharpe_ratio": 1.2,
            "total_trades": 1000
        }
    }
}
```

### 2. Trade Logs (CSV)

#### `trades_full_system_{timestamp}.csv`
- All executed trades (approved by TradeGuard)
- Columns from Executed Trades Log structure

#### `blocked_trades_full_system_{timestamp}.csv`
- All blocked trades with theoretical outcomes
- Columns from Blocked Trades Log structure

#### `all_signals_full_system_{timestamp}.csv`
- Combined view: executed + blocked
- Useful for ML model retraining

### 3. Analysis Reports

#### `tradeguard_analysis_{timestamp}.json`
- Complete TradeGuard performance breakdown
- From TradeGuard Analysis Summary structure

#### `asset_breakdown_full_system_{timestamp}.csv`
- Per-asset performance metrics
- Compare approval rates and block quality across assets

### 4. Visualizations

#### `equity_curve_comparison_{timestamp}.png`
- Three lines:
  1. Full system (Alpha + Risk + TradeGuard)
  2. Without TradeGuard (Alpha + Risk only)
  3. Difference (value added by TradeGuard)

#### `tradeguard_decision_distribution_{timestamp}.png`
- Histogram of TradeGuard probabilities
- Separate distributions for:
  - Approved trades
  - Blocked trades (good blocks)
  - Blocked trades (bad blocks)

#### `block_quality_timeline_{timestamp}.png`
- Over time: ratio of good blocks vs bad blocks
- Identify if TradeGuard degrades in certain market regimes

#### `approval_rate_by_asset_{timestamp}.png`
- Bar chart: approval rate for each asset
- Identify if TradeGuard is biased toward/against certain assets

---

## Implementation Phases

### Phase 1: Core Structure (No Code Yet)
**Objective**: Class skeleton and data flow

1. Clone `backtest_combined.py` → `backtest_full_system.py`
2. Rename class: `CombinedBacktest` → `ThreeLayerBacktest`
3. Update docstrings and file header
4. Add TradeGuard-specific imports
5. Update `__init__` signature to accept TradeGuard model

### Phase 2: TradeGuard Integration
**Objective**: Load model and make predictions

1. Implement `load_tradeguard_model()` function
   - Load LightGBM booster
   - Load metadata.json for threshold
   - Validate model structure

2. Implement `build_tradeguard_features()`
   - Extract Alpha confidence
   - Extract Risk parameters
   - Compute derived features (RR ratio, etc.)
   - Build feature vector matching training schema

3. Implement `evaluate_tradeguard()`
   - Call `model.predict(features)`
   - Compare to threshold
   - Return decision + probability

### Phase 3: Decision Flow Integration
**Objective**: Insert TradeGuard into main loop

1. Modify `run_backtest()` main loop:
   - After Risk model prediction
   - Before position size calculation
   - Add TradeGuard evaluation

2. Branch logic:
   - IF approved: proceed to execution (existing code)
   - IF blocked: create virtual trade for simulation

3. Record keeping:
   - Append to `executed_trades` list
   - Append to `blocked_trades` list

### Phase 4: Virtual Trade Simulation
**Objective**: Simulate blocked trades to measure opportunity cost

1. Reuse existing `_simulate_trade_outcome()` from combined backtest
2. For each blocked trade:
   - Create virtual position
   - Look ahead to SL/TP hit
   - Record theoretical PnL
   - Classify as good/bad block

### Phase 5: Metrics & Analysis
**Objective**: Comprehensive reporting

1. Extend `BacktestMetrics` class:
   - Add TradeGuard-specific metrics
   - Compute block quality statistics
   - Calculate value-add metrics

2. Implement comparison metrics:
   - Simulate "what if we ignored TradeGuard"
   - Compare Sharpe, drawdown, total return

3. Generate analysis reports:
   - `tradeguard_analysis.json`
   - Per-asset breakdowns

### Phase 6: Visualization
**Objective**: Visual analysis tools

1. Equity curve comparison
2. TradeGuard probability distributions
3. Block quality over time
4. Approval rate by asset

### Phase 7: Validation & Testing
**Objective**: Ensure correctness

1. Unit tests for feature construction
2. Validate against training dataset schema
3. Sanity checks:
   - Approval rate reasonable (30-70%)
   - Block quality better than random (>50%)
   - Net value-add positive

4. Edge case handling:
   - Missing features
   - Model loading failures
   - Data gaps

---

## Feature Construction Details

### Schema Validation Strategy

**CRITICAL**: Features MUST match training schema exactly

1. **Extract Training Schema**:
   ```python
   # Load training dataset
   df = pd.read_parquet('TradeGuard/data/guard_dataset.parquet')
   
   # Get feature columns (exclude label, asset, timestamp)
   feature_cols = [c for c in df.columns 
                   if c not in ['label', 'asset', 'timestamp']]
   
   print(f"Expected features: {len(feature_cols)}")
   print(feature_cols)
   ```

2. **Create Feature Builder**:
   ```python
   class TradeGuardFeatureBuilder:
       def __init__(self, feature_schema):
           self.feature_schema = feature_schema
           self.n_features = len(feature_schema)
       
       def build_features(self, context):
           """
           Args:
               context: dict with all required inputs
           Returns:
               np.ndarray: Feature vector in correct order
           """
           features = np.zeros(self.n_features, dtype=np.float32)
           
           # Fill features according to schema
           for i, feature_name in enumerate(self.feature_schema):
               features[i] = self._get_feature_value(feature_name, context)
           
           return features
       
       def _get_feature_value(self, feature_name, context):
           """Map feature name to extraction logic"""
           if feature_name.startswith('alpha_'):
               return self._extract_alpha_feature(feature_name, context)
           elif feature_name.startswith('risk_'):
               return self._extract_risk_feature(feature_name, context)
           # ... etc
   ```

3. **Validate at Runtime**:
   ```python
   # Before prediction
   assert features.shape[0] == tradeguard_model.num_feature()
   ```

### Recommended Feature Groups

Based on `generate_dataset.py` analysis:

| Group | Count | Source | Examples |
|-------|-------|--------|----------|
| **Alpha Confidence** | 5 | Alpha model output | `alpha_eurusd_confidence`, `alpha_gbpusd_confidence` |
| **Market Indicators** | 20-30 | Precomputed features | `rsi_14`, `atr_14`, `macd`, `bb_width` |
| **Risk Parameters** | 3 | Risk model output | `sl_multiplier`, `tp_multiplier`, `risk_percentage` |
| **Trade Derived** | 5-10 | Calculated | `rr_ratio`, `position_size_pct`, `drawdown`, `equity_ratio` |
| **Signal Context** | 2-4 | Historical tracking | `persistence_score`, `reversal_score` |
| **Cross-Asset** | 4-6 | Multi-asset signals | `eurusd_gbpusd_correlation`, `total_long_signals` |

**Total**: 40-60 features (verify against actual training dataset)

---

## Risk Mitigation

### 1. Feature Mismatch
**Risk**: Production features differ from training  
**Impact**: Silent model degradation  
**Mitigation**:
- Load training dataset schema at startup
- Validate feature vector shape before every prediction
- Log first 10 feature vectors for manual inspection
- Unit tests comparing to known-good examples

### 2. Model Loading Failure
**Risk**: TradeGuard model file missing or corrupted  
**Impact**: Backtest cannot run  
**Mitigation**:
- Validate file exists before loading
- Fallback mode: run without TradeGuard (warn user)
- Try-except with detailed error messages

### 3. Data Type Inconsistencies
**Risk**: Float32 vs Float64, NaN handling  
**Impact**: Model errors or incorrect predictions  
**Mitigation**:
- Explicit dtype conversion: `features.astype(np.float32)`
- NaN safety: `np.nan_to_num(features, nan=0.0)`
- Match training pipeline exactly

### 4. Performance Degradation
**Risk**: 2x model calls per signal (Risk + TradeGuard)  
**Impact**: Slow backtests  
**Mitigation**:
- LightGBM is fast (~0.1ms per prediction)
- Batch predictions if possible
- Profile and optimize bottlenecks

### 5. Virtual Trade Simulation Accuracy
**Risk**: Theoretical PnL doesn't match reality  
**Impact**: Incorrect block quality assessment  
**Mitigation**:
- Reuse proven `_simulate_trade_outcome()` logic
- Include slippage in virtual trades
- Validate against actual executed trades (sample comparison)

---

## Success Criteria

### Functional Requirements
- ✅ Backtest completes without errors
- ✅ All three models load correctly
- ✅ Executed trades match expected format
- ✅ Blocked trades have theoretical outcomes
- ✅ Output files generated correctly

### Performance Requirements
- ✅ Backtest runtime < 2x combined backtest (acceptable overhead)
- ✅ Memory usage reasonable (< 16GB)

### Quality Requirements
- ✅ TradeGuard approval rate: 30-70% (not too permissive or restrictive)
- ✅ Block accuracy > 55% (better than random)
- ✅ Net value-add > 0 (TradeGuard improves results)

### Validation Requirements
- ✅ Feature vector matches training schema (dimension check)
- ✅ TradeGuard probabilities in valid range [0, 1]
- ✅ Comparison metrics show statistical significance

---

## Command Line Interface

```bash
python Alpha/backtest/backtest_full_system.py \
    --alpha-model checkpoints/8.03.zip \
    --risk-model RiskLayer/models/risk_model_final.zip \
    --tradeguard-model TradeGuard/tradeguard_results/guard_model.txt \
    --tradeguard-metadata TradeGuard/tradeguard_results/model_metadata.json \
    --tradeguard-dataset TradeGuard/data/guard_dataset.parquet \
    --data-dir Alpha/backtest/data \
    --output-dir Alpha/backtest/results \
    --episodes 1 \
    --verbose
```

### Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--alpha-model` | Yes | Path to Alpha PPO model | - |
| `--risk-model` | Yes | Path to Risk PPO model | - |
| `--tradeguard-model` | Yes | Path to TradeGuard LightGBM model | - |
| `--tradeguard-metadata` | Yes | Path to metadata.json (threshold) | - |
| `--tradeguard-dataset` | Yes | Path to training dataset (for schema) | - |
| `--data-dir` | No | Backtest data directory | `Alpha/backtest/data` |
| `--output-dir` | No | Results output directory | `Alpha/backtest/results` |
| `--episodes` | No | Number of backtest episodes | `1` |
| `--verbose` | No | Enable detailed logging | `False` |

---

## Testing Strategy

### Unit Tests

```python
# tests/test_tradeguard_features.py

def test_feature_vector_shape():
    """Ensure feature vector matches training schema"""
    builder = TradeGuardFeatureBuilder(schema)
    features = builder.build_features(mock_context)
    assert features.shape[0] == expected_n_features

def test_feature_dtype():
    """Ensure correct data type"""
    features = builder.build_features(mock_context)
    assert features.dtype == np.float32

def test_no_nans():
    """Ensure no NaN values"""
    features = builder.build_features(mock_context)
    assert not np.any(np.isnan(features))

def test_tradeguard_prediction():
    """Test TradeGuard model prediction"""
    probability = tradeguard_model.predict([features])[0]
    assert 0.0 <= probability <= 1.0
```

### Integration Tests

```python
def test_full_backtest_run():
    """Test complete backtest execution"""
    metrics = run_full_system_backtest(args)
    assert metrics is not None
    assert 'tradeguard_metrics' in metrics

def test_blocked_trade_simulation():
    """Test virtual trade outcome calculation"""
    blocked_trade = simulate_blocked_trade(context)
    assert 'theoretical_pnl' in blocked_trade
    assert 'theoretical_outcome' in blocked_trade
```

### Validation Tests

```python
def test_schema_match():
    """Compare production features to training dataset"""
    training_df = pd.read_parquet('TradeGuard/data/guard_dataset.parquet')
    training_features = training_df.drop(['label', 'asset', 'timestamp'], axis=1)
    
    production_features = builder.build_features(sample_context)
    
    assert production_features.shape[0] == training_features.shape[1]
```

---

## Expected Outcomes

### Baseline Comparison

| Metric | Alpha Only | Alpha + Risk | Alpha + Risk + TradeGuard |
|--------|------------|--------------|---------------------------|
| Total Return | ~15% | ~20% | **25%** ⬆️ |
| Sharpe Ratio | ~0.8 | ~1.2 | **1.5** ⬆️ |
| Max Drawdown | -25% | -18% | **-15%** ⬆️ |
| Win Rate | 48% | 50% | **52%** ⬆️ |
| Total Trades | 1500 | 1200 | **800** ⬇️ |
| Profit Factor | 1.2 | 1.5 | **1.8** ⬆️ |

### TradeGuard Value-Add

- **Approval Rate**: ~60% (blocks 40% of signals)
- **Block Accuracy**: ~70% (7 out of 10 blocks are good)
- **Net Value-Add**: +$150 (avoided $200 losses, missed $50 profits)
- **Confidence Separation**: 0.25 (approved avg 0.70, blocked avg 0.45)

### Insights Expected

1. **Asset-Specific Behavior**:
   - XAUUSD may have lower approval rate (more volatile)
   - EURUSD may have higher block accuracy (more predictable)

2. **Market Regime Sensitivity**:
   - Trending markets: Higher approval rate
   - Choppy markets: More blocks (false signals filtered)

3. **Trade Quality Improvement**:
   - Executed trades have higher avg profit
   - Fewer whipsaw losses
   - Better risk-adjusted returns

---

## Next Steps After Implementation

### 1. Hyperparameter Tuning
- Adjust TradeGuard threshold (0.4, 0.5, 0.6)
- Test different approval rate targets
- Optimize for Sharpe vs total return

### 2. Feature Engineering
- Add new features to TradeGuard training
- Re-train model with expanded feature set
- A/B test improvements

### 3. Live Paper Trading
- Deploy three-layer system to paper account
- Monitor approval rates in real-time
- Validate backtest assumptions

### 4. Continuous Improvement
- Retrain TradeGuard monthly on recent data
- Track block quality over time
- Adapt to market regime changes

---

## Appendix: Code Snippets (Reference Only)

### A. TradeGuard Model Loading

```python
def load_tradeguard_model(model_path, metadata_path):
    """Load TradeGuard LightGBM model and threshold"""
    import lightgbm as lgb
    import json
    
    # Load model
    model = lgb.Booster(model_file=str(model_path))
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    threshold = metadata.get('threshold', 0.5)
    
    logger.info(f"Loaded TradeGuard model: {model.num_feature()} features")
    logger.info(f"Threshold: {threshold:.4f}")
    
    return model, threshold, metadata
```

### B. Feature Schema Extraction

```python
def extract_feature_schema(dataset_path):
    """Extract feature names from training dataset"""
    df = pd.read_parquet(dataset_path)
    
    # Exclude non-feature columns
    exclude_cols = ['label', 'asset', 'timestamp', 'date']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    logger.info(f"Extracted {len(feature_cols)} features from training dataset")
    
    return feature_cols
```

### C. Virtual Trade Outcome

```python
def simulate_blocked_trade(env, asset, direction, sl, tp, entry_price, atr):
    """
    Simulate what would have happened if trade was executed.
    Reuses existing _simulate_trade_outcome logic.
    """
    # Create virtual position
    virtual_pos = {
        'direction': direction,
        'entry_price': entry_price,
        'size': 1.0,  # Dummy size for percentage calculation
        'sl': sl,
        'tp': tp,
        'entry_step': env.current_step
    }
    
    # Store original position
    original_pos = env.positions[asset]
    env.positions[asset] = virtual_pos
    
    # Simulate outcome
    theoretical_pnl = env._simulate_trade_outcome(asset)
    
    # Restore state
    env.positions[asset] = original_pos
    
    return theoretical_pnl
```

---

## Document Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-21 | 1.0 | Initial comprehensive plan |

---

**Plan Status**: ✅ **COMPLETE - READY FOR IMPLEMENTATION**

**Approval Required**: Yes  
**Estimated Implementation Time**: 6-8 hours  
**Complexity**: High (three model integration + virtual simulation)  
**Risk Level**: Medium (feature schema validation critical)
