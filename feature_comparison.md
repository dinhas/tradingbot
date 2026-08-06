# Feature Comparison: Hour Feature Calculation

This document outlines and compares how the **Hour of Day** feature is calculated between the offline **Filter model training pipeline** and **live execution system**, including details on the timezone and exact time source used.

---

## 1. Feature Definition and Calculation

### Offline Feature Generator (`Filter/src/feature_engine.py`)
During offline training, features are generated from historical DataFrames indexed by datetime:
1. **Extraction**: The raw hour is extracted from the DataFrame's datetime index:
   ```python
   hours = df.index.hour
   new_cols['hour_of_day'] = hours
   ```
2. **Normalization**: The raw hour values (integers `[0, 23]`) are scaled to center-and-range scale `[-1.0, 1.0]` for the model:
   ```python
   df['hour_of_day'] = (df['hour_of_day'] - 12) / 12.0
   ```

### Live Execution System (`LiveExecution/src/features.py`)
During live trading, the pipeline is as follows:
1. **Extraction**: When cTrader sends a candle close event, `FeatureManager` pushes the new bar to the history buffer.
2. **Delegation**: When executing the inference chain, `FeatureManager.get_filter_features` delegates the history DataFrame directly to the **exact same** `FilterFeatureEngine` class:
   ```python
   _, normalized_df = self.filter_fe.preprocess_data(data_dict)
   obs = self.filter_fe.get_observation_vectorized(normalized_df, asset)
   ```
3. **Normalization**: The same normalization logic scales the live index hour as `(hour - 12) / 12.0`.

---

## 2. Detailed Comparison Matrix

| Aspect | Offline Feature Engine (Training) | Live Execution Engine (Trading) |
| :--- | :--- | :--- |
| **Logic Source** | `Filter/src/feature_engine.py` | `Filter/src/feature_engine.py` via `LiveExecution/src/features.py` |
| **Raw Value Range** | `[0, 23]` (integer) | `[0, 23]` (integer) |
| **Scaling Formula** | `(hour - 12.0) / 12.0` | `(hour - 12.0) / 12.0` |
| **Normalized Range**| `[-1.0, 1.0]` | `[-1.0, 1.0]` |
| **Timezone** | **UTC** | **UTC** |
| **Implementation parity** | **100% Identical** (uses same underlying class) | **100% Identical** (uses same underlying class) |

---

## 3. Time Source and Timezone

The system uses **UTC (Coordinated Universal Time)** as its sole time source, guaranteeing 100% training-serving feature alignment.

1. **cTrader Server Timestamp**: cTrader OpenAPI sends historical bars and live spot events with a native field `utcTimestampInMinutes`. This is the exact number of minutes elapsed since the Unix epoch (Jan 1, 1970 00:00:00 UTC) as defined by the cTrader exchange servers.
2. **Conversion to DateTime**: `FeatureManager` converts this UTC integer timestamp into a Python datetime object:
   ```python
   'timestamp': dt.fromtimestamp(bar.utcTimestampInMinutes * 60)
   ```
   Since the original field is in UTC, `fromtimestamp` yields the correct UTC datetime representing the exact exchange-close moment of the trendbar.
3. **Zero Time-skew**: Using cTrader's native UTC timestamps as the single source of truth completely avoids local server timezone biases, daylight saving time (DST) shifts, or lag-induced clock disparities.
