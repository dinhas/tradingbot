# Feature Calculation and Normalization Discrepancy Report

## Executive Summary

The investigation into the alpha model's performance degradation from backtesting to live execution has revealed a critical discrepancy in the feature normalization process. The training environment utilizes a dynamic, running normalization (`VecNormalize`), while the live environment employs a static normalization based on a "frozen" set of statistics from a specific point in the training data (`FrozenFeatureNormalizer`). This mismatch is the likely cause of the model's poor performance in live trading.

## Analysis of the Discrepancy

### Training Environment

- The training pipeline, as defined in `Alpha/src/train.py`, uses the `stable_baselines3.common.vec_env.VecNormalize` wrapper.
- `VecNormalize` calculates a running average of the mean and standard deviation of the observation space and normalizes the features on-the-fly.
- This means that the normalization statistics are constantly evolving as the model trains, adapting to the distribution of the training data.

### Live Execution Environment

- The live execution environment, as seen in `LiveExecution/src/features.py`, uses a custom `FrozenFeatureNormalizer` class.
- This class loads normalization statistics (median and interquartile range) from a `feature_normalizer.pkl` file, which is generated from a snapshot of the training data (the last 10,000 rows).
- This approach assumes that the statistical properties of the live market data will be identical to the snapshot of the training data from which the normalization statistics were derived.

### The Core Problem

The discrepancy between these two approaches is a significant issue. Market conditions are non-stationary, and the statistical distribution of market data can change over time. The `VecNormalize` approach used in training can adapt to these changes, but the `FrozenFeatureNormalizer` in the live environment cannot.

When the live market data deviates from the distribution of the training data snapshot, the static normalization will produce skewed feature values. These skewed values will be fed into the model, leading to suboptimal or incorrect predictions.

## Recommendations

To address this issue, the normalization strategy must be consistent between the training and live environments. The following solutions are recommended:

1. **Retrain the Model with `FrozenFeatureNormalizer`:** The most robust solution is to modify the training pipeline to use the `FrozenFeatureNormalizer` instead of `VecNormalize`. This would ensure that the model is trained on features that are normalized in the exact same way as they will be in the live environment. This can be achieved by:
    - Running an initial pass over the training data to calculate and save the normalization statistics using `FrozenFeatureNormalizer`.
    - Modifying the `TradingEnv` to apply the `FrozenFeatureNormalizer` transformation to the observations before they are passed to the agent.

2. **Implement a More Adaptive Live Normalization:** If retraining the model is not feasible, an alternative is to implement a more adaptive normalization strategy in the live environment. This could involve:
    - Using a rolling window to calculate normalization statistics in the live environment, similar to the approach in `VecNormalize`.
    - Periodically recalculating and updating the `feature_normalizer.pkl` file to reflect more recent market conditions.

## Conclusion

The difference in normalization strategies between the training and live environments is a critical flaw that is likely the primary cause of the alpha model's poor performance. By aligning the normalization methods, the model's performance in the live market should more closely reflect its backtested results.
