# Statistical Profile: TRENDING

- Bar count: 869
- Mean return: 0.000004
- Std return: 0.000268
- Skewness: -0.275105
- Kurtosis: 2.762986

### ADF Test (Stationarity)
- Test Statistic: -1.3791
- p-value: 0.5922
- Interpretation: Non-stationary

### Ljung-Box Test (Autocorrelation)
- p-value (lag 10): 0.0729
- Interpretation: No significant autocorr

### Jarque-Bera Test (Normality)
- p-value: 0.0000
- Interpretation: Non-normal

### Variance Inflation Factor (VIF)

VIF calculation failed.

## Signal Quality (IC & ICIR)

| Feature | IC@1 | IC@5 | IC@10 | ICIR (avg) | Verdict |
|---|---|---|---|---|---|
| rsi14 | -0.067 | -0.136 | -0.148 | 3.300 | USABLE |
| adx14 | -0.017 | -0.013 | -0.027 | 3.090 | USABLE |
| atr_ratio | -0.023 | -0.109 | -0.102 | 2.011 | USABLE |
| hurst | 0.026 | 0.066 | 0.041 | 2.669 | USABLE |
| perm_entropy | -0.005 | 0.005 | 0.037 | 0.883 | USABLE |
| spectral_entropy | -0.013 | -0.011 | -0.026 | 2.510 | USABLE |
| autocorr_1 | 0.033 | 0.036 | 0.035 | 33.886 | USABLE |
| vwap_dev | -0.030 | -0.050 | -0.084 | 2.444 | USABLE |
