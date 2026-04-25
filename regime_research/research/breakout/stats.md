# Statistical Profile: BREAKOUT

- Bar count: 218
- Mean return: 0.000103
- Std return: 0.000666
- Skewness: 1.855629
- Kurtosis: 10.662453

### ADF Test (Stationarity)
- Test Statistic: -1.4986
- p-value: 0.5342
- Interpretation: Non-stationary

### Ljung-Box Test (Autocorrelation)
- p-value (lag 10): 0.9863
- Interpretation: No significant autocorr

### Jarque-Bera Test (Normality)
- p-value: 0.0000
- Interpretation: Non-normal

### Variance Inflation Factor (VIF)

VIF calculation failed.

## Signal Quality (IC & ICIR)

| Feature | IC@1 | IC@5 | IC@10 | ICIR (avg) | Verdict |
|---|---|---|---|---|---|
| rsi14 | 0.098 | 0.044 | 0.047 | 2.570 | USABLE |
| adx14 | 0.061 | 0.131 | 0.297 | 1.653 | USABLE |
| atr_ratio | 0.098 | 0.203 | 0.381 | 1.950 | USABLE |
| hurst | 0.125 | 0.361 | 0.317 | 2.615 | USABLE |
| perm_entropy | -0.071 | -0.130 | -0.203 | 2.497 | USABLE |
| spectral_entropy | -0.024 | -0.069 | -0.000 | 1.083 | USABLE |
| autocorr_1 | 0.013 | -0.030 | -0.046 | 1.192 | USABLE |
| vwap_dev | -0.019 | -0.138 | -0.091 | 1.684 | USABLE |
