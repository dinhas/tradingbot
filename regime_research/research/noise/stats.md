# Statistical Profile: NOISE

- Bar count: 1912
- Mean return: -0.000001
- Std return: 0.000257
- Skewness: -0.336369
- Kurtosis: 4.678532

### ADF Test (Stationarity)
- Test Statistic: -1.5538
- p-value: 0.5067
- Interpretation: Non-stationary

### Ljung-Box Test (Autocorrelation)
- p-value (lag 10): 0.0777
- Interpretation: No significant autocorr

### Jarque-Bera Test (Normality)
- p-value: 0.0000
- Interpretation: Non-normal

### Variance Inflation Factor (VIF)

VIF calculation failed.

## Signal Quality (IC & ICIR)

| Feature | IC@1 | IC@5 | IC@10 | ICIR (avg) | Verdict |
|---|---|---|---|---|---|
| rsi14 | -0.039 | -0.018 | 0.009 | 1.125 | USABLE |
| adx14 | -0.003 | 0.004 | 0.022 | 0.920 | USABLE |
| atr_ratio | -0.031 | -0.090 | -0.167 | 1.720 | USABLE |
| hurst | -0.003 | -0.016 | -0.029 | 1.517 | USABLE |
| perm_entropy | -0.043 | -0.066 | -0.092 | 3.299 | USABLE |
| spectral_entropy | 0.015 | 0.023 | -0.015 | 1.081 | USABLE |
| autocorr_1 | 0.002 | -0.030 | -0.044 | 1.316 | USABLE |
| vwap_dev | -0.025 | -0.024 | -0.044 | 3.415 | USABLE |
