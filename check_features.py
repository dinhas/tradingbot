
expected_pressure = [
            "rsi_14", "rsi_velocity", "macd_hist", "macd_div_proxy",
            "obv_slope", "cmf", "force_index", 
            "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
            "volume_ratio", "buying_pressure", "selling_pressure",
            "stoch_k", "stoch_d", "mfi", "williams_r", "roc",
            "gap_size", "close_loc_value"
        ]
        
expected_volatility = [
            "atr_14", "atr_ratio", "bb_width", "bb_pct_b", "std_dev_20",
            "keltner_pos", "donchian_high_dist", "donchian_low_dist",
            "pivot_dist", "avg_candle_range", "parkinson_vol", "z_score",
            "prev_day_high_dist", "prev_day_low_dist", "vol_skew",
            "atr_slope", "high_low_range_pct", "shadow_symmetry"
        ]

expected_trend = [
            "adx_14", "di_plus", "di_minus", "aroon_up", "aroon_down",
            "ema_9", "ema_21", "ema_50", "ema_200",
            "price_vs_ema50", "price_vs_ema200", "ema_ribbon_spread",
            "supertrend", "cloud_pos", "parabolic_sar_pos"
        ]

expected_regime = [
            "hour_sin", "hour_cos", "day_sin", "day_cos",
            "session_london", "session_ny", "session_asian",
            "efficiency_ratio", "hurst_exp", "fractal_dim"
        ]
        
expected_alpha = [
            "alpha_signal", "alpha_conf", 
            "corr_usd", "corr_gold", "corr_sp500",
            "rel_strength", "beta", "spread",
            "tick_vol", "vol_variance", "price_variance",
            "cov_lead", "sector_mom", "global_vol", "global_trend"
        ]

all_expected = expected_pressure + expected_volatility + expected_trend + expected_regime + expected_alpha
print(f"Expected count from FeatureEngine lists: {len(all_expected)}")

# From columns_debug.txt (stripped of prefix)
generated = [
'open', 'high', 'low', 'close', 'volume',
'rsi_14', 'rsi_velocity', 'macd_hist', 'macd_div_proxy', 'obv_slope', 'cmf', 'force_index', 'candle_body_pct', 'upper_wick_pct', 'lower_wick_pct', 'volume_ratio', 'buying_pressure', 'selling_pressure', 'stoch_k', 'stoch_d', 'mfi', 'williams_r', 'roc', 'gap_size', 'close_loc_value',
'atr_14', 'atr_ratio', 'atr_slope', 'bb_width', 'bb_pct_b', 'std_dev_20', 'keltner_pos', 'donchian_high_dist', 'donchian_low_dist', 'pivot_dist', 'avg_candle_range', 'parkinson_vol', 'z_score', 'prev_day_high_dist', 'prev_day_low_dist', 'vol_skew', 'high_low_range_pct', 'shadow_symmetry',
'adx_14', 'di_plus', 'di_minus', 'aroon_up', 'aroon_down', 'ema_9', 'ema_21', 'ema_50', 'ema_200', 'price_vs_ema50', 'price_vs_ema200', 'ema_ribbon_spread', 'supertrend', 'cloud_pos', 'parabolic_sar_pos',
'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'session_london', 'session_ny', 'session_asian', 'efficiency_ratio', 'hurst_exp', 'fractal_dim',
'alpha_signal', 'alpha_conf', 'corr_usd', 'rel_strength', 'beta', 'global_vol', 'global_trend', 'sector_mom', 'corr_gold', 'corr_sp500', 'spread', 'vol_variance', 'price_variance', 'cov_lead', 'tick_vol'
]

print(f"Generated count: {len(generated)}")

missing = [f for f in all_expected if f not in generated]
extra = [f for f in generated if f not in all_expected and f not in ['open', 'high', 'low', 'close', 'volume']]

print(f"Missing from generated: {missing}")
print(f"Extra in generated: {extra}")
