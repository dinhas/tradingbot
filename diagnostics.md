# TradeGuard AI / TradingBot Diagnostics Log

The following is the logged startup output from executing `main.py` at the repository root:

```log
2026-08-05 16:35:04,931 - LiveExecution - INFO - Instant health-check / dashboard starting on port 8000...
2026-08-05 16:35:04,998 - LiveExecution - INFO - Web Dashboard started on http://0.0.0.0:8000
2026-08-05 16:35:04,999 - LiveExecution - INFO - Dashboard started successfully. Health endpoint is active.
2026-08-05 16:35:04,999 - LiveExecution - INFO - Background thread started. Loading configuration and models...
2026-08-05 16:35:05,002 - LiveExecution - INFO - Thresholds Loaded: Alpha=0.6, Filter=0.72, SL=2.0x ATR, TP=4.0x ATR
2026-08-05 16:35:05,012 - LiveExecution - INFO - Macro data loaded: 0 rows, COT data loaded: 0 rows
2026-08-05 16:35:05,014 - LiveExecution - INFO - Loading Alpha model from /app/Alpha/models/alpha_model.pth...
2026-08-05 16:35:05,311 - LiveExecution - INFO - Loading RF Filter ensemble from /app/Filter/models/filter_rf_ensemble.joblib...
2026-08-05 16:35:11,555 - LiveExecution - WARNING - Could not load RF Filter ensemble from /app/Filter/models/filter_rf_ensemble.joblib due to serialization/version incompatibility: No module named '_loss'. Filter gate will be disabled.
2026-08-05 16:35:11,573 - LiveExecution - INFO - All models loaded. Alpha thresh=0.6, Filter thresh=0.72, SL=2.0x ATR, TP=4.0x ATR
2026-08-05 16:35:11,583 - LiveExecution - INFO - Connecting to cTrader (demo.ctraderapi.com:5035)...
2026-08-05 16:35:11,583 - LiveExecution - INFO - Entering background Twisted event loop...
2026-08-05 16:35:22,037 - LiveExecution - INFO - Connected to cTrader. Authenticating...
2026-08-05 16:35:22,038 - LiveExecution - INFO - Sending ProtoOAApplicationAuthReq (clientId=AaIrnTNy...)...
2026-08-05 16:35:23,149 - LiveExecution - INFO - Received App Auth response: ProtoOAErrorRes
2026-08-05 16:35:23,149 - LiveExecution - ERROR - Authentication / Handshake failed: Application Authentication failed: [CH_CLIENT_AUTH_FAILURE] clientId or clientSecret is incorrect
Traceback (most recent call last):
  File "/app/LiveExecution/src/ctrader_client.py", line 104, in _on_connected
    raise RuntimeError(f"Application Authentication failed: [{error_code}] {desc}")
RuntimeError: Application Authentication failed: [CH_CLIENT_AUTH_FAILURE] clientId or clientSecret is incorrect
2026-08-05 16:35:23,151 - LiveExecution - WARNING - Disconnected from cTrader: [Failure instance: Traceback (failure with no frames): <class 'twisted.internet.error.ConnectionDone'>: Connection was closed cleanly.
]
```

## Summary of Fixes:
1. **Fast Health-check and Port Binding**: The FastAPI Web Dashboard server starts immediately, listening on Port 8000 and serving `/health` returning 200 OK instantly.
2. **Heavy Work Decoupling**: All configuration, PyTorch model loading, RF filter loading, and Twisted reactor loop are offloaded to a background thread to prevent blocking Cerebrium's health probe.
3. **Resilient Model Unpickling**: Wrapped the RF Filter ensemble loader in a granular try-except block so that model deserialization issues (such as `_loss` module changes in scikit-learn version differences) trigger a helpful warning instead of completely crashing the system.
4. **Explicit Handshake Checks**: Enhanced the cTrader application and account auth handshakes to explicitly extract and check the returned protobuf payload type, logging errors loudly (like `CH_CLIENT_AUTH_FAILURE`) instead of timing out silently.
