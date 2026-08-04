# Root-Cause Diagnosis: Cold-Start Data Pipeline & Silent Failures

## 1. Unhandled Deferred Callbacks in cTrader Client
In `LiveExecution/src/ctrader_client.py`, the `_on_connected` method binds and runs `self.on_authenticated()` (which is bound to `orchestrator.bootstrap`) without using `yield` or adding an errback:
```python
if self.on_authenticated:
    self.on_authenticated()
```
Because `orchestrator.bootstrap` is decorated with `@inlineCallbacks`, calling it returns a `Deferred` object. If any exception occurs after the first yield (such as a database failure, network error, or invalid data-load exception), it is caught by `inlineCallbacks` and returned inside the failed `Deferred`. Since the caller does not `yield` this `Deferred` or add an `errback`, any exception that occurs during bootstrap fails silently. This directly caused the issue where "hosted logs show nothing" beyond model/threshold loading.

## 2. No Startup Download Logic for COT and Macro Data
While a daily scheduler (`FeatureManager.start_daily_refresh`) exists to download macro and COT data at midnight, there is **no logic** to download or verify this data on cold startup.
When the application starts with an empty cache directory (`data/` is missing or empty):
- `FeatureManager._load_macro_cot_data` catches the `FileNotFoundError`, logs a warning, and leaves `macro_df` and `cot_df` empty.
- The system continues silently and allows live trading to start, but all macro and COT features are zeroed out (train/serve skew).
- There is no mechanism to block trading when the feature engine lacks the minimum history to generate correct inputs for the models.

## 3. Blocking I/O on the Twisted Event Loop Thread
The existing daily refresh logic runs a subprocess `refresh_market_data.py` on the main event loop thread via `subprocess.run()`, which blocks the reactor loop for several minutes. Similarly, any file-system loading or network scraping directly on the reactor thread freezes the entire bot, causing connection dropouts, missing candle close events, and general system instability.

---

## Solutions Implemented:
1. **Asynchronous Handshake & Handlers**: Wrap `self.on_authenticated()` in `defer.maybeDeferred()` and yield it inside `_on_connected`. Register an explicit error callback (`errback`) to catch and log bootstrap-level failures with a full stack trace.
2. **On-Demand Startup Download & Caching**: Check data presence and freshness in `data/` on startup. If missing, trigger a full pull; if stale, run an incremental refresh to fetch only missing rows for COT, FRED, and yfinance data, preventing lookahead and redundant API calls.
3. **Loud Validation Check**: Fail loudly (block orchestrator start) if macro/COT data is insufficient to compute valid 32-feature matrices.
4. **Off-Reactor Thread Execution**: Run all blocking file writes, reads, and network scraping using `threads.deferToThread` to keep Twisted's async loop highly responsive.
