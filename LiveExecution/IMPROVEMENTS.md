# LiveExecution System Improvements Guide

This document outlines recommended improvements for the logging, dashboard, and telegram messaging systems in the LiveExecution module.

---

## Table of Contents

1. [Logging Improvements](#logging-improvements)
2. [Telegram Messaging Improvements](#telegram-messaging-improvements)
3. [Dashboard Improvements](#dashboard-improvements)
4. [Implementation Priority](#implementation-priority)

---

## Logging Improvements

### 1. Structured Logging (JSON Format)

The current logger uses plain text format. JSON structured logging enables better log parsing, searching, and analysis.

**Current** (`LiveExecution/src/logger.py:23`):
```python
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
```

**Recommended**: Create a JSON formatter class:
```python
import json
import logging
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        if hasattr(record, 'correlation_id'):
            log_obj['correlation_id'] = record.correlation_id
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_obj)
```

### 2. Log Correlation IDs

Add correlation IDs to trace execution across components - essential for debugging candle-to-execution flows.

**Implementation**:
```python
import uuid

def generate_correlation_id(symbol_id, timestamp):
    return f"{symbol_id}-{timestamp}-{uuid.uuid4()[:8]}"

# Usage in orchestrator.py
correlation_id = generate_correlation_id(asset_name, time.time())
self.logger.info(f"[{correlation_id}] M5 candle closed for {asset_name}")
```

### 3. Performance Logging

Add explicit timing metrics throughout the execution pipeline.

**Add to `orchestrator.py`**:
```python
import time

def on_m5_candle_close(self, symbol_id, trendbar):
    start_time = time.time()
    # ... existing code ...
    
    # Log timing breakdown
    self.logger.info(f"[PERF] Total cycle: {time.time() - start_time:.3f}s")
```

**Recommended timing metrics**:
- Data fetch latency
- Feature extraction time
- Inference time per model
- Order execution latency
- Database operation times
- Total cycle time

### 4. Granular Log Levels

Currently using mostly `info()` and `error()`. Add more granular levels:

| Current | Recommended | Use Case |
|---------|-------------|----------|
| `logger.info()` | `logger.debug()` | Inference details, model inputs/outputs |
| - | `logger.warning()` | Soft limits, position count near max |
| - | `logger.info()` | Trade decisions |
| - | `logger.info()` | System events (connections, startups) |
| `logger.error()` | `logger.error()` | Failures |
| - | `logger.critical()` | Kill switch, fatal errors |

### 5. Missing Logged Events

Add logging for these untracked events:

| Event | Location | Priority |
|-------|----------|----------|
| Position close reason | `orchestrator.py:on_order_execution` | High |
| Risk model decisions | `orchestrator.py:run_inference_chain` | High |
| Threshold values used | `orchestrator.py:run_inference_chain` | Medium |
| Account state changes | `orchestrator.py:update_account_state` | Medium |
| Connection state changes | `ctrader_client.py` | High |
| Model loading events | `main.py` | Low |
| Order fill details | `orchestrator.py:execute_decision` | High |

### 6. Log Rotation Policies

**Current** (`LiveExecution/src/logger.py:26-27`):
```python
file_handler = RotatingFileHandler(
    log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
```

**Recommended improvements**:
- Separate logs for trades vs system events
- Compress rotated logs
- Add log retention policies
- Add size-based and time-based rotation

```python
from logging.handlers import TimedRotatingFileHandler

# Separate trade logger
trade_logger = logging.getLogger("LiveExecution.Trades")
trade_handler = TimedRotatingFileHandler(
    log_dir / "trades.log",
    when="midnight",
    interval=1,
    backupCount=30,
    encoding='utf-8'
)
trade_handler.suffix = "%Y%m%d"
trade_logger.addHandler(trade_handler)
```

---

## Telegram Messaging Improvements

### 1. Enhanced Command Set

**Current commands** (`LiveExecution/src/notifications.py:51-53`):
```python
app.add_handler(CommandHandler("start", self._start_command))
app.add_handler(CommandHandler("status", self._status_command))
app.add_handler(CommandHandler("positions", self._positions_command))
```

**Add these commands**:

```python
# Add to notifications.py
from telegram.ext import CommandHandler, filters

app.add_handler(CommandHandler("help", self._help_command))
app.add_handler(CommandHandler("daily", self._daily_command))
app.add_handler(CommandHandler("trades", self._trades_command, filters=filters.Regex(r'^\d+$')))
app.add_handler(CommandHandler("config", self._config_command))
app.add_handler(CommandHandler("health", self._health_command))

async def _help_command(self, update, context):
    msg = """
📖 **Available Commands:**

/start - Register for notifications
/status - Account summary
/positions - Active positions
/daily - Force daily summary
/trades [n] - Recent n trades (default 10)
/config - Current thresholds
/health - System health status
    """
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def _daily_command(self, update, context):
    # Force daily summary
    ...

async def _trades_command(self, update, context):
    limit = int(context.args[0]) if context.args else 10
    # Return recent trades
    ...

async def _config_command(self, update, context):
    msg = f"⚙️ **Current Thresholds**\nMeta: {self.config.get('META_THRESHOLD')}\n..."
    ...

async def _health_command(self, update, context):
    # Return system health
    ...
```

### 2. Notification Types - Missing Alerts

**Current notifications** (`LiveExecution/src/notifications.py`):
- Trade executed
- Errors
- Pulse check (every 2 hours)
- Daily summary

**Add these notifications**:

| Notification | Trigger | Priority |
|--------------|---------|----------|
| Position closed | `orchestrator.py:on_order_execution` | High |
| SL hit | Position closed with reason="SL" | High |
| TP hit | Position closed with reason="TP" | High |
| Manual close | `orchestrator.py:close_position_by_id` | Medium |
| Threshold breach | Meta/Quality/Risk below threshold | Medium |
| Drawdown alerts | Configurable % drawdown | High |
| Connection lost | WebSocket disconnection | High |
| Connection restored | WebSocket reconnection | Medium |
| Model reload | Model refresh completed | Low |

**Example - Enhanced trade notification**:
```python
def send_trade_closed(self, details):
    """Enhanced trade closure notification."""
    symbol = details.get('symbol')
    pnl = details.get('pnl', 0)
    reason = details.get('reason', 'Unknown')  # SL, TP, MANUAL, SIGNAL
    
    emoji = "🔴" if pnl < 0 else "🟢"
    reason_emoji = {
        "SL": "🛑", "TP": "🎯", "MANUAL": "👤", "SIGNAL": "📡"
    }.get(reason, "❓")
    
    msg = (
        f"{emoji} **POSITION CLOSED**\n"
        f"Symbol: `{symbol}`\n"
        f"PnL: `${pnl:+.2f}`\n"
        f"Reason: {reason_emoji} {reason}"
    )
    self.send_message(msg)
```

### 3. Alert Thresholds Configuration

Make alerts configurable in `LiveExecution/src/config.py`:

```python
# Add to config
config["TELEGRAM_DRAWDOWN_ALERT"] = float(env.get("TELEGRAM_DRAWDOWN_ALERT", "5.0"))
config["TELEGRAM_PNL_MILESTONE"] = float(env.get("TELEGRAM_PNL_MILESTONE", "1.0"))
config["TELEGRAM_PULSE_INTERVAL"] = int(env.get("TELEGRAM_PULSE_INTERVAL", "7200"))
config["TELEGRAM_ERROR_ALERTS"] = env.get("TELEGRAM_ERROR_ALERTS", "true").lower() == "true"
```

**Use in orchestrator**:
```python
def _check_drawdown_alert(self):
    equity = self.portfolio_state.get('equity', 0)
    peak = self.portfolio_state.get('peak_equity', equity)
    drawdown = 1.0 - (equity / peak) if peak > 0 else 0
    threshold = self.config.get('TELEGRAM_DRAWDOWN_ALERT', 5.0) / 100.0
    
    if drawdown >= threshold:
        self.notifier.send_message(f"⚠️ **DRAWDOWN ALERT**: {drawdown:.2%}")
```

### 4. Message Reliability

Add message queuing and retry logic:

```python
class TelegramNotifier:
    def __init__(self, config):
        # ... existing code ...
        self.message_queue = asyncio.Queue()
        self.failed_messages = []
        
    def send_message(self, content):
        """Queue message for reliable delivery."""
        if not self.chat_id:
            self.logger.warning("Telegram chat_id not set.")
            return
            
        asyncio.run_coroutine_threadsafe(
            self._send_with_retry(content),
            self.loop
        )
        
    async def _send_with_retry(self, content, max_retries=3):
        for attempt in range(max_retries):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id, 
                    text=content, 
                    parse_mode=ParseMode.MARKDOWN
                )
                return
            except Exception as e:
                self.logger.warning(f"Telegram send attempt {attempt+1} failed: {e}")
                await asyncio.sleep(2 ** attempt)
                
        self.logger.error(f"Failed to send message after {max_retries} attempts")
        self.failed_messages.append(content)
```

### 5. Rich Message Formatting

Enhance trade notifications with more details:

**Current** (`LiveExecution/src/notifications.py:118-130`):
```python
def send_trade_event(self, details):
    msg = (
        "🚀 **TRADE EXECUTED**\n"
        f"**Symbol:** `{symbol}`\n"
        f"**Action:** {action}\n"
        f"**Size:** {size}"
    )
```

**Enhanced**:
```python
def send_trade_event(self, details):
    symbol = details.get('symbol')
    action = details.get('action')
    size = details.get('size')
    entry = details.get('entry_price', 'N/A')
    sl = details.get('sl', 'N/A')
    tp = details.get('tp', 'N/A')
    
    emoji = "🟢" if action == "BUY" else "🔴"
    
    msg = (
        f"{emoji} **TRADE EXECUTED**\n"
        f"**Symbol:** `{symbol}`\n"
        f"**Action:** {action}\n"
        f"**Size:** {size}\n"
        f"**Entry:** {entry}\n"
        f"**SL:** {sl} | **TP:** {tp}"
    )
```

---

## Dashboard Improvements

### 1. Real-Time Updates

**Current**: Full page reload every 30 seconds (`index.html:391`):
```javascript
setTimeout(() => window.location.reload(), 30000);
```

**Recommended**: Add WebSocket for live updates.

**Backend** (`dashboard/main.py`):
```python
from fastapi import WebSocket

class DashboardServer:
    def __init__(self, orchestrator):
        # ... existing code ...
        self.websockets = set()
        
    def _setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websockets.add(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle incoming messages if needed
            finally:
                self.websockets.discard(websocket)
                
    async def broadcast_update(self, event_type, data):
        """Broadcast update to all connected clients."""
        import json
        message = json.dumps({"type": event_type, "data": data})
        for ws in self.websockets:
            await ws.send_text(message)
```

**Frontend** (`index.html`):
```javascript
// Replace polling with WebSocket
let ws = new WebSocket(`ws://${location.host}/ws`);

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'equity_update':
            updateEquityDisplay(data.data);
            break;
        case 'trade_closed':
            showTradeClosedNotification(data.data);
            window.location.reload();
            break;
        case 'position_update':
            updatePositionsDisplay(data.data);
            break;
    }
};

// Keep-alive
setInterval(() => ws.send('ping'), 30000);
```

### 2. Enhanced Metrics Display

Add new metrics to dashboard API and frontend:

**Backend - Add to `dashboard/main.py`**:
```python
@self.app.get("/api/system_health")
async def system_health():
    import psutil
    import time
    
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "uptime": time.time() - getattr(self, 'start_time', time.time()),
        "last_inference": self.orchestrator.last_inference_time,
        "connection_status": "connected",  # Check from client
    }

@self.app.get("/api/positions")
async def positions():
    return {
        "active": self.orchestrator.active_positions,
        "exposure": calculate_total_exposure(),
        "margin_used": get_margin_usage(),
    }

@self.app.get("/api/stats/extended")
async def extended_stats():
    return {
        "win_streak": calculate_win_streak(),
        "loss_streak": calculate_loss_streak(),
        "avg_win": calculate_avg_win(),
        "avg_loss": calculate_avg_loss(),
        "avg_duration": calculate_avg_duration(),
    }
```

### 3. System Health Monitoring

Add system health panel to dashboard:

**Frontend** (`index.html`):
```html
<!-- Add to top stats cards -->
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-6">
    <!-- Existing 4 cards -->
    
    <!-- CPU -->
    <div class="glass-panel rounded-xl p-5">
        <p class="text-gray-400 text-sm">CPU</p>
        <h3 class="text-xl font-bold" id="cpu-display">--%</h3>
    </div>
    
    <!-- Memory -->
    <div class="glass-panel rounded-xl p-5">
        <p class="text-gray-400 text-sm">Memory</p>
        <h3 class="text-xl font-bold" id="memory-display">--%</h3>
    </div>
    
    <!-- Uptime -->
    <div class="glass-panel rounded-xl p-5">
        <p class="text-gray-400 text-sm">Uptime</p>
        <h3 class="text-xl font-bold" id="uptime-display">--</h3>
    </div>
</div>
```

```javascript
// Add to JavaScript
async function loadSystemHealth() {
    const res = await fetch('/api/system_health');
    const data = await res.json();
    
    document.getElementById('cpu-display').textContent = data.cpu_percent + '%';
    document.getElementById('memory-display').textContent = data.memory_percent + '%';
    document.getElementById('uptime-display').textContent = formatUptime(data.uptime);
}

function formatUptime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return `${h}h ${m}m`;
}

setInterval(loadSystemHealth, 10000);
loadSystemHealth();
```

### 4. Inference Debugging View

Add a panel showing current model inputs/outputs:

**Backend** (`dashboard/main.py`):
```python
@self.app.get("/api/debug/inference")
async def debug_inference():
    if not self.orchestrator.fm.is_ready():
        return {"error": "System not ready"}
    
    debug_data = {}
    for asset in self.orchestrator.fm.assets:
        alpha_obs = self.orchestrator.fm.get_alpha_observation(
            asset, self.orchestrator.portfolio_state
        )
        debug_data[asset] = {
            "alpha_observation": alpha_obs.tolist() if hasattr(alpha_obs, 'tolist') else alpha_obs,
            "current_price": float(self.orchestrator.fm.history[asset].iloc[-1]['close']),
            "position_locked": asset in self.orchestrator.active_positions,
        }
    
    return debug_data
```

### 5. Trade Analysis Enhancement

Enhance trade history with duration and more details:

**Database** (`database.py`):
```python
def get_trade_duration(self, pos_id):
    """Calculate trade duration in minutes."""
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT entry_time, exit_time 
            FROM trades 
            WHERE pos_id = ?
        ''', (pos_id,))
        row = cursor.fetchone()
        if row and row[0] and row[1]:
            from datetime import datetime
            entry = datetime.fromisoformat(row[0])
            exit = datetime.fromisoformat(row[1])
            return (exit - entry).total_seconds() / 60
    return None
```

### 6. Backtest Comparison

Add comparison with backtest results:

**Backend** (`dashboard/main.py`):
```python
@self.app.get("/api/backtest/comparison")
async def backtest_comparison():
    backtest_path = Path(project_root) / "backtest" / "results" / "equity_curve.json"
    if backtest_path.exists():
        with open(backtest_path) as f:
            backtest_data = json.load(f)
    
    live_equity = self.orchestrator.db.get_equity_history()
    
    return {
        "backtest": backtest_data,
        "live": live_equity,
        "comparison": {
            "backtest_return": calculate_return(backtest_data),
            "live_return": calculate_return(live_equity),
        }
    }
```

### 7. API Endpoints Enhancement

**Current endpoints** (`dashboard/main.py`):
- `/` - Main dashboard
- `/api/equity_history` - Equity curve
- `/api/close/{pos_id}` - Close position
- `/api/kill` - Kill switch

**Recommended additions**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system_health` | GET | CPU, memory, uptime |
| `/api/model_outputs` | GET | Current inference data |
| `/api/positions` | GET | Detailed position info |
| `/api/config` | GET | Current thresholds |
| `/api/trades` | GET | Filterable trade history |
| `/api/logs` | GET | Recent log entries |
| `/api/debug/inference` | GET | Model inputs/outputs |
| `/api/backtest/comparison` | GET | Backtest vs live |

---

## Implementation Priority

### Phase 1 - Quick Wins (High Impact, Low Effort)

| # | Feature | Files to Modify |
|---|---------|-----------------|
| 1 | Telegram: Add position closed notifications | `orchestrator.py`, `notifications.py` |
| 2 | Telegram: Enhanced trade messages | `notifications.py` |
| 3 | Dashboard: System health panel | `dashboard/main.py`, `index.html` |
| 4 | Logging: Add correlation IDs | `orchestrator.py`, `logger.py` |
| 5 | Telegram: Add `/help`, `/config`, `/health` commands | `notifications.py` |

### Phase 2 - Core Improvements (High Impact, Medium Effort)

| # | Feature | Files to Modify |
|---|---------|-----------------|
| 1 | Dashboard: WebSocket real-time updates | `dashboard/main.py`, `index.html` |
| 2 | Telegram: Configurable alert thresholds | `config.py`, `orchestrator.py` |
| 3 | Logging: Structured JSON logging | `logger.py` |
| 4 | Dashboard: Inference debug view | `dashboard/main.py`, `index.html` |
| 5 | Telegram: Message retry logic | `notifications.py` |

### Phase 3 - Advanced Features (High Impact, High Effort)

| # | Feature | Files to Modify |
|---|---------|-----------------|
| 1 | Dashboard: Backtest comparison | `dashboard/main.py`, `index.html` |
| 2 | Dashboard: Advanced trade analytics | `database.py`, `dashboard/main.py`, `index.html` |
| 3 | Logging: Separate trade/system logs | `logger.py` |
| 4 | Telegram: Full command suite | `notifications.py` |
| 5 | Dashboard: User settings panel | `dashboard/main.py`, `index.html` |

---

## Code Location Reference

| Component | File Path |
|-----------|-----------|
| Logger | `LiveExecution/src/logger.py` |
| Telegram Notifier | `LiveExecution/src/notifications.py` |
| Orchestrator | `LiveExecution/src/orchestrator.py` |
| Dashboard Backend | `LiveExecution/dashboard/main.py` |
| Dashboard Frontend | `LiveExecution/dashboard/templates/index.html` |
| Database | `LiveExecution/src/database.py` |
| Config | `LiveExecution/src/config.py` |
| Main Entry | `LiveExecution/main.py` |

---

*Document generated for TradeGuard LiveExecution System improvements.*
