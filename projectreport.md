# AI Trading Agent - Project Requirements & Design Document (RPD)

**Version:** 2.0  
**Date:** October 16, 2025  
**Project Type:** Autonomous AI-Powered Scalping System  
**Trading Style:** Scalping (5min-15min)  
**AI Model:** Google Gemini 2.5  
**Broker Integration:** cTrader Open API  

---

## 1. Executive Summary

An autonomous scalping system that leverages Google Gemini 2.5 for rapid trading decisions while executing trades through cTrader's Open API. The system focuses on high-frequency forex scalping with aggressive risk tolerance, targeting quick profits from small price movements.

**Key Objectives:**
- Automate forex scalping (5min-15min timeframes)
- Utilize Gemini 2.5 for intelligent rapid trade decisions
- Execute multiple trades per day (10-30 trades)
- Target 5-20 pip movements per trade
- Manage risk dynamically based on account size
- Provide transparency through decision logging and monitoring
- Close all positions daily to avoid overnight risk

---

## 2. Trading Strategy Specifications

### 2.1 Market & Instruments
- **Market:** Forex
- **Pairs:** Predefined by user (e.g., EURUSD, GBPUSD, USDJPY, AUDUSD)
- **Trading Style:** Scalping
- **Session Focus:** High volatility sessions
  - London Open: 08:00-12:00 GMT
  - NY Open: 13:00-17:00 GMT
  - London/NY Overlap: 13:00-16:00 GMT (best)

### 2.2 Timeframe Strategy
| Timeframe | Purpose | Update Frequency |
|-----------|---------|------------------|
| 1-hour | Trend context | On candle close |
| 15-minute | Trend direction | On candle close |
| 5-minute | **PRIMARY - Entry/Exit** | On candle close |
| 1-minute | Precision timing (optional) | Real-time monitoring |

**Decision Trigger:** Every 5-minute candle close

### 2.3 Trading Hours
- **Active trading:** 07:00 - 22:00 GMT
- **Peak activity:** 08:00-12:00, 13:00-17:00 GMT
- **Stop new entries:** 22:00 GMT
- **Force close all:** 23:30 GMT (30min before daily close)
- **No weekend positions:** All closed before Friday 23:30 GMT

### 2.4 Scalping Targets
| Parameter | Value | Notes |
|-----------|-------|-------|
| Target per trade | 5-20 pips | Quick in/out |
| Typical hold time | 5-45 minutes | Not holding for hours |
| Trades per day | 10-30 trades | High frequency |
| Win rate target | >55% | Need higher win rate for scalping |
| Risk:Reward | 1:1.5 to 1:2.5 | Realistic for scalping |

### 2.5 Decision Model
**Hybrid Approach Optimized for Scalping:**

1. **Technical Layer (Fast Indicators):**
   - EMA (8, 21) - short-term trend
   - RSI (9) - faster momentum
   - Stochastic (5,3,3) - overbought/oversold
   - Bollinger Bands (20,2) - volatility
   - ATR (14) - volatility measurement
   - Volume spikes

2. **AI Layer (Rapid Decision):**
   - Gemini 2.5 analyzes technical data + price action
   - Makes quick decisions (must respond in <3 seconds)
   - Focuses on momentum and short-term patterns
   - Provides reasoning for each decision (logged)
   - Prioritizes quick profits over holding

**Scalping Entry Signals:**
- Price bouncing off EMA
- RSI divergence on 5min
- Stochastic crossover in oversold/overbought
- Breakout from consolidation
- Support/resistance bounces
- Momentum bursts with volume

---

## 3. Risk Management Framework

### 3.1 Position Risk Parameters (Scalping Adjusted)
| Parameter | Value | Notes |
|-----------|-------|-------|
| Risk per trade | 0.5-1% of capital | Lower per trade, more trades |
| Max daily loss | 4% of capital | Tighter control for scalping |
| Max weekly loss | 10% of capital | Stop trading if hit |
| Max concurrent positions | 3 trades | Focus on quality, not quantity |
| Max positions per currency | 2 | Avoid over-exposure |
| Leverage | User-defined | Typically 1:100 to 1:500 |
| Min confidence threshold | 65% | Higher threshold for scalping |

### 3.2 Position Sizing (Scalping)
**Risk-based calculation:**
```
Lot Size = (Account Balance × 0.5-1%) / (Stop Loss in pips × Pip Value)
```
- Smaller risk per trade (0.5-1% vs 2-3%)
- Compensated by higher trade frequency
- Tighter stops (5-15 pips typical)
- Allows for more trades without overleveraging

### 3.3 Stop Loss & Take Profit (Scalping Style)
**Tight Stops:**
- **Stop Loss:** 5-15 pips (based on ATR, typically 1× ATR)
- **Take Profit Ladder:**
  - TP1: 8 pips (close 70%) - lock profit quickly
  - TP2: 15 pips (close 30%) - let runner go
- **Time-based stop:** Close after 1 hour if not moving
- **Breakeven:** Move stop to breakeven after 5 pip profit

**No Trailing Stop:** Too risky for scalping, takes profits too early

### 3.4 Safety Mechanisms (Scalping Specific)
1. **Daily Loss Limit:** Trading pauses if 4% daily loss hit
2. **Consecutive Loss Rule:** Pause after 3 consecutive losses, resume after 30min
3. **Mandatory Daily Close:** All positions closed at 23:30 GMT
4. **Spread Protection:** Don't trade if spread >2.5 pips
5. **Volatility Filter:** Don't trade during extreme volatility (ATR >3× average)
6. **News Filter:** Pause 5min before and 10min after major news
7. **Connection Loss:** Close all positions if API connection lost >2 minutes

---

## 4. System Architecture

### 4.1 High-Level Architecture
```
┌─────────────────────────────────────────────────────────┐
│              Main Orchestrator (main.py)                 │
│           [5-Minute Decision Loop]                       │
└────────────┬────────────────────────────┬───────────────┘
             │                            │
    ┌────────▼─────────┐         ┌───────▼────────┐
    │  Market Data     │         │   AI Decision   │
    │    Service       │────────▶│     Engine      │
    │  (Real-time)     │         │  (Gemini 2.5)   │
    │  WebSocket       │         │  <3sec response │
    └────────┬─────────┘         └───────┬────────┘
             │                            │
             │                   ┌────────▼────────┐
             │                   │ Risk Management │
             │                   │  (Fast Validate)│
             │                   └────────┬────────┘
             │                            │
    ┌────────▼────────────────────────────▼────────┐
    │    Trade Execution Service (Ultra-Fast)      │
    │         Market Orders Only                   │
    └────────┬─────────────────────────────────────┘
             │
    ┌────────▼─────────┐         ┌────────────────┐
    │    Database      │         │   Monitoring   │
    │    (SQLite)      │         │   Dashboard    │
    └──────────────────┘         └────────────────┘
```

### 4.2 Component Specifications

#### 4.2.1 Market Data Service (`market_data.py`)
**Responsibilities:**
- Connect to cTrader Open API via WebSocket (real-time)
- Stream 5-minute candle data
- Calculate technical indicators in real-time
- Monitor spread and volatility
- Ultra-low latency (<500ms data delay)

**Data Structure (Optimized for Speed):**
```python
{
  "pair": "EURUSD",
  "timeframe": "5min",
  "current_candle": {
    "open": 1.0850,
    "high": 1.0855,
    "low": 1.0848,
    "close": 1.0853,
    "volume": 1250
  },
  "last_50_candles": [...],  # Minimal history for speed
  "indicators": {
    "ema_8": 1.0850,
    "ema_21": 1.0848,
    "rsi_9": 62.3,
    "stochastic_k": 75.2,
    "stochastic_d": 72.8,
    "bb_upper": 1.0858,
    "bb_lower": 1.0842,
    "atr": 0.0008  # 8 pips
  },
  "price_action": {
    "trend_5min": "bullish",
    "trend_15min": "bullish",
    "momentum": "strong"
  },
  "spread": 1.2,  # pips
  "timestamp": "2025-10-16T14:35:00Z"
}
```

**Update Frequency:**
- Real-time tick data via WebSocket
- Process indicators on every 5min candle close
- Cache last 50 candles only (speed optimization)
- Spread monitoring: Continuous

**Spread Filter:**
```python
def is_spread_acceptable(pair):
    max_spreads = {
        "EURUSD": 2.0,
        "GBPUSD": 2.5,
        "USDJPY": 2.0,
        "AUDUSD": 2.5
    }
    return current_spread <= max_spreads[pair]
```

#### 4.2.2 AI Decision Engine (`ai_engine.py`)
**Responsibilities:**
- Interface with Gemini 2.5 API
- **CRITICAL:** Get response in <3 seconds
- Format minimal data for AI (reduce tokens)
- Parse AI responses into structured decisions
- Maintain minimal context (last 10 trades only)

**Input to Gemini 2.5 (Minimal for Speed):**
```python
{
  "pair": "EURUSD",
  "price": 1.0853,
  "trend_15min": "bullish",
  "indicators": {
    "ema_8": 1.0850,
    "ema_21": 1.0848,
    "rsi": 62.3,
    "stochastic": 75.2,
    "bb_position": "mid"
  },
  "momentum": "strong",
  "spread": 1.2,
  "open_positions": 1,
  "session": "london_open",
  "last_3_trades": [
    {"outcome": "win", "pips": 8},
    {"outcome": "loss", "pips": -7},
    {"outcome": "win", "pips": 12}
  ]
}
```

**Output from Gemini 2.5 (Fast Response Required):**
```python
{
  "decision": "ENTER_LONG",  # ENTER_LONG/ENTER_SHORT/PASS/EXIT
  "confidence": 72,
  "entry": "market",  # Always market orders for scalping
  "stop_loss_pips": 10,
  "take_profit_pips": 15,
  "reasoning": "Bullish momentum, RSI strong, price above EMAs",
  "expected_duration": "15-30min"
}
```

**Speed Optimizations:**
- Minimal context (cut prompt size by 70%)
- Cache API connection
- Parallel requests for multiple pairs
- Timeout: 3 seconds (fail fast, skip trade)
- Fallback: If Gemini slow, use simple technical rules

**Decision Frequency:**
- Every 5-minute candle close
- Additional check: If price moves >5 pips in 1 minute (momentum scalp)

#### 4.2.3 Risk Management Service (`risk_manager.py`)
**Responsibilities:**
- Ultra-fast validation (<100ms)
- Calculate position sizes
- Monitor account in real-time
- Enforce scalping-specific rules

**Fast Validation Checks:**
```python
def validate_scalp_trade(ai_decision):
    # Speed-optimized checks (fail fast)
    if confidence < 65: return False
    if current_spread > max_spread: return False
    if open_positions >= 3: return False
    if daily_loss >= 4%: return False
    if consecutive_losses >= 3: return False
    if stop_loss < 5 or stop_loss > 15: return False
    if major_news_in_15min(): return False
    return True
```

**Position Sizing (Scalping):**
```python
def calculate_scalp_lot_size(account_balance, stop_loss_pips, pair):
    risk_amount = account_balance * 0.005  # 0.5% risk
    pip_value = get_pip_value(pair)
    lot_size = risk_amount / (stop_loss_pips * pip_value)
    
    # Max size check
    max_lot = (account_balance * leverage) / 100000
    
    return min(lot_size, max_lot)
```

**Emergency Actions (Scalping Specific):**
- 3 consecutive losses → Pause 30 minutes
- Daily loss 4% → Stop trading for day
- Spread widening suddenly → Close all open trades
- Connection unstable → Close all, reconnect

#### 4.2.4 Trade Execution Service (`executor.py`)
**Responsibilities:**
- **CRITICAL:** Ultra-fast execution (<1 second)
- Market orders ONLY (no limit orders for scalping)
- Real-time position monitoring
- Quick exits

**Execution Flow (Optimized for Speed):**
```
1. Receive validated decision (already checked)
2. Place market order immediately (no delays)
3. Set stop loss (server-side)
4. Set take profit (server-side)
5. Start monitoring position
6. Execute partial close at TP1 (70%)
7. Monitor for TP2 or stop loss
8. Update database (async, non-blocking)
```

**Speed Optimizations:**
- Pre-authenticated API connection (keep-alive)
- Market orders only (instant fill)
- Server-side stops (no local monitoring needed)
- Async database writes
- No retry logic (fail = skip trade)

**Position Monitoring (Real-time):**
```python
def monitor_scalp_position(position):
    # Check every 5 seconds
    if unrealized_profit >= 5_pips:
        move_stop_to_breakeven()
    
    if unrealized_profit >= tp1:
        close_partial(70%)  # Lock profit
    
    if time_in_trade > 60_minutes:
        close_full("time_limit")  # Cut dead weight
```

#### 4.2.5 Database Service (`database.py`)
**Database:** SQLite (simple, fast enough for scalping)

**Optimizations:**
- Write async (don't block trading loop)
- Batch inserts every 10 seconds
- Minimal logging during active trading
- Detailed logging after trade closes

**Schema (Same as before, optimized indexes):**
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    pair TEXT,
    direction TEXT,
    entry_time TIMESTAMP,
    entry_price REAL,
    exit_time TIMESTAMP,
    exit_price REAL,
    lot_size REAL,
    stop_loss_pips INTEGER,
    take_profit_pips INTEGER,
    realized_pips REAL,
    pnl REAL,
    hold_time_minutes INTEGER,
    confidence INTEGER,
    reasoning TEXT,
    session TEXT
);
CREATE INDEX idx_entry_time ON trades(entry_time);
CREATE INDEX idx_pair ON trades(pair);
```

#### 4.2.6 Monitoring Dashboard (`monitor.py`)
**Real-Time Scalping Dashboard:**

1. **Live Ticker (Auto-refresh every 2 seconds):**
   - Current positions with live P&L (pips)
   - Entry price, current price, target, stop
   - Time in trade
   - Current spread

2. **Today's Scalping Stats:**
   - Total trades
   - Wins/Losses
   - Total pips captured
   - Win rate
   - Average pips per trade
   - Fastest win/loss

3. **Active Monitoring:**
   - Current spread for each pair
   - Upcoming news events
   - Daily loss meter (%)
   - Consecutive loss counter
   - System latency

4. **Recent Trades (Last 20):**
   - Time, Pair, Direction, Pips, Hold time
   - Color coded: Green (win), Red (loss)

5. **Performance Chart:**
   - Cumulative pips (intraday)
   - Equity curve
   - Trades per hour heatmap

---

## 5. AI Integration Specifications

### 5.1 Gemini 2.5 Prompt Engineering (Scalping Optimized)

**System Prompt (Concise for Speed):**
```
You are an expert forex scalper. You make quick decisions based on 5-minute charts.

Goals:
- Capture 5-20 pip moves
- High win rate (>55%)
- Quick entries, quick exits
- Trade with momentum
- Avoid ranging markets

Respond in JSON format within 2 seconds.
```

**Decision Prompt Template (Minimal):**
```
Scalp decision for {pair}:

Price: {price} | Trend: {trend} | Session: {session}
EMA(8/21): {ema8}/{ema21} | RSI: {rsi} | Stoch: {stoch}
Spread: {spread} pips | Open trades: {count}
Last 3: {recent_trades}

Decision: ENTER_LONG/ENTER_SHORT/PASS/EXIT
Confidence: 0-100
Stop: pips
Target: pips
Why: brief reason

JSON only.
```

**Example Response:**
```json
{
  "decision": "ENTER_LONG",
  "confidence": 75,
  "stop_loss_pips": 10,
  "take_profit_pips": 15,
  "reasoning": "Strong bull momentum, RSI climbing, price above EMAs"
}
```

### 5.2 Context Management (Minimal for Speed)
- **Recent trades:** Last 3 trades only
- **No historical summaries:** Too slow
- **Current session:** London/NY/Overlap
- **Current momentum:** Strong/Weak/Neutral
- **Token limit:** <500 tokens per request

### 5.3 Latency Requirements
| Component | Max Latency | Critical? |
|-----------|-------------|-----------|
| Market data | 500ms | Yes |
| Gemini API | 3 seconds | Yes |
| Risk validation | 100ms | Yes |
| Order execution | 1 second | Yes |
| **Total decision → order** | **<5 seconds** | **CRITICAL** |

If Gemini takes >3 seconds, skip the trade and log timeout.

---

## 6. Configuration Management

### 6.1 Configuration File (`config.json`) - Scalping Version
```json
{
  "trading": {
    "style": "scalping",
    "pairs": ["EURUSD", "GBPUSD", "USDJPY"],
    "leverage": 200,
    "starting_capital": 1000,
    "risk_per_trade_percent": 0.5,
    "max_daily_loss_percent": 4,
    "max_weekly_loss_percent": 10,
    "max_concurrent_positions": 3,
    "max_positions_per_currency": 2,
    "min_confidence_threshold": 65,
    "close_all_daily": true,
    "daily_close_time": "23:30",
    "stop_new_trades_time": "22:00",
    "trading_start_time": "07:00",
    "pause_after_consecutive_losses": 3,
    "pause_duration_minutes": 30
  },
  "scalping": {
    "min_candle_interval": "5min",
    "target_pips_per_trade": [8, 15],
    "max_stop_loss_pips": 15,
    "min_stop_loss_pips": 5,
    "max_spread_pips": {
      "EURUSD": 2.0,
      "GBPUSD": 2.5,
      "USDJPY": 2.0
    },
    "breakeven_trigger_pips": 5,
    "max_hold_time_minutes": 60,
    "tp1_close_percent": 70,
    "tp2_close_percent": 30
  },
  "technical": {
    "ema_periods": [8, 21],
    "rsi_period": 9,
    "stochastic": [5, 3, 3],
    "bollinger_bands": [20, 2],
    "atr_period": 14
  },
  "api": {
    "ctrader": {
      "client_id": "YOUR_CLIENT_ID",
      "client_secret": "YOUR_CLIENT_SECRET",
      "account_id": "YOUR_ACCOUNT_ID",
      "environment": "demo",
      "websocket_url": "wss://...",
      "timeout_seconds": 5
    },
    "gemini": {
      "api_key": "YOUR_GEMINI_API_KEY",
      "model": "gemini-2.5-pro",
      "max_tokens": 500,
      "timeout_seconds": 3
    }
  },
  "monitoring": {
    "dashboard_port": 8080,
    "refresh_interval_seconds": 2,
    "enable_telegram": true,
    "telegram_bot_token": "",
    "telegram_chat_id": ""
  }
}
```

---

## 7. Deployment & Operations

### 7.1 Infrastructure Requirements (Scalping Specific)

**CRITICAL:** Low latency is essential for scalping!

**Option 1: Local Machine (If Close to Broker)**
- Python 3.10+
- 8GB RAM (for fast processing)
- SSD storage
- **Ultra-stable internet** (fiber, <20ms latency to cTrader)
- Wired connection (no WiFi)

**Option 2: Cloud VPS (RECOMMENDED)**
- **Location:** Same region as cTrader servers (Europe/London)
- **Provider:** Vultr, DigitalOcean (choose nearest datacenter)
- **Specs:** 4GB RAM, 2 CPU cores, SSD
- **Network:** Low latency (<10ms to cTrader)
- **Cost:** $20-40/month (worth it for scalping)

### 7.2 Operational Procedures (Scalping)

**Pre-Market Startup (06:45 GMT):**
```
1. System health check
2. Test cTrader connection latency
3. Test Gemini API response time
4. Load configuration
5. Verify spreads are normal
6. Check economic calendar for news
7. Enter active mode at 07:00
```

**During Active Scalping:**
- 5-minute decision loop (continuous)
- Monitor every position (5-second refresh)
- Log all latency metrics
- Auto-restart on component failure

**Hourly Checks:**
- Connection latency still good?
- Spreads still tight?
- System performing as expected?

**Daily Shutdown (23:30 GMT):**
```
1. Stop accepting new trades at 22:00
2. Monitor open positions
3. Close all positions at 23:30 (market orders)
4. Calculate daily stats
5. Generate performance report
6. Backup database
7. Sleep mode until next day
```

### 7.3 Monitoring & Alerts (Scalping)

**Critical Alerts (Immediate Action):**
- Latency spike (>3 seconds to Gemini)
- Connection lost
- 3 consecutive losses
- Daily loss approaching 4%
- Spread widening abnormally

**Warning Alerts:**
- Position held >45 minutes
- Win rate <50% today
- Unusual market volatility

---

## 8. Testing Strategy

### 8.1 Testing Phases (Scalping Focused)

**Phase 1: Latency Testing (Week 1)**
- Test all API response times
- Optimize for speed
- Ensure <5 second decision→execution

**Phase 2: Paper Trading (1 Month Minimum)**
- Demo account with full scalping logic
- Track all metrics
- Monitor latency issues
- Validate 10-30 trades per day

**Phase 3: Micro Live Account (2-3 Months)**
- $500-1000 account
- Prove consistency
- Build confidence

### 8.2 Success Criteria (Paper Trading)
- Win rate: >55%
- Average pips per trade: >3 pips (after spread)
- Trades per day: 10-30
- Max drawdown: <15%
- System latency: <5 seconds
- No connection failures

---

## 9. Performance Metrics & KPIs (Scalping)

### 9.1 Scalping-Specific Metrics
| Metric | Target | Notes |
|--------|--------|-------|
| Win Rate | >55% | Higher needed for scalping |
| Avg Pips Per Trade | >3 pips | After spread costs |
| Trades Per Day | 10-30 | Frequency is key |
| Avg Hold Time | <30 min | Quick scalps |
| Profit Factor | >1.5 | Gross profit / loss |
| Max Drawdown | <15% | Stay tight |
| Sharpe Ratio | >1.5 | Risk-adjusted return |

### 9.2 System Performance (Critical for Scalping)
- **Decision Latency:** <5 seconds (critical)
- **Order Fill Time:** <1 second (critical)
- **Slippage:** <0.5 pips average
- **API Success Rate:** >99.5%
- **System Uptime:** >99.9%

### 9.3 Cost Analysis
**Scalping costs more due to frequency:**
- Spread cost per trade: 1-2 pips
- Total spread cost per day: 10-60 pips
- Must overcome spread costs to profit

---

## 10. Risk Disclosure & Limitations

### 10.1 Scalping-Specific Risks
1. **Spread Costs:** High frequency = high spread costs
2. **Latency Risk:** Slow execution = slippage losses
3. **Over-trading:** 30 trades/day = 30× the risk exposure
4. **Psychological Stress:** High frequency = high intensity
5. **API Rate Limits:** Too many requests = throttling
6. **News Events:** Can wipe out multiple scalp profits instantly

### 10.2 Scalping Challenges
- Requires perfect execution
- No room for system errors
- Extremely time-sensitive
- Higher transaction costs
- More prone to technical issues
- Exhausting to monitor

### 10.3 When to Stop Scalping
- Win rate drops below 50% for 3+ days
- System latency consistently >5 seconds
- 5+ losing days in a row
- Spread costs eating all profits
- Stress/burnout (if manually monitoring)

---

## 11. Development Roadmap

### Phase 1: Core Development (Weeks 1-2)
- [ ] Real-time WebSocket market data
- [ ] cTrader API integration (optimized)
- [ ] Fast indicator calculations
- [ ] Database with async writes

### Phase 2: AI Integration (Week 3)
- [ ] Gemini 2.5 API with timeout handling
- [ ] Minimal prompt engineering
- [ ] Response parsing (<100ms)

### Phase 3: Scalping Logic (Week 4)
- [ ] Risk manager (fast validation)
- [ ] Market order execution
- [ ] Position monitoring (5-second loop)
- [ ] Breakeven logic

### Phase 4: Monitoring & Safety (Week 5)
- [ ] Real-time dashboard
- [ ] Latency monitoring
- [ ] Alert system
- [ ] Emergency stops

### Phase 5: Testing (Weeks 6-8)
- [ ] Latency testing
- [ ] Unit tests
- [ ] Simulation testing

### Phase 6: Paper Trading (Month 3)
- [ ] Demo account deployment
- [ ] Daily monitoring
- [ ] Performance validation

### Phase 7: Live Micro Account (Months 4-5)
- [ ] $500 account
- [ ] Prove profitability
- [ ] Scale gradually

---

## 12. Success Factors (Scalping)

### 12.1 Critical Success Factors
1. **SPEED:** <5 second latency is non-negotiable
2. **Discipline:** Follow risk rules strictly (0.5% per trade)
3. **Uptime:** System must be reliable 99.9%
4. **Spread Awareness:** Only trade during tight spreads
5. **News Avoidance:** Pause during major events
6. **Realistic Expectations:** 3-5 pips per trade is good
7. **Patience:** Paper trade until proven

### 12.2 Why Scalping is Hard
- Requires perfect execution
- Small margins for error
- High spread costs
- Needs low latency infrastructure
- Mentally exhausting if monitored
- Easy to over-trade

### 12.3 Scalping Advantages
- Quick feedback (know results in minutes)
- No overnight risk
- Many opportunities per day
- Can profit in sideways markets
- Positions don't tie up capital long

---

## 13. Appendices

### Appendix A: Technology Stack (Scalping Optimized)
- **Language:** Python 3.10+ (with asyncio)
- **AI API:** Google Gemini 2.5 (timeout: 3s)
- **Broker API:** cTrader WebSocket (real-time)
- **Database:** SQLite (async writes)
- **Web Framework:** FastAPI (async)
- **WebSocket:** websockets library
- **Technical Analysis:** pandas-ta (fast)
- **Scheduling:** asyncio event loop
- **Monitoring:** Real-time dashboard

### Appendix B: Key Libraries
```
google-generativeai  # Gemini API
websockets           # Real-time WebSocket
aiohttp              # Async HTTP
pandas               # Data processing
pandas-ta            # Technical indicators
fastapi              # Async dashboard
uvicorn              # ASGI server
asyncio              # Async operations
sqlalchemy           # Database ORM (async)
python-dotenv        # Environment variables
```

### Appendix C: Performance Benchmarks
**Target latencies for scalping:**
- Market data update: <500ms
- Indicator calculation: <200ms
- Gemini API call: <3s
- Risk validation: <100ms
- Order placement: <1s
- **Total pipeline: <5s**

### Appendix D: File Structure
```
ai-scalping-bot/
├── main.py                    # Main event loop
├── config.json                # Scalping config
├── requirements.txt
├── .env
├── services/
│   ├── market_data.py         # WebSocket real-time
│   ├── ai_engine.py           # Fast Gemini calls
│   ├── risk_manager.py        # Quick validation
│   ├── executor.py            # Market orders
│   └── database.py            # Async writes
├── monitor/
│   ├── dashboard.py           # Real-time FastAPI
│   └── templates/             # Live refresh UI
├── utils/
│   ├── indicators.py          # Fast calculations
│   ├── latency_monitor.py     # Track all delays
│   └── logger.py
├── tests/
│   ├── test_latency.py        # Speed tests
│   ├── test_market_data.py
│   ├── test_ai_engine.py
│   ├── test_risk_manager.py
│   └── test_executor.py
├── data/
│   └── trading_data.db        # SQLite
└── logs/
    ├── trades.log             # All trades
    ├── latency.log            # Performance metrics
    └── errors.log             # System errors
```

---

## 14. Scalping-Specific Implementation Details

### 14.1 Main Event Loop (main.py)
```python
# Pseudo-code for scalping loop

import asyncio
from datetime import datetime

async def scalping_main_loop():
    """
    Main event loop - runs every 5 minutes
    """
    while True:
        try:
            # Wait for next 5-minute candle close
            await wait_for_candle_close("5min")
            
            # Get current time
            current_time = datetime.utcnow()
            
            # Check if within trading hours
            if not is_trading_hours(current_time):
                continue
            
            # Check if close to daily close (22:00-23:30)
            if should_close_all_positions(current_time):
                await close_all_positions()
                continue
            
            # For each configured pair
            for pair in config['pairs']:
                # Skip if spread too wide
                if not is_spread_acceptable(pair):
                    log(f"Skipping {pair} - spread too wide")
                    continue
                
                # Skip if major news in next 15min
                if upcoming_news(pair, minutes=15):
                    log(f"Skipping {pair} - news upcoming")
                    continue
                
                # Get market data
                market_data = await get_market_data(pair)
                
                # Get AI decision (with timeout)
                try:
                    ai_decision = await asyncio.wait_for(
                        get_ai_decision(market_data),
                        timeout=3.0
                    )
                except asyncio.TimeoutError:
                    log(f"AI timeout for {pair} - skipping")
                    continue
                
                # Validate with risk manager
                if validate_trade(ai_decision):
                    # Execute trade
                    await execute_scalp_trade(ai_decision)
                else:
                    log(f"Trade rejected by risk manager")
            
            # Monitor existing positions
            await monitor_open_positions()
            
        except Exception as e:
            log_error(f"Error in main loop: {e}")
            await asyncio.sleep(10)

async def monitor_open_positions():
    """
    Check open positions every 5 seconds
    """
    for position in get_open_positions():
        # Move to breakeven if profit >= 5 pips
        if position.unrealized_pips >= 5:
            await move_stop_to_breakeven(position)
        
        # Close partial at TP1
        if position.unrealized_pips >= position.tp1_pips:
            await close_partial(position, percent=70)
        
        # Time-based exit (1 hour max)
        if position.time_in_trade > 60:
            await close_position(position, reason="time_limit")
```

### 14.2 Fast Indicator Calculation
```python
# Optimized for speed - calculate only what's needed

import pandas as pd
import pandas_ta as ta

def calculate_scalping_indicators(candles):
    """
    Fast indicator calculation for 5-min scalping
    Only last 50 candles needed
    """
    df = pd.DataFrame(candles[-50:])
    
    # EMAs (fast)
    df['ema_8'] = ta.ema(df['close'], length=8)
    df['ema_21'] = ta.ema(df['close'], length=21)
    
    # RSI (fast period)
    df['rsi'] = ta.rsi(df['close'], length=9)
    
    # Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'], 
                     k=5, d=3, smooth_k=3)
    df['stoch_k'] = stoch['STOCHk_5_3_3']
    df['stoch_d'] = stoch['STOCHd_5_3_3']
    
    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    df['bb_upper'] = bbands['BBU_20_2.0']
    df['bb_lower'] = bbands['BBL_20_2.0']
    df['bb_mid'] = bbands['BBM_20_2.0']
    
    # ATR for stop loss
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # Return only latest values
    latest = df.iloc[-1]
    return {
        'ema_8': latest['ema_8'],
        'ema_21': latest['ema_21'],
        'rsi': latest['rsi'],
        'stoch_k': latest['stoch_k'],
        'stoch_d': latest['stoch_d'],
        'bb_upper': latest['bb_upper'],
        'bb_lower': latest['bb_lower'],
        'bb_mid': latest['bb_mid'],
        'atr': latest['atr']
    }
```

### 14.3 Gemini Fast Prompt
```python
# Minimal prompt for <3 second response

def build_scalp_prompt(market_data):
    """
    Ultra-concise prompt for speed
    """
    return f"""
Scalp {market_data['pair']} - Quick decision:

Price: {market_data['price']:.5f}
Trend 15m: {market_data['trend_15min']}
EMA: 8={market_data['ema_8']:.5f}, 21={market_data['ema_21']:.5f}
RSI: {market_data['rsi']:.1f}
Stoch: K={market_data['stoch_k']:.1f}, D={market_data['stoch_d']:.1f}
Spread: {market_data['spread']:.1f} pips
Session: {market_data['session']}
Open: {market_data['open_positions']}/3

Last 3: {format_recent_trades(market_data['last_3_trades'])}

JSON response:
{{
  "decision": "ENTER_LONG|ENTER_SHORT|PASS",
  "confidence": 0-100,
  "stop_pips": 5-15,
  "target_pips": 8-20,
  "why": "brief"
}}
"""

async def get_gemini_decision(prompt, timeout=3):
    """
    Call Gemini with strict timeout
    """
    try:
        response = await asyncio.wait_for(
            gemini_api_call(prompt),
            timeout=timeout
        )
        return parse_json_response(response)
    except asyncio.TimeoutError:
        log_warning("Gemini timeout - skipping trade")
        return {"decision": "PASS"}
    except Exception as e:
        log_error(f"Gemini error: {e}")
        return {"decision": "PASS"}
```

### 14.4 Position Management for Scalping
```python
class ScalpPosition:
    """
    Manages a single scalp position
    """
    def __init__(self, order_response, ai_decision):
        self.id = order_response['orderId']
        self.pair = order_response['pair']
        self.direction = ai_decision['decision']  # LONG/SHORT
        self.entry_price = order_response['fill_price']
        self.lot_size = order_response['lot_size']
        self.stop_loss = self.calculate_stop_price(
            ai_decision['stop_pips']
        )
        self.tp1 = self.calculate_tp_price(
            ai_decision['target_pips'] * 0.5
        )
        self.tp2 = self.calculate_tp_price(
            ai_decision['target_pips']
        )
        self.entry_time = datetime.utcnow()
        self.breakeven_set = False
        self.tp1_hit = False
        
    async def monitor(self):
        """
        Monitor position every 5 seconds
        """
        current_price = await get_current_price(self.pair)
        unrealized_pips = self.calculate_unrealized_pips(current_price)
        time_in_trade = (datetime.utcnow() - self.entry_time).seconds / 60
        
        # Move to breakeven after 5 pip profit
        if not self.breakeven_set and unrealized_pips >= 5:
            await self.move_to_breakeven()
            self.breakeven_set = True
        
        # Close 70% at TP1
        if not self.tp1_hit and self.check_tp1_hit(current_price):
            await self.close_partial(0.7)
            self.tp1_hit = True
        
        # Close all at TP2
        if self.check_tp2_hit(current_price):
            await self.close_full("tp2_hit")
        
        # Time-based exit (1 hour max)
        if time_in_trade > 60:
            await self.close_full("time_limit")
    
    async def move_to_breakeven(self):
        """
        Move stop loss to entry price
        """
        await modify_order(
            self.id,
            new_stop_loss=self.entry_price
        )
        log(f"{self.pair} - Moved to breakeven")
    
    async def close_partial(self, percent):
        """
        Close partial position (e.g., 70% at TP1)
        """
        close_lot_size = self.lot_size * percent
        await close_position_partial(
            self.id,
            lot_size=close_lot_size
        )
        self.lot_size -= close_lot_size
        log(f"{self.pair} - Closed {percent*100}% at TP1")
```

### 14.5 Spread Monitoring
```python
# Critical for scalping - only trade tight spreads

async def monitor_spreads():
    """
    Continuously monitor spreads for all pairs
    """
    spreads = {}
    
    while True:
        for pair in config['pairs']:
            bid, ask = await get_bid_ask(pair)
            spread_pips = calculate_spread_pips(bid, ask, pair)
            spreads[pair] = spread_pips
            
            # Alert if spread widens
            if spread_pips > config['max_spread'][pair]:
                log_warning(f"{pair} spread widened to {spread_pips}")
                
                # Close any open positions if spread too wide
                if spread_pips > config['max_spread'][pair] * 2:
                    await emergency_close_positions(pair)
        
        await asyncio.sleep(1)  # Check every second
    
    return spreads

def calculate_spread_pips(bid, ask, pair):
    """
    Calculate spread in pips
    """
    if 'JPY' in pair:
        return (ask - bid) * 100  # JPY pairs (2 decimals)
    else:
        return (ask - bid) * 10000  # Other pairs (4 decimals)
```

### 14.6 News Calendar Integration
```python
# Avoid trading around major news

import requests
from datetime import datetime, timedelta

def check_upcoming_news(pair, minutes_ahead=15):
    """
    Check if major news in next X minutes
    Returns True if should pause trading
    """
    currency1 = pair[:3]
    currency2 = pair[3:]
    
    # Get news from API (e.g., ForexFactory, Investing.com)
    news_events = get_economic_calendar()
    
    now = datetime.utcnow()
    threshold = now + timedelta(minutes=minutes_ahead)
    
    for event in news_events:
        # Check if high-impact news
        if event['impact'] == 'HIGH':
            # Check if affects our pair
            if event['currency'] in [currency1, currency2]:
                # Check if within time window
                if now < event['time'] < threshold:
                    log(f"HIGH impact news in {minutes_ahead}min: {event['title']}")
                    return True
    
    return False

# Major news to avoid (examples):
HIGH_IMPACT_NEWS = [
    'NFP',  # Non-Farm Payrolls
    'FOMC',  # Federal Reserve
    'CPI',  # Inflation
    'GDP',
    'Interest Rate Decision',
    'ECB Press Conference'
]
```

### 14.7 Latency Monitoring
```python
# Track all component latencies

import time

class LatencyMonitor:
    """
    Monitor system latency for scalping
    """
    def __init__(self):
        self.metrics = {
            'market_data': [],
            'gemini_api': [],
            'risk_validation': [],
            'order_execution': [],
            'total_pipeline': []
        }
    
    def record(self, component, duration_ms):
        """
        Record latency measurement
        """
        self.metrics[component].append(duration_ms)
        
        # Keep only last 100 measurements
        if len(self.metrics[component]) > 100:
            self.metrics[component].pop(0)
        
        # Alert if too slow
        if component == 'gemini_api' and duration_ms > 3000:
            alert(f"Gemini slow: {duration_ms}ms")
        
        if component == 'total_pipeline' and duration_ms > 5000:
            alert(f"Pipeline slow: {duration_ms}ms")
    
    def get_average(self, component):
        """
        Get average latency for component
        """
        if not self.metrics[component]:
            return 0
        return sum(self.metrics[component]) / len(self.metrics[component])
    
    def get_report(self):
        """
        Generate latency report
        """
        return {
            component: {
                'avg_ms': self.get_average(component),
                'max_ms': max(metrics) if metrics else 0,
                'min_ms': min(metrics) if metrics else 0
            }
            for component, metrics in self.metrics.items()
        }

# Usage in main loop
latency_monitor = LatencyMonitor()

async def execute_with_timing(component, func):
    """
    Execute function and record latency
    """
    start = time.time()
    result = await func()
    duration_ms = (time.time() - start) * 1000
    latency_monitor.record(component, duration_ms)
    return result
```

### 14.8 Emergency Stop Mechanisms
```python
# Multiple layers of emergency stops for scalping

class EmergencyStop:
    """
    Emergency stop system
    """
    def __init__(self):
        self.active = False
        self.reason = None
    
    async def check_triggers(self):
        """
        Check all emergency stop conditions
        """
        # 1. Daily loss limit
        if get_daily_loss_percent() >= 4:
            await self.trigger("Daily loss limit (4%) reached")
        
        # 2. Consecutive losses
        if get_consecutive_losses() >= 5:
            await self.trigger("5 consecutive losses")
        
        # 3. System latency
        if latency_monitor.get_average('total_pipeline') > 8000:
            await self.trigger("System latency too high (>8s)")
        
        # 4. Connection issues
        if not await check_connection_health():
            await self.trigger("Connection unstable")
        
        # 5. Unusual market conditions
        if detect_extreme_volatility():
            await self.trigger("Extreme volatility detected")
        
        # 6. Manual trigger
        if check_manual_stop_flag():
            await self.trigger("Manual stop triggered")
    
    async def trigger(self, reason):
        """
        Activate emergency stop
        """
        if self.active:
            return
        
        self.active = True
        self.reason = reason
        
        log_critical(f"EMERGENCY STOP: {reason}")
        
        # Close all positions immediately
        await close_all_positions_market()
        
        # Send alerts
        await send_telegram_alert(f"🚨 EMERGENCY STOP: {reason}")
        await send_email_alert(f"Trading stopped: {reason}")
        
        # Stop trading loop
        stop_trading_loop()
    
    def reset(self):
        """
        Reset emergency stop (manual only)
        """
        if self.active:
            log(f"Emergency stop reset. Previous reason: {self.reason}")
            self.active = False
            self.reason = None
```

---

## 15. Advanced Features (Optional)

### 15.1 Dynamic Position Sizing Based on Win Streak
```python
def calculate_dynamic_position_size(base_risk_percent):
    """
    Increase size during win streaks, decrease during losses
    """
    recent_trades = get_last_10_trades()
    wins = sum(1 for t in recent_trades if t.pnl > 0)
    
    # Win rate adjustment
    win_rate = wins / len(recent_trades)
    
    if win_rate >= 0.7:  # 70%+ win rate
        multiplier = 1.2  # Increase position 20%
    elif win_rate <= 0.4:  # 40%- win rate
        multiplier = 0.7  # Decrease position 30%
    else:
        multiplier = 1.0  # Normal
    
    adjusted_risk = base_risk_percent * multiplier
    
    # Cap at max 1.5%
    return min(adjusted_risk, 1.5)
```

### 15.2 Session-Based Strategy Adjustment
```python
def get_session_parameters():
    """
    Adjust strategy based on trading session
    """
    hour = datetime.utcnow().hour
    
    # London Open (08:00-12:00 GMT) - High volatility
    if 8 <= hour < 12:
        return {
            'confidence_threshold': 70,  # Higher threshold
            'target_pips': [10, 18],  # Larger targets
            'max_positions': 2  # Fewer positions
        }
    
    # NY Open (13:00-17:00 GMT) - Very high volatility
    elif 13 <= hour < 17:
        return {
            'confidence_threshold': 65,
            'target_pips': [12, 20],
            'max_positions': 3
        }
    
    # Overlap (13:00-16:00 GMT) - Best for scalping
    elif 13 <= hour < 16:
        return {
            'confidence_threshold': 60,  # Lower threshold, more trades
            'target_pips': [8, 15],
            'max_positions': 3
        }
    
    # Quiet hours (00:00-07:00, 18:00-23:00 GMT)
    else:
        return {
            'confidence_threshold': 75,  # Very selective
            'target_pips': [5, 10],  # Smaller targets
            'max_positions': 1  # Very few positions
        }
```

### 15.3 Performance Analytics Dashboard
```python
# Real-time analytics for scalping performance

def generate_scalping_analytics():
    """
    Generate detailed scalping statistics
    """
    today_trades = get_today_trades()
    
    return {
        'overview': {
            'total_trades': len(today_trades),
            'wins': sum(1 for t in today_trades if t.pnl > 0),
            'losses': sum(1 for t in today_trades if t.pnl < 0),
            'win_rate': calculate_win_rate(today_trades),
            'total_pips': sum(t.pips for t in today_trades),
            'net_pnl': sum(t.pnl for t in today_trades)
        },
        'timing': {
            'avg_hold_time': calculate_avg_hold_time(today_trades),
            'fastest_win': get_fastest_win(today_trades),
            'slowest_trade': get_slowest_trade(today_trades)
        },
        'quality': {
            'avg_pips_per_trade': calculate_avg_pips(today_trades),
            'avg_win_pips': calculate_avg_win_pips(today_trades),
            'avg_loss_pips': calculate_avg_loss_pips(today_trades),
            'profit_factor': calculate_profit_factor(today_trades)
        },
        'by_session': {
            'london_open': filter_trades_by_session('london', today_trades),
            'ny_open': filter_trades_by_session('ny', today_trades),
            'overlap': filter_trades_by_session('overlap', today_trades)
        },
        'by_pair': {
            pair: filter_trades_by_pair(pair, today_trades)
            for pair in config['pairs']
        },
        'streaks': {
            'current_streak': get_current_streak(today_trades),
            'longest_win_streak': get_longest_win_streak(today_trades),
            'longest_loss_streak': get_longest_loss_streak(today_trades)
        }
    }
```

---

## 16. Final Checklist Before Going Live

### 16.1 Pre-Launch Checklist

**Technical Setup:**
- [ ] Python 3.10+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] cTrader API credentials configured
- [ ] Gemini API key configured
- [ ] Database initialized
- [ ] Config file properly set
- [ ] VPS located near cTrader servers (if using cloud)
- [ ] Latency tested (<5s total pipeline)

**Risk Management:**
- [ ] Risk per trade set to 0.5-1% (not higher!)
- [ ] Daily loss limit set to 4%
- [ ] Max concurrent positions set to 3
- [ ] Spread limits configured for each pair
- [ ] Emergency stop mechanisms tested
- [ ] Consecutive loss pause configured

**Testing Completed:**
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Latency tests pass (<5s)
- [ ] Paper trading completed (min 1 month)
- [ ] Win rate >55% in paper trading
- [ ] Max drawdown <15% in paper trading
- [ ] System uptime >99% in paper trading

**Monitoring Setup:**
- [ ] Dashboard running and accessible
- [ ] Telegram alerts configured (optional)
- [ ] Email alerts configured (optional)
- [ ] Latency monitoring active
- [ ] Trade logging working
- [ ] Error logging working

**Knowledge & Preparation:**
- [ ] Understand scalping is high risk
- [ ] Know how to manually stop the bot
- [ ] Know how to manually close all positions
- [ ] Have cTrader login ready for manual intervention
- [ ] Economic calendar bookmarked
- [ ] Committed to monitoring daily (first month)

**Capital & Account:**
- [ ] Demo account tested thoroughly
- [ ] Starting with micro live account ($500-1000)
- [ ] Never risking money you can't afford to lose
- [ ] Understand this is experimental
- [ ] Ready to stop if underperforming

---

## 17. Conclusion

This AI Scalping Trading Agent is designed for **aggressive, high-frequency forex trading** with the following key characteristics:

**Core Approach:**
- 5-minute scalping with 10-30 trades per day
- AI-powered decisions via Gemini 2.5
- Target 5-20 pips per trade
- Ultra-fast execution (<5 seconds)
- No overnight positions

**Critical Success Factors:**
1. **Speed:** Sub-5-second latency is non-negotiable
2. **Discipline:** Stick to 0.5-1% risk per trade
3. **Spread Awareness:** Only trade tight spreads
4. **Testing:** Mandatory paper trading before live
5. **Monitoring:** Daily oversight, especially first month

**Realistic Expectations:**
- Win rate: 55-65% (good for scalping)
- Average per trade: 3-5 pips net (after spreads)
- Daily target: 20-50 pips (if market cooperates)
- Monthly target: 5-15% (if consistently profitable)
- Drawdowns: Expect 10-20% drawdowns periodically

**Remember:**
- Scalping is one of the hardest trading styles
- Most scalpers fail due to costs and psychology
- This system removes emotion but not risk
- Paper trade until proven (minimum 1 month)
- Start micro, scale slowly if profitable
- Be ready to stop if not working

**Next Steps:**
1. Review this document thoroughly
2. Set up development environment
3. Begin Phase 1 development
4. Test extensively
5. Paper trade patiently
6. Only go live when proven

---

*Good luck, trade safe, and may the pips be with you! 📈*

---

**Document Version:** 2.0 (Scalping Optimized)  
**Last Updated:** October 16, 2025  
**Status:** Ready for Development

---

*End of Document*