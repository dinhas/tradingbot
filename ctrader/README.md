# cTrader Module - Modular Trading Bot Architecture

A comprehensive, modular Python trading bot for cTrader API integration. This module provides a scalable architecture for trading operations including market data, order management, account information, and more.

## 🏗️ Architecture Overview

This module follows a **Service-Oriented Modular Design** pattern, providing clean separation of concerns and easy extensibility.

```
ctrader/
├── __init__.py
├── core/                       # Core Infrastructure
│   ├── __init__.py
│   ├── connection_manager.py   # Authentication & connection lifecycle
│   ├── message_router.py       # Route messages to appropriate services
│   └── base_service.py         # Base class for all services
├── services/                   # Specialized Service Modules
│   ├── __init__.py
│   ├── symbols_service.py      # Symbol management (refactored existing)
│   ├── market_data_service.py  # Candles, ticks, order book data
│   ├── account_service.py      # Account info, balance, positions
│   ├── trading_service.py      # Place/modify/cancel orders
│   ├── position_service.py     # Position management
│   └── history_service.py      # Historical data, trade history
├── models/                     # Data Structures
│   ├── __init__.py
│   ├── symbol.py              # Symbol data structures
│   ├── candle.py              # Market data structures
│   ├── order.py               # Order structures
│   └── account.py             # Account structures
├── utils/                      # Utility Functions
│   ├── __init__.py
│   ├── validators.py          # Input validation
│   └── formatters.py          # Data formatting
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install ctrader-open-api twisted python-dotenv
   ```

2. **Environment Variables** (`.env` file in project root):
   ```env
   CTRADER_APP_CLIENT_ID=your_client_id
   CTRADER_APP_CLIENT_SECRET=your_client_secret
   CTRADER_ACCOUNT_ID=your_account_id
   ACCESS_TOKEN=your_access_token
   ```

### Basic Usage

```python
from ctrader.core import ConnectionManager, MessageRouter
from ctrader.services import SymbolsService, AccountService, TradingService
from ctrader.models.account import TradeSide

# Initialize core components
connection_manager = ConnectionManager(use_demo=True)
message_router = MessageRouter()

# Initialize services
symbols_service = SymbolsService(connection_manager, message_router)
account_service = AccountService(connection_manager, message_router)
trading_service = TradingService(connection_manager, message_router)

# Connect and start
connection_manager.connect()

# Place a market order with automatic SL/TP
trading_service.place_market_order(
    symbol_id="1",  # EURUSD
    trade_side=TradeSide.BUY,
    volume=0.01,
    stop_loss=1.0950,
    take_profit=1.1050,
    comment="Automated trading"
)
```

## 📋 Service Modules

### Core Infrastructure

#### ConnectionManager
- **Purpose**: Handles authentication and connection lifecycle
- **Features**:
  - Demo/Live environment switching
  - Automatic authentication flow
  - Connection state management
  - Callback system for connection events

#### MessageRouter
- **Purpose**: Routes incoming API messages to appropriate service handlers
- **Features**:
  - Message type registration
  - Handler management
  - Default handler support
  - Error handling

#### BaseService
- **Purpose**: Common functionality for all services
- **Features**:
  - Data persistence (JSON files)
  - Memory caching
  - Logging utilities
  - Service lifecycle management

### Trading Services

#### SymbolsService
- **Purpose**: Manage trading symbols
- **Operations**:
  - Fetch symbols from API
  - Cache symbols locally
  - Search and filter symbols
  - Symbol metadata management

#### MarketDataService
- **Purpose**: Real-time and historical market data
- **Operations**:
  - Fetch candlestick data (OHLCV)
  - Subscribe to tick data
  - Order book snapshots
  - Market depth information

#### AccountService
- **Purpose**: Account information and management
- **Operations**:
  - Account details
  - Balance information
  - Margin calculations
  - Account permissions

#### TradingService ✅
- **Purpose**: Order management and execution
- **Operations**:
  - Place market orders with automatic SL/TP management
  - Place limit and stop orders
  - Modify position stop loss and take profit
  - Cancel existing orders
  - Track order execution and status
  - Handle order errors and callbacks
- **Key Feature**: Automatically adds SL/TP to market orders after execution

#### PositionService
- **Purpose**: Position management
- **Operations**:
  - Current positions
  - Position modifications
  - Close positions
  - Position history

#### HistoryService
- **Purpose**: Historical data and trade history
- **Operations**:
  - Trade history
  - Order history
  - Historical market data
  - Performance analytics

## 🔧 Key Features

### Modular Design Benefits
- **Maintainability**: Each service is focused and easy to understand
- **Scalability**: Add new features without touching existing code
- **Testability**: Mock and test individual components
- **Reusability**: Use services in different contexts
- **Flexibility**: Swap implementations without affecting other parts

### Service Pattern Advantages
- **Single Responsibility**: Each service handles one domain
- **Loose Coupling**: Services work independently
- **Easy Testing**: Mock individual services
- **Extensible**: Add new services seamlessly

### Data Management
- **Service-specific storage**: Each service manages its own data files
- **Unified cache**: Optional shared cache for cross-service data
- **Event system**: Services notify each other of important events

## 📚 Implementation Phases

### Phase 1: Core Foundation ✅
- [x] Create folder structure
- [x] Implement ConnectionManager
- [x] Implement MessageRouter
- [x] Implement BaseService

### Phase 2: Basic Services ✅
- [x] Refactor existing symbols logic to SymbolsService
- [x] Add AccountService for account information
- [x] Implement complete account models and data structures
- [ ] Implement MarketDataService for candles/ticks

### Phase 3: Trading Operations ✅
- [x] Implement TradingService for order operations
- [x] Add comprehensive order models (Market, Limit, Stop orders)
- [x] Implement automatic SL/TP management for market orders
- [x] Add position SL/TP modification capabilities
- [x] Implement order tracking and error handling
- [ ] Add PositionService for position management
- [ ] Create HistoryService for historical data

### Phase 4: Advanced Features
- [x] Add comprehensive error handling for trading operations
- [x] Implement extensive order lifecycle management
- [ ] Create unified facade interface
- [ ] Add plugin system for custom services

## 🎯 Usage Patterns

### Facade Pattern (Planned)
```python
# Simple unified interface
trading_bot = CTraderBot()
trading_bot.get_account_info()
trading_bot.fetch_market_data("EURUSD", timeframe="H1")
trading_bot.place_order("EURUSD", "BUY", volume=0.1)
```

### Factory Pattern (Planned)
```python
# Services created on demand
market_data = ServiceFactory.create_market_data_service()
trading = ServiceFactory.create_trading_service()
```

### Event-Driven Pattern
```python
# Subscribe to service events
symbols_service.on_symbols_updated(callback)
market_data_service.on_new_candle(callback)
trading_service.on_order_filled(callback)
```

## 📝 Configuration

### Environment-based Configuration
- **Demo Environment**: Safe testing environment
- **Live Environment**: Real trading environment
- **Service-specific configs**: Each service can have custom settings

### Error Handling Strategy
- **Service-level exceptions**: Each service defines custom exception types
- **Global error handler**: Catches and routes errors appropriately
- **Retry mechanisms**: Built into services for transient failures

## 🔌 Extension Points

### Plugin System (Planned)
- **Custom Services**: Add your own trading services
- **Hook System**: Pre/post operation hooks for custom logic
- **Strategy Pattern**: Different algorithms for same operations

### Integration Options
- **Standalone Usage**: Use individual services independently
- **Full Bot**: Complete trading bot with all services
- **Custom Integration**: Pick and choose services for your needs

## 📊 Data Storage

### File Structure
```
data/
├── symbols_service/
│   ├── symbols.json
│   └── categories.json
├── market_data_service/
│   ├── candles_EURUSD_H1.json
│   └── ticks_EURUSD.json
├── account_service/
│   └── account_info.json
└── trading_service/
    ├── orders.json
    └── trade_history.json
```

### Cache Management
- **Memory Cache**: Fast access to frequently used data
- **Persistent Cache**: JSON files for data persistence
- **Cache Invalidation**: Automatic cache updates

## 🚨 Important Notes

### Security
- **Environment Variables**: Keep API credentials in `.env` file
- **Never Commit Secrets**: Add `.env` to `.gitignore`
- **Demo First**: Always test with demo environment

### Performance
- **Async Operations**: Non-blocking API calls
- **Data Caching**: Reduce API calls with intelligent caching
- **Connection Pooling**: Efficient connection management

### Error Handling
- **Graceful Degradation**: Services continue working if others fail
- **Comprehensive Logging**: Detailed logs for debugging
- **Retry Logic**: Automatic retry for transient failures

## 🤝 Contributing

When adding new services:

1. **Inherit from BaseService**: Use the common base class
2. **Register Message Handlers**: Implement `get_message_handlers()`
3. **Follow Naming Conventions**: Use consistent naming patterns
4. **Add Documentation**: Document your service thoroughly
5. **Include Tests**: Add comprehensive tests

## 📄 License

This module is part of the trading bot project. Please refer to the main project license.

---

**Note**: This is a modular architecture designed for scalability and maintainability. Start with the core services and gradually add more functionality as needed.