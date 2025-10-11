"""
cTrader Module - Modular Trading Bot Architecture

This module provides a comprehensive, scalable architecture for cTrader API integration
including market data, trading operations, account management, and more.

Usage:
    from ctrader.core import ConnectionManager, MessageRouter
    from ctrader.services import SymbolsService, AccountService, TradingService
    
    # Initialize core components
    connection_manager = ConnectionManager(use_demo=True)
    message_router = MessageRouter()
    
    # Initialize services
    symbols_service = SymbolsService(connection_manager, message_router)
    account_service = AccountService(connection_manager, message_router)
    trading_service = TradingService(connection_manager, message_router)
    
    # Connect and start
    connection_manager.connect()
"""

from .core import ConnectionManager, MessageRouter, BaseService
from .services import SymbolsService, AccountService, TradingService
from .models import (
    AccountType, AccountInfo, Position, AssetInfo,
    OrderType, TradeSide, OrderRequest, Order, OrderModelFactory
)

__version__ = "2.2.0"
__all__ = [
    'ConnectionManager', 'MessageRouter', 'BaseService',
    'SymbolsService', 'AccountService', 'TradingService',
    'AccountType', 'AccountInfo', 'Position', 'AssetInfo',
    'OrderType', 'TradeSide', 'OrderRequest', 'Order', 'OrderModelFactory'
]