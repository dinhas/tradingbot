"""
Symbols Service for cTrader API

This service handles fetching and managing trading symbols from cTrader API.
It inherits from BaseService and provides symbol management functionality.
"""

from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
from ctrader_open_api import Protobuf
from ctrader.core.base_service import BaseService
from twisted.internet import reactor
from datetime import datetime
from typing import Dict, List, Optional, Any


class SymbolsService(BaseService):
    """Service for managing trading symbols from cTrader API"""
    
    def __init__(self, connection_manager, message_router):
        """
        Initialize Symbols Service
        
        Args:
            connection_manager: Connection manager instance
            message_router: Message router instance
        """
        super().__init__(connection_manager, message_router, "symbols")
        self.symbols_data: Optional[Dict[str, Any]] = None
        
    def initialize(self):
        """Initialize the symbols service"""
        self.register_message_handlers()
        self.symbols_data = self.load_data("symbols")
        self.is_initialized = True
        self.log("Symbols service initialized")
        
        # Load existing symbols if available
        if self.symbols_data:
            self.log(f"Loaded {self.symbols_data.get('totalSymbols', 0)} symbols from cache")
    
    def get_message_handlers(self) -> Dict[int, callable]:
        """Return message handlers for this service"""
        return {
            2127: self.on_symbols_received  # ProtoOASymbolsListRes
        }
    
    def fetch_symbols(self) -> bool:
        """
        Request symbols list from cTrader API
        
        Returns:
            bool: True if request was sent successfully
        """
        if not self.connection_manager.is_ready:
            self.log("Connection not ready. Cannot fetch symbols.", "ERROR")
            return False
            
        try:
            self.log("Requesting symbols from cTrader API...")
            request = api_msgs.ProtoOASymbolsListReq()
            request.ctidTraderAccountId = int(self.account_id)
            
            deferred = self.client.send(request)
            deferred.addErrback(self.on_error)
            return True
            
        except Exception as e:
            self.log(f"Error requesting symbols: {e}", "ERROR")
            return False
    
    def on_symbols_received(self, client, message: api_msgs.ProtoOASymbolsListRes):
        """
        Callback function when symbols list is received from cTrader API
        
        Args:
            client: cTrader client instance
            message: ProtoOASymbolsListRes message containing symbols
        """
        self.log("Received symbols list response from cTrader API.")
        try:
            # Convert the protobuf message to a list of dictionaries
            symbols_list = []
            symbol_count = len(message.symbol)
            
            self.log(f"Received {symbol_count} symbols from cTrader API")
            print("=" * 60)
            
            for symbol in message.symbol:
                symbol_data = {
                    "symbolId": str(symbol.symbolId),
                    "symbolName": symbol.symbolName,
                    "enabled": symbol.enabled,
                    "baseAssetId": str(symbol.baseAssetId),
                    "quoteAssetId": str(symbol.quoteAssetId),
                    "symbolCategoryId": str(symbol.symbolCategoryId),
                    "description": symbol.description,
                }
                symbols_list.append(symbol_data)
                
                # Print each symbol to terminal
                print("symbol {")
                print(f"  symbolId: {symbol.symbolId}")
                print(f"  symbolName: \"{symbol.symbolName}\"")
                print(f"  enabled: {str(symbol.enabled).lower()}")
                print(f"  baseAssetId: {symbol.baseAssetId}")
                print(f"  quoteAssetId: {symbol.quoteAssetId}")
                print(f"  symbolCategoryId: {symbol.symbolCategoryId}")
                print(f"  description: \"{symbol.description}\"")
                print("}")

            # Create structured data
            self.symbols_data = {
                "ctidTraderAccountId": str(message.ctidTraderAccountId),
                "lastUpdated": datetime.now().isoformat(),
                "totalSymbols": len(symbols_list),
                "enabledSymbols": sum(1 for s in symbols_list if s['enabled']),
                "disabledSymbols": sum(1 for s in symbols_list if not s['enabled']),
                "symbol": symbols_list
            }

            # Save to file and cache
            self.save_data("symbols", self.symbols_data)
            self.cache_data("symbols", self.symbols_data)
            
            print("=" * 60)
            self.log(f"Successfully saved {len(symbols_list)} symbols to data/symbols/symbols.json")
            self.log(f"Enabled: {self.symbols_data['enabledSymbols']}, Disabled: {self.symbols_data['disabledSymbols']}")
            
            # Stop reactor if running for standalone usage
            if reactor.running:
                reactor.callLater(1, reactor.stop)
                
        except Exception as e:
            self.log(f"Error processing symbols: {e}", "ERROR")
    
    def get_symbols(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get all symbols
        
        Returns:
            List of symbol dictionaries or None if no symbols loaded
        """
        if not self.symbols_data:
            self.symbols_data = self.load_data("symbols")
            
        return self.symbols_data.get('symbol', []) if self.symbols_data else None
    
    def get_enabled_symbols(self) -> List[Dict[str, Any]]:
        """
        Get list of enabled symbols
        
        Returns:
            List of enabled symbol dictionaries
        """
        symbols = self.get_symbols()
        if symbols:
            return [s for s in symbols if s.get('enabled', False)]
        return []
    
    def get_symbol_by_name(self, symbol_name: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol data by symbol name
        
        Args:
            symbol_name (str): Symbol name to search for
            
        Returns:
            Symbol data dictionary or None if not found
        """
        symbols = self.get_symbols()
        if symbols:
            for symbol in symbols:
                if symbol.get('symbolName') == symbol_name:
                    return symbol
        return None
    
    def get_symbol_by_id(self, symbol_id: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol data by symbol ID
        
        Args:
            symbol_id (str): Symbol ID to search for
            
        Returns:
            Symbol data dictionary or None if not found
        """
        symbols = self.get_symbols()
        if symbols:
            for symbol in symbols:
                if symbol.get('symbolId') == str(symbol_id):
                    return symbol
        return None
    
    def get_symbols_by_category(self, category_id: str) -> List[Dict[str, Any]]:
        """
        Get symbols by category ID
        
        Args:
            category_id (str): Symbol category ID
            
        Returns:
            List of symbols in the category
        """
        symbols = self.get_symbols()
        if symbols:
            return [s for s in symbols if s.get('symbolCategoryId') == str(category_id)]
        return []
    
    def search_symbols(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search symbols by name or description
        
        Args:
            search_term (str): Term to search for
            
        Returns:
            List of matching symbols
        """
        symbols = self.get_symbols()
        if symbols:
            search_term = search_term.upper()
            return [s for s in symbols 
                   if search_term in s.get('symbolName', '').upper() 
                   or search_term in s.get('description', '').upper()]
        return []
    
    def get_symbols_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get symbols summary information
        
        Returns:
            Summary dictionary with counts and metadata
        """
        if not self.symbols_data:
            self.symbols_data = self.load_data("symbols")
            
        if self.symbols_data:
            return {
                "accountId": self.symbols_data.get('ctidTraderAccountId'),
                "lastUpdated": self.symbols_data.get('lastUpdated'),
                "totalSymbols": self.symbols_data.get('totalSymbols', 0),
                "enabledSymbols": self.symbols_data.get('enabledSymbols', 0),
                "disabledSymbols": self.symbols_data.get('disabledSymbols', 0)
            }
        return None
    
    def print_symbols_summary(self):
        """Print a summary of loaded symbols"""
        summary = self.get_symbols_summary()
        if summary:
            print(f"\n📊 Symbols Summary:")
            print(f"   Account ID: {summary.get('accountId', 'N/A')}")
            print(f"   Last Updated: {summary.get('lastUpdated', 'N/A')}")
            print(f"   Total Symbols: {summary.get('totalSymbols', 0)}")
            print(f"   Enabled: {summary.get('enabledSymbols', 0)}")
            print(f"   Disabled: {summary.get('disabledSymbols', 0)}")
        else:
            print("📄 No symbols data available. Fetch symbols first.")
    
    def is_symbol_enabled(self, symbol_name: str) -> bool:
        """
        Check if a symbol is enabled for trading
        
        Args:
            symbol_name (str): Symbol name to check
            
        Returns:
            bool: True if symbol is enabled, False otherwise
        """
        symbol = self.get_symbol_by_name(symbol_name)
        return symbol.get('enabled', False) if symbol else False