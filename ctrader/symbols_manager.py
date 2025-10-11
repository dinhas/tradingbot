"""
cTrader Symbols Manager Module

This module handles fetching and managing trading symbols from cTrader API.
It saves symbols to a JSON file in the project root and provides utilities
for symbol management.
"""

from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
from ctrader_open_api import Protobuf
import json
import os
from twisted.internet import reactor
from datetime import datetime


class SymbolsManager:
    def __init__(self, project_root=None):
        """
        Initialize the SymbolsManager
        
        Args:
            project_root (str): Path to project root. If None, auto-detects.
        """
        if project_root is None:
            # Auto-detect project root (parent of ctrader folder)
            self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            self.project_root = project_root
            
        self.symbols_file = os.path.join(self.project_root, 'symbols.json')
        self.symbols_data = None
        self.client = None
        
    def on_symbols_received(self, client, message: api_msgs.ProtoOASymbolsListRes):
        """
        Callback function when symbols list is received from cTrader API
        
        Args:
            client: cTrader client instance
            message: ProtoOASymbolsListRes message containing symbols
        """
        self.client = client
        
        # Convert the protobuf message to a list of dictionaries
        symbols_list = []
        symbol_count = len(message.symbol)
        
        print(f"\n🔄 Received {symbol_count} symbols from cTrader API")
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
            
            # Print each symbol to terminal (as requested)
            print("symbol {")
            print(f"  symbolId: {symbol.symbolId}")
            print(f"  symbolName: \"{symbol.symbolName}\"")
            print(f"  enabled: {str(symbol.enabled).lower()}")
            print(f"  baseAssetId: {symbol.baseAssetId}")
            print(f"  quoteAssetId: {symbol.quoteAssetId}")
            print(f"  symbolCategoryId: {symbol.symbolCategoryId}")
            print(f"  description: \"{symbol.description}\"")
            print("}")

        # Create structured JSON data
        self.symbols_data = {
            "ctidTraderAccountId": str(message.ctidTraderAccountId),
            "lastUpdated": datetime.now().isoformat(),
            "totalSymbols": len(symbols_list),
            "enabledSymbols": sum(1 for s in symbols_list if s['enabled']),
            "disabledSymbols": sum(1 for s in symbols_list if not s['enabled']),
            "symbol": symbols_list
        }

        # Save to JSON file (overwrites existing data)
        self._save_symbols_to_file()
        
        print("=" * 60)
        print(f"✅ {len(symbols_list)} symbols successfully saved to symbols.json")
        print(f"📁 File location: {os.path.abspath(self.symbols_file)}")
        print(f"📊 Enabled: {self.symbols_data['enabledSymbols']}, Disabled: {self.symbols_data['disabledSymbols']}")
        
        if reactor.running:
            reactor.stop()
    
    def _save_symbols_to_file(self):
        """
        Save symbols data to JSON file, overwriting existing content
        """
        try:
            # Check if file exists and remove content (overwrite)
            if os.path.exists(self.symbols_file):
                print(f"📝 Existing symbols.json found - overwriting with new data...")
            
            with open(self.symbols_file, "w", encoding='utf-8') as f:
                json.dump(self.symbols_data, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Error saving symbols to file: {e}")
            raise
    
    def load_symbols_from_file(self):
        """
        Load symbols from existing JSON file
        
        Returns:
            dict: Symbols data or None if file doesn't exist
        """
        try:
            if os.path.exists(self.symbols_file):
                with open(self.symbols_file, 'r', encoding='utf-8') as f:
                    self.symbols_data = json.load(f)
                return self.symbols_data
            else:
                print(f"📄 No existing symbols.json file found at {self.symbols_file}")
                return None
        except Exception as e:
            print(f"❌ Error loading symbols from file: {e}")
            return None
    
    def get_enabled_symbols(self):
        """
        Get list of enabled symbols
        
        Returns:
            list: List of enabled symbol dictionaries
        """
        if not self.symbols_data:
            self.load_symbols_from_file()
            
        if self.symbols_data:
            return [s for s in self.symbols_data.get('symbol', []) if s.get('enabled', False)]
        return []
    
    def get_symbol_by_name(self, symbol_name):
        """
        Get symbol data by symbol name
        
        Args:
            symbol_name (str): Symbol name to search for
            
        Returns:
            dict: Symbol data or None if not found
        """
        if not self.symbols_data:
            self.load_symbols_from_file()
            
        if self.symbols_data:
            for symbol in self.symbols_data.get('symbol', []):
                if symbol.get('symbolName') == symbol_name:
                    return symbol
        return None
    
    def get_symbols_by_category(self, category_id):
        """
        Get symbols by category ID
        
        Args:
            category_id (str or int): Symbol category ID
            
        Returns:
            list: List of symbols in the category
        """
        if not self.symbols_data:
            self.load_symbols_from_file()
            
        if self.symbols_data:
            return [s for s in self.symbols_data.get('symbol', []) 
                   if s.get('symbolCategoryId') == str(category_id)]
        return []
    
    def print_symbols_summary(self):
        """
        Print a summary of loaded symbols
        """
        if not self.symbols_data:
            self.load_symbols_from_file()
            
        if self.symbols_data:
            print(f"\n📊 Symbols Summary:")
            print(f"   Account ID: {self.symbols_data.get('ctidTraderAccountId', 'N/A')}")
            print(f"   Last Updated: {self.symbols_data.get('lastUpdated', 'N/A')}")
            print(f"   Total Symbols: {self.symbols_data.get('totalSymbols', 0)}")
            print(f"   Enabled: {self.symbols_data.get('enabledSymbols', 0)}")
            print(f"   Disabled: {self.symbols_data.get('disabledSymbols', 0)}")
        else:
            print("📄 No symbols data available. Run symbol fetch first.")


def get_symbols_from_api(client, ctid_trader_account_id, symbols_manager=None):
    """
    Request symbols list from cTrader API
    
    Args:
        client: cTrader client instance
        ctid_trader_account_id (str): Account ID
        symbols_manager (SymbolsManager): Optional symbols manager instance
    """
    if symbols_manager is None:
        symbols_manager = SymbolsManager()
    
    def on_error(failure):
        print(f"❌ Error fetching symbols: {failure}")
    
    request = api_msgs.ProtoOASymbolsListReq()
    request.ctidTraderAccountId = int(ctid_trader_account_id)
    deferred = client.send(request)
    deferred.addErrback(on_error)
    
    return symbols_manager


# Module-level instance for easy access
_symbols_manager = SymbolsManager()

def get_symbols_manager():
    """Get the module-level symbols manager instance"""
    return _symbols_manager