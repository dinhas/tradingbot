"""
Account Service for cTrader API

This service handles account-related operations including account information,
balance, positions, margin calculations, and account permissions.
"""

from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
from ctrader_open_api import Protobuf
from ctrader.core.base_service import BaseService
from datetime import datetime
from typing import Dict, List, Optional, Any


class AccountService(BaseService):
    """Service for managing account information and operations"""
    
    def __init__(self, connection_manager, message_router):
        """
        Initialize Account Service
        
        Args:
            connection_manager: Connection manager instance
            message_router: Message router instance
        """
        super().__init__(connection_manager, message_router, "account")
        self.account_info: Optional[Dict[str, Any]] = None
        self.positions: List[Dict[str, Any]] = []
        self.assets: Dict[str, Dict[str, Any]] = {}
        
    def initialize(self):
        """Initialize the account service"""
        self.register_message_handlers()
        self.account_info = self.load_data("account_info")
        self.assets = self.load_data("assets") or {}
        self.is_initialized = True
        self.log("Account service initialized")
        
        # Load existing data if available
        if self.account_info:
            self.log(f"Loaded cached account info for account {self.account_info.get('login')}")
    
    def get_message_handlers(self) -> Dict[int, callable]:
        """Return message handlers for this service"""
        return {
            2103: self.on_account_info_received,      # ProtoOAAccountAuthRes (contains some account info)
            2122: self.on_trader_info_received,       # ProtoOATraderRes  
            2113: self.on_assets_list_received,       # ProtoOAAssetListRes
            2125: self.on_positions_received,         # ProtoOAReconcileRes
            2141: self.on_account_logout,             # ProtoOAAccountLogoutRes
        }
    
    def fetch_account_info(self) -> bool:
        """
        Request complete account information from cTrader API
        
        Returns:
            bool: True if requests were sent successfully
        """
        if not self.connection_manager.is_ready:
            self.log("Connection not ready. Cannot fetch account info.", "ERROR")
            return False
            
        try:
            self.log("Requesting account information from cTrader API...")
            
            # Request trader info (main account details)
            trader_request = api_msgs.ProtoOATraderReq()
            trader_request.ctidTraderAccountId = int(self.account_id)
            
            trader_deferred = self.client.send(trader_request)
            trader_deferred.addErrback(self.on_error)
            
            # Request assets list
            assets_request = api_msgs.ProtoOAAssetListReq()
            assets_request.ctidTraderAccountId = int(self.account_id)
            
            assets_deferred = self.client.send(assets_request)
            assets_deferred.addErrback(self.on_error)
            
            # Request reconcile (positions and orders)
            reconcile_request = api_msgs.ProtoOAReconcileReq()
            reconcile_request.ctidTraderAccountId = int(self.account_id)
            
            reconcile_deferred = self.client.send(reconcile_request)
            reconcile_deferred.addErrback(self.on_error)
            
            return True
            
        except Exception as e:
            self.log(f"Error requesting account info: {e}", "ERROR")
            return False
    
    def on_account_info_received(self, client, message):
        """
        Handle basic account authentication response (contains some account info)
        
        Args:
            client: cTrader client instance
            message: ProtoOAAccountAuthRes message
        """
        try:
            # This is called during authentication - extract basic account info
            account_id = str(message.ctidTraderAccountId)
            
            basic_info = {
                "ctidTraderAccountId": account_id,
                "lastAuthTime": datetime.now().isoformat(),
                "authStatus": "authenticated"
            }
            
            if not self.account_info:
                self.account_info = basic_info
            else:
                self.account_info.update(basic_info)
            
            self.log(f"Account {account_id} authentication confirmed")
            
        except Exception as e:
            self.log(f"Error processing account auth info: {e}", "ERROR")
    
    def on_trader_info_received(self, client, message):
        """
        Handle trader information response (main account details)
        
        Args:
            client: cTrader client instance
            message: ProtoOATraderRes message containing trader info
        """
        try:
            # Extract the protobuf message content
            trader_res = Protobuf.extract(message)
            trader = trader_res.trader
            
            # Extract comprehensive account information
            account_data = {
                "ctidTraderAccountId": str(trader.ctidTraderAccountId),
                "login": str(trader.traderLogin),
                "balance": trader.balance,
                "balanceVersion": str(trader.balanceVersion),
                "leverageInCents": trader.leverageInCents,
                "totalMarginCalculationType": getattr(trader, 'totalMarginCalculationType', 0),
                "maxLeverage": trader.maxLeverage,
                "brokerName": trader.brokerName,
                "accountType": self._get_account_type_name(trader.accountType),
                "lastUpdated": datetime.now().isoformat(),
                "depositAssetId": str(trader.depositAssetId),
                "swapFree": trader.swapFree,
                "moneyDigits": trader.moneyDigits,
                "isLimitedRisk": trader.isLimitedRisk,
                "registrationTimestamp": str(trader.registrationTimestamp)
            }
            
            # Add optional fields if present
            if hasattr(trader, 'nonWithdrawableBonus'):
                account_data["nonWithdrawableBonus"] = trader.nonWithdrawableBonus
            if hasattr(trader, 'accessRights'):
                account_data["accessRights"] = str(trader.accessRights)
            if hasattr(trader, 'frenchRisk'):
                account_data["frenchRisk"] = trader.frenchRisk
            
            # For demo accounts, some fields might not be present
            # Set default values for missing trading-related fields
            account_data.update({
                "equity": trader.balance,  # For demo, equity equals balance initially
                "margin": 0.0,  # No open positions initially
                "marginLevel": 0.0 if trader.balance == 0 else float('inf'),
                "freeMargin": trader.balance,
                "credit": 0.0,
                "swap": 0.0
            })
            
            self.account_info = account_data
            
            # Save to file and cache
            self.save_data("account_info", self.account_info)
            self.cache_data("account_info", self.account_info)
            
            self.log("Account information updated successfully")
            self._print_account_summary()
            
        except Exception as e:
            self.log(f"Error processing trader info: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    def on_assets_list_received(self, client, message):
        """
        Handle assets list response (currency and asset information)
        
        Args:
            client: cTrader client instance
            message: ProtoOAAssetListRes message containing assets
        """
        try:
            # Extract the protobuf message content
            assets_res = Protobuf.extract(message)
            assets_data = {}
            
            self.log(f"Received {len(assets_res.asset)} assets from cTrader API")
            
            for asset in assets_res.asset:
                asset_info = {
                    "assetId": str(asset.assetId),
                    "name": asset.name,
                    "displayName": asset.displayName,
                    "digits": asset.digits,
                }
                
                # Add optional fields
                if hasattr(asset, 'country'):
                    asset_info["country"] = asset.country
                if hasattr(asset, 'assetClass'):
                    asset_info["assetClass"] = asset.assetClass
                
                assets_data[str(asset.assetId)] = asset_info
            
            self.assets = assets_data
            
            # Save to file and cache
            self.save_data("assets", self.assets)
            self.cache_data("assets", self.assets)
            
            self.log(f"Assets information updated - {len(self.assets)} assets loaded")
            
        except Exception as e:
            self.log(f"Error processing assets list: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    def on_positions_received(self, client, message):
        """
        Handle reconcile response (positions and orders)
        
        Args:
            client: cTrader client instance
            message: ProtoOAReconcileRes message containing positions
        """
        try:
            # Extract the protobuf message content
            reconcile_res = Protobuf.extract(message)
            positions_data = []
            
            # Check if message has positions
            if hasattr(reconcile_res, 'position') and reconcile_res.position:
                position_count = len(reconcile_res.position)
                self.log(f"Received {position_count} positions from cTrader API")
                
                for position in reconcile_res.position:
                    position_info = {
                        "positionId": str(position.positionId),
                        "tradeData": {
                            "symbolId": str(position.tradeData.symbolId),
                            "volume": position.tradeData.volume,
                            "tradeSide": "BUY" if position.tradeData.tradeSide == 1 else "SELL",
                            "openPrice": position.tradeData.openPrice,
                            "openTimestamp": str(position.tradeData.openTimestamp),
                        },
                        "positionStatus": self._get_position_status_name(position.positionStatus),
                        "swap": position.swap,
                        "price": position.price,
                        "stopLoss": getattr(position, 'stopLoss', None),
                        "takeProfit": getattr(position, 'takeProfit', None),
                        "utcLastUpdateTimestamp": str(position.utcLastUpdateTimestamp),
                        "commission": position.commission,
                        "marginRate": getattr(position, 'marginRate', None),
                        "mirroringCommission": getattr(position, 'mirroringCommission', None),
                    }
                    
                    positions_data.append(position_info)
            else:
                self.log("No positions in reconcile response (account has no open positions)")
            
            self.positions = positions_data
            
            # Save to file and cache
            self.save_data("positions", self.positions)
            self.cache_data("positions", self.positions)
            
            self.log(f"Positions updated - {len(self.positions)} active positions")
            
            if self.positions:
                self._print_positions_summary()
            
        except Exception as e:
            self.log(f"Error processing positions: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    def on_account_logout(self, client, message):
        """Handle account logout response"""
        self.log("Account logged out from cTrader API")
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get complete account information
        
        Returns:
            Dict with account information or None if not available
        """
        if not self.account_info:
            self.account_info = self.load_data("account_info")
        return self.account_info
    
    def get_balance(self) -> float:
        """
        Get current account balance
        
        Returns:
            float: Account balance or 0.0 if not available
        """
        account_info = self.get_account_info()
        return account_info.get("balance", 0.0) if account_info else 0.0
    
    def get_equity(self) -> float:
        """
        Get current account equity
        
        Returns:
            float: Account equity or 0.0 if not available
        """
        account_info = self.get_account_info()
        return account_info.get("equity", 0.0) if account_info else 0.0
    
    def get_margin(self) -> float:
        """
        Get used margin
        
        Returns:
            float: Used margin or 0.0 if not available
        """
        account_info = self.get_account_info()
        return account_info.get("margin", 0.0) if account_info else 0.0
    
    def get_free_margin(self) -> float:
        """
        Get free margin
        
        Returns:
            float: Free margin or 0.0 if not available
        """
        account_info = self.get_account_info()
        return account_info.get("freeMargin", 0.0) if account_info else 0.0
    
    def get_margin_level(self) -> float:
        """
        Get margin level percentage
        
        Returns:
            float: Margin level percentage or 0.0 if not available
        """
        account_info = self.get_account_info()
        return account_info.get("marginLevel", 0.0) if account_info else 0.0
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current positions
        
        Returns:
            List of position dictionaries
        """
        if not self.positions:
            cached_positions = self.load_data("positions")
            if cached_positions:
                self.positions = cached_positions
        return self.positions
    
    def get_position_by_id(self, position_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific position by ID
        
        Args:
            position_id (str): Position ID to search for
            
        Returns:
            Position dictionary or None if not found
        """
        positions = self.get_positions()
        for position in positions:
            if position.get("positionId") == str(position_id):
                return position
        return None
    
    def get_positions_by_symbol(self, symbol_id: str) -> List[Dict[str, Any]]:
        """
        Get positions for a specific symbol
        
        Args:
            symbol_id (str): Symbol ID to filter by
            
        Returns:
            List of positions for the symbol
        """
        positions = self.get_positions()
        return [p for p in positions if p.get("tradeData", {}).get("symbolId") == str(symbol_id)]
    
    def get_assets(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all assets information
        
        Returns:
            Dictionary of assets keyed by asset ID
        """
        if not self.assets:
            cached_assets = self.load_data("assets")
            if cached_assets:
                self.assets = cached_assets
        return self.assets
    
    def get_asset_by_id(self, asset_id: str) -> Optional[Dict[str, Any]]:
        """
        Get asset information by ID
        
        Args:
            asset_id (str): Asset ID to search for
            
        Returns:
            Asset dictionary or None if not found
        """
        assets = self.get_assets()
        return assets.get(str(asset_id))
    
    def get_asset_by_name(self, asset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get asset information by name
        
        Args:
            asset_name (str): Asset name to search for
            
        Returns:
            Asset dictionary or None if not found
        """
        assets = self.get_assets()
        for asset in assets.values():
            if asset.get("name", "").upper() == asset_name.upper():
                return asset
        return None
    
    def is_margin_sufficient(self, required_margin: float) -> bool:
        """
        Check if there's sufficient free margin for a trade
        
        Args:
            required_margin (float): Required margin for the trade
            
        Returns:
            bool: True if sufficient margin is available
        """
        free_margin = self.get_free_margin()
        return free_margin >= required_margin
    
    def get_account_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get a summary of account information
        
        Returns:
            Dictionary with key account metrics
        """
        account_info = self.get_account_info()
        if account_info:
            return {
                "login": account_info.get("login"),
                "balance": account_info.get("balance", 0.0),
                "equity": account_info.get("equity", 0.0),
                "margin": account_info.get("margin", 0.0),
                "freeMargin": account_info.get("freeMargin", 0.0),
                "marginLevel": account_info.get("marginLevel", 0.0),
                "positionsCount": len(self.get_positions()),
                "brokerName": account_info.get("brokerName"),
                "accountType": account_info.get("accountType"),
                "lastUpdated": account_info.get("lastUpdated")
            }
        return None
    
    def print_account_summary(self):
        """Print a formatted account summary"""
        summary = self.get_account_summary()
        if summary:
            print(f"\n💰 Account Summary:")
            print(f"   Login: {summary.get('login', 'N/A')}")
            print(f"   Broker: {summary.get('brokerName', 'N/A')}")
            print(f"   Account Type: {summary.get('accountType', 'N/A')}")
            print(f"   Balance: ${summary.get('balance', 0):,.2f}")
            print(f"   Equity: ${summary.get('equity', 0):,.2f}")
            print(f"   Used Margin: ${summary.get('margin', 0):,.2f}")
            print(f"   Free Margin: ${summary.get('freeMargin', 0):,.2f}")
            print(f"   Margin Level: {summary.get('marginLevel', 0):.2f}%")
            print(f"   Active Positions: {summary.get('positionsCount', 0)}")
            print(f"   Last Updated: {summary.get('lastUpdated', 'N/A')}")
        else:
            print("📄 No account information available. Fetch account info first.")
    
    def _print_account_summary(self):
        """Internal method to print account summary"""
        self.print_account_summary()
    
    def _print_positions_summary(self):
        """Print a summary of current positions"""
        positions = self.get_positions()
        if positions:
            print(f"\n📈 Active Positions ({len(positions)}):")
            print("-" * 80)
            
            total_unrealized_pnl = 0
            for i, position in enumerate(positions[:10], 1):  # Show first 10 positions
                trade_data = position.get("tradeData", {})
                symbol_id = trade_data.get("symbolId", "Unknown")
                volume = trade_data.get("volume", 0)
                side = trade_data.get("tradeSide", "Unknown")
                open_price = trade_data.get("openPrice", 0)
                current_price = position.get("price", 0)
                swap = position.get("swap", 0)
                commission = position.get("commission", 0)
                
                # Calculate unrealized P&L (simplified)
                if side == "BUY":
                    unrealized_pnl = (current_price - open_price) * volume
                else:
                    unrealized_pnl = (open_price - current_price) * volume
                
                total_unrealized_pnl += unrealized_pnl
                
                print(f"   {i}. Symbol: {symbol_id} | {side} {volume:,.0f} @ {open_price}")
                print(f"      Current Price: {current_price} | P&L: ${unrealized_pnl:,.2f}")
                print(f"      Swap: ${swap:,.2f} | Commission: ${commission:,.2f}")
                print()
            
            if len(positions) > 10:
                print(f"   ... and {len(positions) - 10} more positions")
            
            print(f"   📊 Total Unrealized P&L: ${total_unrealized_pnl:,.2f}")
    
    def _get_account_type_name(self, account_type: int) -> str:
        """Convert account type number to name"""
        account_types = {
            0: "HEDGED",
            1: "NETTED",
            2: "SPREAD_BETTING"
        }
        return account_types.get(account_type, f"UNKNOWN_{account_type}")
    
    def _get_position_status_name(self, status: int) -> str:
        """Convert position status number to name"""
        status_names = {
            1: "POSITION_STATUS_OPEN",
            2: "POSITION_STATUS_CLOSED",
            3: "POSITION_STATUS_ERROR"
        }
        return status_names.get(status, f"UNKNOWN_STATUS_{status}")