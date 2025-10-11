"""
Position Service for cTrader API

This service handles position management operations including fetching positions,
monitoring position status, and managing position-related data.
"""

from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
from ctrader_open_api.messages import OpenApiModelMessages_pb2 as model_msgs
from ctrader_open_api import Protobuf
from ctrader.core.base_service import BaseService
from ctrader.models.account import Position, AccountModelFactory, TradeSide
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import uuid


class PositionService(BaseService):
    """Service for managing position operations"""
    
    def __init__(self, connection_manager, message_router):
        """
        Initialize Position Service
        
        Args:
            connection_manager: Connection manager instance
            message_router: Message router instance
        """
        super().__init__(connection_manager, message_router, "position")
        self.positions: Dict[str, Position] = {}
        self.position_callbacks: Dict[str, Callable] = {}
        self.last_position_update = None
        
    def initialize(self):
        """Initialize the position service"""
        self.register_message_handlers()
        self.is_initialized = True
        self.log("Position service initialized")
        
        # Request initial positions
        self.refresh_positions()
    
    def get_message_handlers(self) -> Dict[int, callable]:
        """Return message handlers for this service"""
        return {
            2126: self.on_execution_event,           # ProtoOAExecutionEvent - position updates
            2121: self.on_position_list_response,    # ProtoOAPositionListRes
            2140: self.on_position_unrealized_pnl,   # ProtoOAGetPositionUnrealizedPnLRes
        }
    
    def refresh_positions(self) -> bool:
        """
        Refresh positions from server
        
        Returns:
            bool: True if request was sent successfully
        """
        if not self.connection_manager.is_ready:
            self.log("Connection not ready. Cannot refresh positions.", "ERROR")
            return False
            
        try:
            # Create position list request
            request = api_msgs.ProtoOAPositionListReq()
            request.ctidTraderAccountId = int(self.connection_manager.account_id)
            
            self.log("Requesting position list from server")
            
            deferred = self.connection_manager.client.send(request)
            deferred.addErrback(self.on_error)
            
            return True
            
        except Exception as e:
            self.log(f"Error requesting positions: {e}", "ERROR")
            return False
    
    def get_positions(self, symbol_id: Optional[str] = None) -> List[Position]:
        """
        Get list of positions, optionally filtered by symbol
        
        Args:
            symbol_id (str, optional): Filter by symbol
            
        Returns:
            List[Position]: List of positions
        """
        if symbol_id:
            return [pos for pos in self.positions.values() if pos.symbolId == symbol_id]
        return list(self.positions.values())
    
    def get_open_positions(self, symbol_id: Optional[str] = None) -> List[Position]:
        """
        Get list of open positions, optionally filtered by symbol
        
        Args:
            symbol_id (str, optional): Filter by symbol
            
        Returns:
            List[Position]: List of open positions
        """
        open_positions = [pos for pos in self.positions.values() if pos.is_open]
        if symbol_id:
            return [pos for pos in open_positions if pos.symbolId == symbol_id]
        return open_positions
    
    def get_position_by_id(self, position_id: str) -> Optional[Position]:
        """
        Get position by ID
        
        Args:
            position_id (str): Position ID
            
        Returns:
            Optional[Position]: Position if found, None otherwise
        """
        return self.positions.get(position_id)
    
    def close_position(self, position_id: str, volume: Optional[float] = None, 
                      callback: Optional[Callable] = None) -> bool:
        """
        Close a position (full or partial)
        
        Args:
            position_id (str): Position ID to close
            volume (float, optional): Volume to close (if None, closes full position)
            callback (callable, optional): Callback function for result
            
        Returns:
            bool: True if request was sent successfully
        """
        if not self.connection_manager.is_ready:
            self.log("Connection not ready. Cannot close position.", "ERROR")
            return False
        
        position = self.get_position_by_id(position_id)
        if not position:
            self.log(f"Position {position_id} not found", "ERROR")
            return False
        
        try:
            # Use full volume if not specified
            close_volume = volume if volume is not None else position.tradeData.volume
            
            # Create close position request (market order in opposite direction)
            request = api_msgs.ProtoOANewOrderReq()
            request.ctidTraderAccountId = int(self.connection_manager.account_id)
            request.symbolId = int(position.tradeData.symbolId)
            request.orderType = model_msgs.MARKET
            request.tradeSide = model_msgs.SELL if position.tradeData.tradeSide == TradeSide.BUY else model_msgs.BUY
            request.volume = int(close_volume * 100)  # Convert to cents
            request.positionId = int(position_id)
            request.comment = f"Close position {position_id}"
            
            if callback:
                request_id = str(uuid.uuid4())
                self.position_callbacks[request_id] = callback
            
            self.log(f"Closing position {position_id}: {close_volume} volume")
            
            deferred = self.connection_manager.client.send(request)
            deferred.addErrback(self.on_error)
            
            return True
            
        except Exception as e:
            self.log(f"Error closing position: {e}", "ERROR")
            return False
    
    def on_execution_event(self, client, message):
        """Handle execution events that may contain position updates"""
        try:
            data = Protobuf.extract(message)
            
            # Check if this event contains position information
            if hasattr(message, 'position') and message.position:
                position_data = Protobuf.extract(message.position)
                position = AccountModelFactory.create_position(position_data)
                
                # Update or add position
                old_position = self.positions.get(position.positionId)
                self.positions[position.positionId] = position
                
                self.log(f"Position update: {position.positionId} - {position.positionStatus.name}")
                
                # If position was closed, remove from active positions
                if position.is_closed and old_position and old_position.is_open:
                    self.log(f"Position {position.positionId} closed")
                
                # Save positions data
                self._save_positions()
                
        except Exception as e:
            self.log(f"Error handling execution event for positions: {e}", "ERROR")
    
    def on_position_list_response(self, client, message):
        """Handle position list responses"""
        try:
            data = Protobuf.extract(message)
            self.log(f"Position list response received")
            
            # Clear existing positions
            self.positions.clear()
            
            # Process position list
            if hasattr(data, 'position') and data.position:
                positions_data = data.position if isinstance(data.position, list) else [data.position]
                
                for pos_data in positions_data:
                    try:
                        position = AccountModelFactory.create_position(pos_data)
                        self.positions[position.positionId] = position
                        self.log(f"Loaded position: {position.positionId} - {position.tradeData.symbolId}")
                    except Exception as e:
                        self.log(f"Error parsing position data: {e}", "ERROR")
            
            self.log(f"Loaded {len(self.positions)} positions from server")
            self.last_position_update = datetime.now()
            
            # Save positions data
            self._save_positions()
            
        except Exception as e:
            self.log(f"Error handling position list response: {e}", "ERROR")
    
    def on_position_unrealized_pnl(self, client, message):
        """Handle position unrealized PnL responses"""
        try:
            data = Protobuf.extract(message)
            self.log(f"Position unrealized PnL: {data}")
            
            # Update position P&L if we have the position
            position_id = str(data.get('positionId', ''))
            if position_id in self.positions:
                position = self.positions[position_id]
                position.pnl = data.get('unrealizedPnL')
                position.grossPnl = data.get('grossUnrealizedPnL')
                self.log(f"Updated P&L for position {position_id}: {position.pnl}")
            
        except Exception as e:
            self.log(f"Error handling position PnL response: {e}", "ERROR")
    
    def _save_positions(self):
        """Save positions to file"""
        try:
            positions_data = {}
            for pos_id, position in self.positions.items():
                positions_data[pos_id] = position.to_dict()
            
            self.save_data("positions", positions_data)
            
        except Exception as e:
            self.log(f"Error saving positions: {e}", "ERROR")
    
    def get_position_count(self) -> int:
        """Get total number of positions"""
        return len(self.positions)
    
    def get_open_position_count(self) -> int:
        """Get number of open positions"""
        return len([pos for pos in self.positions.values() if pos.is_open])
    
    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all open positions"""
        return sum(pos.unrealized_pnl for pos in self.positions.values() if pos.is_open)