"""
Trading Service for cTrader API

This service handles order placement, modification, and cancellation operations.
It manages the complete order lifecycle including placing orders and then
adding stop loss and take profit levels to resulting positions.
"""

from ctrader_open_api.messages import OpenApiMessages_pb2 as api_msgs
from ctrader_open_api.messages import OpenApiModelMessages_pb2 as model_msgs
from ctrader_open_api import Protobuf
from ctrader.core.base_service import BaseService
from ctrader.models.order import (
    OrderRequest, Order, SLTPModification, OrderModelFactory,
    OrderType, OrderStatus, OrderTriggerMethod
)
from ctrader.models.account import TradeSide
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import uuid


class TradingService(BaseService):
    """Service for managing trading operations"""
    
    def __init__(self, connection_manager, message_router):
        """
        Initialize Trading Service
        
        Args:
            connection_manager: Connection manager instance
            message_router: Message router instance
        """
        super().__init__(connection_manager, message_router, "trading")
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.orders: Dict[str, Order] = {}
        self.pending_sltp_modifications: Dict[str, SLTPModification] = {}
        self.order_callbacks: Dict[str, Callable] = {}
        self.positions_with_sltp: set = set()  # Track positions that already have SL/TP
        
    def initialize(self):
        """Initialize the trading service"""
        self.register_message_handlers()
        saved_orders = self.load_data("orders") or {}
        
        # Convert saved orders back to Order objects
        for order_id, order_data in saved_orders.items():
            try:
                self.orders[order_id] = OrderModelFactory.create_order(order_data)
            except Exception as e:
                self.log(f"Error loading order {order_id}: {e}", "ERROR")
        
        self.is_initialized = True
        self.log("Trading service initialized")
        self.log(f"Loaded {len(self.orders)} orders from cache")
    
    def get_message_handlers(self) -> Dict[int, callable]:
        """Return message handlers for this service"""
        return {
            2101: self.on_execution_event,           # ProtoOAExecutionEvent
            2126: self.on_execution_event,           # ProtoOAExecutionEvent (alternative type)
            2132: self.on_order_error_event,         # ProtoOAOrderErrorEvent  
            2138: self.on_order_list_response,       # ProtoOAOrderListRes
            2140: self.on_position_unrealized_pnl,   # ProtoOAGetPositionUnrealizedPnLRes
            2110: self.on_amend_position_response,   # ProtoOAAmendPositionSLTPRes
            2111: self.on_amend_position_response,   # Alternative amend response type
        }
    
    def place_market_order(self, symbol_id: str, trade_side: TradeSide, volume: float,
                          stop_loss: Optional[float] = None, take_profit: Optional[float] = None,
                          comment: Optional[str] = None, callback: Optional[Callable] = None) -> Optional[str]:
        """
        Place a market order
        
        Args:
            symbol_id (str): Symbol to trade
            trade_side (TradeSide): BUY or SELL
            volume (float): Volume to trade
            stop_loss (float, optional): Stop loss price
            take_profit (float, optional): Take profit price
            comment (str, optional): Order comment
            callback (callable, optional): Callback function for order result
            
        Returns:
            str: Client order ID if successful, None otherwise
        """
        if not self.connection_manager.is_ready:
            self.log("Connection not ready. Cannot place order.", "ERROR")
            return None
            
        try:
            # Generate unique client order ID
            client_order_id = str(uuid.uuid4())
            
            # Create order request
            order_request = OrderModelFactory.create_market_order(
                symbol_id=symbol_id,
                trade_side=trade_side,
                volume=volume,
                comment=comment
            )
            
            # Store for later SLTP modification if needed
            if stop_loss or take_profit:
                self.pending_orders[client_order_id] = order_request
                self.pending_orders[client_order_id].stopLoss = stop_loss
                self.pending_orders[client_order_id].takeProfit = take_profit
            
            # Store callback if provided
            if callback:
                self.order_callbacks[client_order_id] = callback
            
            # Create API request
            request = api_msgs.ProtoOANewOrderReq()
            request.ctidTraderAccountId = int(self.connection_manager.account_id)
            request.symbolId = int(symbol_id)
            request.orderType = model_msgs.MARKET
            request.tradeSide = model_msgs.BUY if trade_side == TradeSide.BUY else model_msgs.SELL
            request.volume = int(volume * 100)  # Convert to cTrader units: 1 unit = 0.01 lots
            request.clientOrderId = client_order_id
            
            if comment:
                request.comment = comment
            
            self.log(f"Placing market order: {trade_side.name} {volume} {symbol_id}")
            
            deferred = self.connection_manager.client.send(request)
            deferred.addErrback(self.on_error)
            
            return client_order_id
            
        except Exception as e:
            self.log(f"Error placing market order: {e}", "ERROR")
            return None
    
    def place_limit_order(self, symbol_id: str, trade_side: TradeSide, volume: float,
                         limit_price: float, stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None, comment: Optional[str] = None,
                         callback: Optional[Callable] = None) -> Optional[str]:
        """
        Place a limit order
        
        Args:
            symbol_id (str): Symbol to trade
            trade_side (TradeSide): BUY or SELL
            volume (float): Volume to trade
            limit_price (float): Limit price
            stop_loss (float, optional): Stop loss price
            take_profit (float, optional): Take profit price
            comment (str, optional): Order comment
            callback (callable, optional): Callback function for order result
            
        Returns:
            str: Client order ID if successful, None otherwise
        """
        if not self.connection_manager.is_ready:
            self.log("Connection not ready. Cannot place order.", "ERROR")
            return None
            
        try:
            # Generate unique client order ID
            client_order_id = str(uuid.uuid4())
            
            # Create order request
            order_request = OrderModelFactory.create_limit_order(
                symbol_id=symbol_id,
                trade_side=trade_side,
                volume=volume,
                limit_price=limit_price,
                comment=comment
            )
            
            # Store for later SLTP modification if needed
            if stop_loss or take_profit:
                self.pending_orders[client_order_id] = order_request
                self.pending_orders[client_order_id].stopLoss = stop_loss
                self.pending_orders[client_order_id].takeProfit = take_profit
            
            # Store callback if provided
            if callback:
                self.order_callbacks[client_order_id] = callback
            
            # Create API request
            request = api_msgs.ProtoOANewOrderReq()
            request.ctidTraderAccountId = int(self.connection_manager.account_id)
            request.symbolId = int(symbol_id)
            request.orderType = model_msgs.LIMIT
            request.tradeSide = model_msgs.BUY if trade_side == TradeSide.BUY else model_msgs.SELL
            request.volume = int(volume * 100)  # Convert to cTrader units: 1 unit = 0.01 lots
            request.limitPrice = limit_price
            request.clientOrderId = client_order_id
            
            if comment:
                request.comment = comment
            
            self.log(f"Placing limit order: {trade_side.name} {volume} {symbol_id} @ {limit_price}")
            
            deferred = self.connection_manager.client.send(request)
            deferred.addErrback(self.on_error)
            
            return client_order_id
            
        except Exception as e:
            self.log(f"Error placing limit order: {e}", "ERROR")
            return None
    
    def place_stop_order(self, symbol_id: str, trade_side: TradeSide, volume: float,
                        stop_price: float, stop_loss: Optional[float] = None,
                        take_profit: Optional[float] = None, comment: Optional[str] = None,
                        callback: Optional[Callable] = None) -> Optional[str]:
        """
        Place a stop order
        
        Args:
            symbol_id (str): Symbol to trade
            trade_side (TradeSide): BUY or SELL
            volume (float): Volume to trade
            stop_price (float): Stop price
            stop_loss (float, optional): Stop loss price
            take_profit (float, optional): Take profit price
            comment (str, optional): Order comment
            callback (callable, optional): Callback function for order result
            
        Returns:
            str: Client order ID if successful, None otherwise
        """
        if not self.connection_manager.is_ready:
            self.log("Connection not ready. Cannot place order.", "ERROR")
            return None
            
        try:
            # Generate unique client order ID
            client_order_id = str(uuid.uuid4())
            
            # Create order request
            order_request = OrderModelFactory.create_stop_order(
                symbol_id=symbol_id,
                trade_side=trade_side,
                volume=volume,
                stop_price=stop_price,
                comment=comment
            )
            
            # Store for later SLTP modification if needed
            if stop_loss or take_profit:
                self.pending_orders[client_order_id] = order_request
                self.pending_orders[client_order_id].stopLoss = stop_loss
                self.pending_orders[client_order_id].takeProfit = take_profit
            
            # Store callback if provided
            if callback:
                self.order_callbacks[client_order_id] = callback
            
            # Create API request
            request = api_msgs.ProtoOANewOrderReq()
            request.ctidTraderAccountId = int(self.connection_manager.account_id)
            request.symbolId = int(symbol_id)
            request.orderType = model_msgs.STOP
            request.tradeSide = model_msgs.BUY if trade_side == TradeSide.BUY else model_msgs.SELL
            request.volume = int(volume * 100)  # Convert to cTrader units: 1 unit = 0.01 lots
            request.stopPrice = stop_price
            request.clientOrderId = client_order_id
            
            if comment:
                request.comment = comment
            
            self.log(f"Placing stop order: {trade_side.name} {volume} {symbol_id} @ {stop_price}")
            
            deferred = self.connection_manager.client.send(request)
            deferred.addErrback(self.on_error)
            
            return client_order_id
            
        except Exception as e:
            self.log(f"Error placing stop order: {e}", "ERROR")
            return None
    
    def modify_position_sltp(self, position_id: str, stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None, trailing_stop_loss: bool = False,
                           stop_trigger_method: Optional[OrderTriggerMethod] = None,
                           callback: Optional[Callable] = None) -> bool:
        """
        Modify stop loss and take profit for an existing position
        
        Args:
            position_id (str): Position ID to modify
            stop_loss (float, optional): New stop loss price
            take_profit (float, optional): New take profit price
            trailing_stop_loss (bool): Enable trailing stop loss
            stop_trigger_method (OrderTriggerMethod, optional): Stop trigger method
            callback (callable, optional): Callback function for result
            
        Returns:
            bool: True if request was sent successfully
        """
        if not self.connection_manager.is_ready:
            self.log("Connection not ready. Cannot modify position.", "ERROR")
            return False
        
        # REMOVED: Duplicate prevention logic - allow SL/TP modifications
        # The original logic was too aggressive and prevented legitimate requests
        
        try:
            # Create modification request
            modification = SLTPModification(
                positionId=position_id,
                stopLoss=stop_loss,
                takeProfit=take_profit,
                trailingStopLoss=trailing_stop_loss,
                stopTriggerMethod=stop_trigger_method
            )
            
            # Store for callback handling
            mod_id = str(uuid.uuid4())
            self.pending_sltp_modifications[mod_id] = modification
            if callback:
                self.order_callbacks[mod_id] = callback
            
            # Create API request
            request = api_msgs.ProtoOAAmendPositionSLTPReq()
            request.ctidTraderAccountId = int(self.connection_manager.account_id)
            request.positionId = int(position_id)
            
            # FIXED: Ensure values are properly set
            if stop_loss is not None:
                request.stopLoss = float(stop_loss)
                self.log(f"Setting stop loss: {stop_loss}")
            if take_profit is not None:
                request.takeProfit = float(take_profit)
                self.log(f"Setting take profit: {take_profit}")
            if trailing_stop_loss:
                request.trailingStopLoss = trailing_stop_loss
            if stop_trigger_method is not None:
                request.stopTriggerMethod = stop_trigger_method.value
            
            self.log(f"Sending SL/TP modification for position {position_id}: SL={stop_loss}, TP={take_profit}")
            
            deferred = self.connection_manager.client.send(request)
            deferred.addErrback(self.on_error)
            
            # Add to tracking set AFTER successful send
            self.positions_with_sltp.add(position_id)
            
            return True
            
        except Exception as e:
            self.log(f"Error modifying position: {e}", "ERROR")
            
            # Execute error callback if exists
            if callback:
                callback(None, str(e))
            
            return False
    
    def cancel_order(self, order_id: str, callback: Optional[Callable] = None) -> bool:
        """
        Cancel an existing order
        
        Args:
            order_id (str): Order ID to cancel
            callback (callable, optional): Callback function for result
            
        Returns:
            bool: True if request was sent successfully
        """
        if not self.connection_manager.is_ready:
            self.log("Connection not ready. Cannot cancel order.", "ERROR")
            return False
            
        try:
            if callback:
                self.order_callbacks[order_id] = callback
            
            # Create API request
            request = api_msgs.ProtoOACancelOrderReq()
            request.ctidTraderAccountId = int(self.connection_manager.account_id)
            request.orderId = int(order_id)
            
            self.log(f"Cancelling order {order_id}")
            
            deferred = self.connection_manager.client.send(request)
            deferred.addErrback(self.on_error)
            
            return True
            
        except Exception as e:
            self.log(f"Error cancelling order: {e}", "ERROR")
            return False
    
    def get_orders(self, symbol_id: Optional[str] = None) -> List[Order]:
        """
        Get list of orders, optionally filtered by symbol
        
        Args:
            symbol_id (str, optional): Filter by symbol
            
        Returns:
            List[Order]: List of orders
        """
        if symbol_id:
            return [order for order in self.orders.values() if order.symbolId == symbol_id]
        return list(self.orders.values())
    
    def get_pending_orders(self) -> List[Order]:
        """Get list of pending orders"""
        return [order for order in self.orders.values() if order.is_pending]
    
    def get_positions(self, symbol_id: Optional[str] = None):
        """
        Get positions from position service if available
        
        Args:
            symbol_id (str, optional): Filter by symbol
            
        Returns:
            List: List of positions or empty list if position service not available
        """
        # Try to get position service from connection manager
        if hasattr(self.connection_manager, 'position_service'):
            return self.connection_manager.position_service.get_positions(symbol_id)
        
        # Fallback - return empty list and log warning
        self.log("Position service not available. Use PositionService directly.", "WARNING")
        return []
    
    def get_open_positions(self, symbol_id: Optional[str] = None):
        """
        Get open positions from position service if available
        
        Args:
            symbol_id (str, optional): Filter by symbol
            
        Returns:
            List: List of open positions or empty list if position service not available
        """
        # Try to get position service from connection manager
        if hasattr(self.connection_manager, 'position_service'):
            return self.connection_manager.position_service.get_open_positions(symbol_id)
        
        # Fallback - return empty list and log warning
        self.log("Position service not available. Use PositionService directly.", "WARNING")
        return []
    
    def on_execution_event(self, client, message):
        """Handle order execution events"""
        try:
            data = Protobuf.extract(message)
            self.log(f"Execution event received: {data}")
            
            # Check if this is a position modification event (SL/TP related)
            if hasattr(message, 'position') and message.position:
                position_data = Protobuf.extract(message.position)
                position_id = str(position_data.get('positionId', ''))
                
                # Check if this position has SL/TP data
                has_sl = position_data.get('stopLoss') is not None
                has_tp = position_data.get('takeProfit') is not None
                
                if (has_sl or has_tp) and position_id:
                    self.log(f"Position {position_id} SL/TP detected: SL={position_data.get('stopLoss')}, TP={position_data.get('takeProfit')}")
                    
                    # Execute any pending SL/TP callbacks for this position
                    for mod_id, callback in list(self.order_callbacks.items()):
                        if mod_id in self.pending_sltp_modifications:
                            modification = self.pending_sltp_modifications[mod_id]
                            if str(modification.positionId) == position_id:
                                self.log(f"Executing SL/TP success callback for position {position_id}")
                                try:
                                    callback({
                                        'positionId': position_id,
                                        'stopLoss': position_data.get('stopLoss'),
                                        'takeProfit': position_data.get('takeProfit'),
                                        'success': True
                                    }, None)
                                except Exception as e:
                                    self.log(f"Error in SL/TP callback: {e}", "ERROR")
                                
                                # Clean up
                                del self.order_callbacks[mod_id]
                                del self.pending_sltp_modifications[mod_id]
                                break
            
            # Handle order fill events
            if hasattr(message, 'order') and message.order:
                order_data = Protobuf.extract(message.order)
                order = OrderModelFactory.create_order(order_data)
                self.orders[order.orderId] = order
                
                # Check if this is a SL/TP order response
                client_order_id = order_data.get('clientOrderId')
                order_type = order_data.get('orderType')
                
                # Handle SL/TP modification responses
                if order_type == 'STOP_LOSS_TAKE_PROFIT' or 'STOP_LOSS' in str(order_type) or 'TAKE_PROFIT' in str(order_type):
                    self.log(f"SL/TP order created: {order.orderId}, type: {order_type}")
                    
                    # Find and execute pending SL/TP callbacks
                    position_id_in_order = str(order_data.get('positionId', ''))
                    if position_id_in_order:
                        for mod_id, callback in list(self.order_callbacks.items()):
                            if mod_id in self.pending_sltp_modifications:
                                modification = self.pending_sltp_modifications[mod_id]
                                if str(modification.positionId) == position_id_in_order:
                                    self.log(f"Executing SL/TP callback for order creation {mod_id}")
                                    try:
                                        callback({
                                            'orderId': order.orderId,
                                            'positionId': position_id_in_order,
                                            'orderType': order_type,
                                            'success': True
                                        }, None)
                                    except Exception as e:
                                        self.log(f"Error in SL/TP callback: {e}", "ERROR")
                                    
                                    # Clean up
                                    del self.order_callbacks[mod_id]
                                    del self.pending_sltp_modifications[mod_id]
                                    break
                
                # Handle regular order callbacks
                elif client_order_id in self.pending_orders:
                    pending_order = self.pending_orders[client_order_id]
                    
                    # If order was filled and has SL/TP, add them to the position (only once)
                    if (order.is_filled and 
                        (pending_order.stopLoss or pending_order.takeProfit) and
                        not hasattr(pending_order, '_sltp_added')):
                        
                        # Get position ID from the execution event
                        if hasattr(message, 'position') and message.position:
                            position_id = str(message.position.positionId)
                            self.log(f"Adding SL/TP to position {position_id}")
                            
                            # Mark as SL/TP added to prevent duplicates
                            pending_order._sltp_added = True
                            
                            self.modify_position_sltp(
                                position_id=position_id,
                                stop_loss=pending_order.stopLoss,
                                take_profit=pending_order.takeProfit
                            )
                    
                    # Clean up pending order only after processing
                    if order.is_filled or order.is_cancelled:
                        del self.pending_orders[client_order_id]
                
                # Execute order callback if exists
                if client_order_id in self.order_callbacks and order_type != 'STOP_LOSS_TAKE_PROFIT':
                    callback = self.order_callbacks[client_order_id]
                    try:
                        callback(order, None)  # Success callback
                    except Exception as e:
                        self.log(f"Error in order callback: {e}", "ERROR")
                    del self.order_callbacks[client_order_id]
            
            # Save orders data
            self._save_orders()
            
        except Exception as e:
            self.log(f"Error handling execution event: {e}", "ERROR")
    
    def on_order_error_event(self, client, message):
        """Handle order error events"""
        try:
            data = Protobuf.extract(message)
            self.log(f"Order error event: {data}", "ERROR")
            
            # Check if this is a SL/TP modification error
            position_id = data.get('positionId')
            if position_id:
                # Remove from SL/TP set if modification failed
                self.positions_with_sltp.discard(str(position_id))
                
                # Execute SL/TP error callbacks
                for mod_id, callback in list(self.order_callbacks.items()):
                    if mod_id in self.pending_sltp_modifications:
                        modification = self.pending_sltp_modifications[mod_id]
                        if str(modification.positionId) == str(position_id):
                            try:
                                callback(None, data.get('description', 'Unknown SL/TP error'))
                            except Exception as e:
                                self.log(f"Error in SL/TP error callback: {e}", "ERROR")
                            
                            # Clean up
                            del self.order_callbacks[mod_id]
                            del self.pending_sltp_modifications[mod_id]
                            break
            
            # Execute regular order error callback if exists
            client_order_id = data.get('clientOrderId')
            if client_order_id in self.order_callbacks:
                callback = self.order_callbacks[client_order_id]
                try:
                    callback(None, data.get('description', 'Unknown error'))
                except Exception as e:
                    self.log(f"Error in error callback: {e}", "ERROR")
                del self.order_callbacks[client_order_id]
            
            # Clean up pending order
            if client_order_id in self.pending_orders:
                del self.pending_orders[client_order_id]
                
        except Exception as e:
            self.log(f"Error handling order error event: {e}", "ERROR")
    
    def on_order_list_response(self, client, message):
        """Handle order list responses"""
        try:
            data = Protobuf.extract(message)
            self.log(f"Order list response: {len(data.get('order', []))} orders")
            
            # Update orders from list
            for order_data in data.get('order', []):
                order = OrderModelFactory.create_order(order_data)
                self.orders[order.orderId] = order
            
            self._save_orders()
            
        except Exception as e:
            self.log(f"Error handling order list response: {e}", "ERROR")
    
    def on_position_unrealized_pnl(self, client, message):
        """Handle position unrealized PnL responses"""
        try:
            data = Protobuf.extract(message)
            self.log(f"Position unrealized PnL: {data}")
            
        except Exception as e:
            self.log(f"Error handling position PnL response: {e}", "ERROR")
    
    def on_amend_position_response(self, client, message):
        """Handle amend position SL/TP responses"""
        try:
            data = Protobuf.extract(message)
            self.log(f"Amend position response: {data}")
            
            # Check if this is a successful SL/TP modification
            position_id = str(data.get('positionId', ''))
            
            if position_id:
                self.log(f"SL/TP amendment response for position {position_id}")
                
                # Execute callbacks for pending modifications
                for mod_id, callback in list(self.order_callbacks.items()):
                    if mod_id in self.pending_sltp_modifications:
                        modification = self.pending_sltp_modifications[mod_id]
                        if str(modification.positionId) == position_id:
                            self.log(f"Executing SL/TP success callback for amend response")
                            try:
                                callback({
                                    'positionId': position_id,
                                    'response': data,
                                    'success': True,
                                    'type': 'amend_response'
                                }, None)  # Success callback
                            except Exception as e:
                                self.log(f"Error in amend callback: {e}", "ERROR")
                            
                            # Clean up
                            del self.order_callbacks[mod_id]
                            del self.pending_sltp_modifications[mod_id]
                            break
            else:
                # General success response - execute all pending callbacks
                self.log("General amend position response - executing all pending callbacks")
                for mod_id, callback in list(self.order_callbacks.items()):
                    if mod_id in self.pending_sltp_modifications:
                        try:
                            callback({
                                'response': data,
                                'success': True,
                                'type': 'general_amend_response'
                            }, None)
                        except Exception as e:
                            self.log(f"Error in general amend callback: {e}", "ERROR")
                        
                        # Clean up
                        del self.order_callbacks[mod_id]
                        del self.pending_sltp_modifications[mod_id]
            
        except Exception as e:
            self.log(f"Error handling amend position response: {e}", "ERROR")
    
    def _save_orders(self):
        """Save orders to file"""
        try:
            orders_data = {}
            for order_id, order in self.orders.items():
                orders_data[order_id] = {
                    'orderId': order.orderId,
                    'orderType': order.orderType.value,
                    'orderStatus': order.orderStatus.value,
                    'tradeSide': order.tradeSide.value,
                    'symbolId': order.symbolId,
                    'requestedVolume': int(order.requestedVolume * 100),
                    'executedVolume': int(order.executedVolume * 100),
                    'closingOrder': order.closingOrder,
                    'channel': order.channel,
                    'comment': order.comment,
                    'limitPrice': order.limitPrice,
                    'stopPrice': order.stopPrice,
                    'stopLoss': order.stopLoss,
                    'takeProfit': order.takeProfit,
                    'utcLastUpdateTimestamp': order.utcLastUpdateTimestamp
                }
            
            self.save_data("orders", orders_data)
            
        except Exception as e:
            self.log(f"Error saving orders: {e}", "ERROR")