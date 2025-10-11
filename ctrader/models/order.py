"""
Order Model for cTrader API

Data structures and models for order-related information including
order types, execution types, and order management.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from .account import TradeSide


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = 1
    LIMIT = 2
    STOP = 3
    STOP_LIMIT = 4


class OrderTriggerMethod(Enum):
    """Order trigger method enumeration"""
    TRADE = 1
    OPPOSITE = 2
    DOUBLE_TRADE = 3


class OrderExecutionType(Enum):
    """Order execution type enumeration"""
    FILL_OR_KILL = 1
    IMMEDIATE_OR_CANCEL = 2
    LIMIT_ON_OPEN = 3
    LIMIT_ON_CLOSE = 4
    MARKET_ON_OPEN = 5
    MARKET_ON_CLOSE = 6


class OrderStatus(Enum):
    """Order status enumeration"""
    ORDER_STATUS_ACCEPTED = 1
    ORDER_STATUS_FILLED = 2
    ORDER_STATUS_REJECTED = 3
    ORDER_STATUS_CANCELLED = 4
    ORDER_STATUS_EXPIRED = 5


@dataclass
class OrderRequest:
    """Order request model for placing new orders"""
    symbolId: str
    orderType: OrderType
    tradeSide: TradeSide
    volume: float
    stopLoss: Optional[float] = None
    takeProfit: Optional[float] = None
    expirationTimestamp: Optional[int] = None
    stopTriggerMethod: Optional[OrderTriggerMethod] = None
    comment: Optional[str] = None
    baseSlippagePrice: Optional[float] = None
    slippageInPips: Optional[int] = None
    relativeStopLoss: Optional[int] = None
    relativeTakeProfit: Optional[int] = None
    guaranteedStopLoss: bool = False
    trailingStopLoss: bool = False
    executionType: Optional[OrderExecutionType] = None
    
    # For limit/stop orders
    limitPrice: Optional[float] = None
    stopPrice: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request"""
        data = {
            "symbolId": self.symbolId,
            "orderType": self.orderType.value,
            "tradeSide": self.tradeSide.value,
            "volume": int(self.volume * 100),  # Convert to cTrader units: 1 unit = 0.01 lots
        }
        
        if self.limitPrice is not None:
            data["limitPrice"] = self.limitPrice
        if self.stopPrice is not None:
            data["stopPrice"] = self.stopPrice
        if self.stopLoss is not None:
            data["stopLoss"] = self.stopLoss
        if self.takeProfit is not None:
            data["takeProfit"] = self.takeProfit
        if self.expirationTimestamp is not None:
            data["expirationTimestamp"] = self.expirationTimestamp
        if self.stopTriggerMethod is not None:
            data["stopTriggerMethod"] = self.stopTriggerMethod.value
        if self.comment is not None:
            data["comment"] = self.comment
        if self.baseSlippagePrice is not None:
            data["baseSlippagePrice"] = self.baseSlippagePrice
        if self.slippageInPips is not None:
            data["slippageInPips"] = self.slippageInPips
        if self.relativeStopLoss is not None:
            data["relativeStopLoss"] = self.relativeStopLoss
        if self.relativeTakeProfit is not None:
            data["relativeTakeProfit"] = self.relativeTakeProfit
        if self.guaranteedStopLoss:
            data["guaranteedStopLoss"] = self.guaranteedStopLoss
        if self.trailingStopLoss:
            data["trailingStopLoss"] = self.trailingStopLoss
        if self.executionType is not None:
            data["executionType"] = self.executionType.value
            
        return data


@dataclass
class Order:
    """Order model"""
    orderId: str
    orderType: OrderType
    orderStatus: OrderStatus
    tradeSide: TradeSide
    symbolId: str
    requestedVolume: float
    executedVolume: float
    closingOrder: bool
    channel: Optional[str] = None
    comment: Optional[str] = None
    limitPrice: Optional[float] = None
    stopPrice: Optional[float] = None
    stopLoss: Optional[float] = None
    takeProfit: Optional[float] = None
    baseSlippagePrice: Optional[float] = None
    slippageInPips: Optional[int] = None
    relativeStopLoss: Optional[int] = None
    relativeTakeProfit: Optional[int] = None
    guaranteedStopLoss: bool = False
    trailingStopLoss: bool = False
    stopTriggerMethod: Optional[OrderTriggerMethod] = None
    executionType: Optional[OrderExecutionType] = None
    expirationTimestamp: Optional[int] = None
    utcLastUpdateTimestamp: Optional[int] = None
    executionPrice: Optional[float] = None  # Price at which order was executed
    
    @property
    def is_pending(self) -> bool:
        """Check if order is pending execution"""
        return self.orderStatus == OrderStatus.ORDER_STATUS_ACCEPTED
    
    @property
    def is_filled(self) -> bool:
        """Check if order is filled"""
        return self.orderStatus == OrderStatus.ORDER_STATUS_FILLED
    
    @property
    def is_cancelled(self) -> bool:
        """Check if order is cancelled"""
        return self.orderStatus == OrderStatus.ORDER_STATUS_CANCELLED
    
    @property
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled"""
        return (self.orderStatus == OrderStatus.ORDER_STATUS_ACCEPTED and 
                self.executedVolume > 0 and 
                self.executedVolume < self.requestedVolume)
    
    @property
    def remaining_volume(self) -> float:
        """Get remaining volume to be executed"""
        return self.requestedVolume - self.executedVolume


@dataclass
class SLTPModification:
    """Stop Loss / Take Profit modification request"""
    positionId: str
    stopLoss: Optional[float] = None
    takeProfit: Optional[float] = None
    trailingStopLoss: bool = False
    stopTriggerMethod: Optional[OrderTriggerMethod] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request"""
        data = {"positionId": self.positionId}
        
        if self.stopLoss is not None:
            data["stopLoss"] = self.stopLoss
        if self.takeProfit is not None:
            data["takeProfit"] = self.takeProfit
        if self.trailingStopLoss:
            data["trailingStopLoss"] = self.trailingStopLoss
        if self.stopTriggerMethod is not None:
            data["stopTriggerMethod"] = self.stopTriggerMethod.value
            
        return data


class OrderModelFactory:
    """Factory for creating order models from API data"""
    
    @staticmethod
    def create_order(data: Dict[str, Any]) -> Order:
        """Create Order from dictionary data"""
        return Order(
            orderId=str(data["orderId"]),
            orderType=OrderType(data["orderType"]),
            orderStatus=OrderStatus(data["orderStatus"]),
            tradeSide=TradeSide(data["tradeSide"]),
            symbolId=str(data["symbolId"]),
            requestedVolume=data["requestedVolume"] / 100.0,  # Convert from cTrader units
            executedVolume=data.get("executedVolume", 0) / 100.0,
            closingOrder=data.get("closingOrder", False),
            channel=data.get("channel"),
            comment=data.get("comment"),
            limitPrice=data.get("limitPrice"),
            stopPrice=data.get("stopPrice"),
            stopLoss=data.get("stopLoss"),
            takeProfit=data.get("takeProfit"),
            baseSlippagePrice=data.get("baseSlippagePrice"),
            slippageInPips=data.get("slippageInPips"),
            relativeStopLoss=data.get("relativeStopLoss"),
            relativeTakeProfit=data.get("relativeTakeProfit"),
            guaranteedStopLoss=data.get("guaranteedStopLoss", False),
            trailingStopLoss=data.get("trailingStopLoss", False),
            stopTriggerMethod=OrderTriggerMethod(data["stopTriggerMethod"]) if data.get("stopTriggerMethod") else None,
            executionType=OrderExecutionType(data["executionType"]) if data.get("executionType") else None,
            expirationTimestamp=data.get("expirationTimestamp"),
            utcLastUpdateTimestamp=data.get("utcLastUpdateTimestamp")
        )
    
    @staticmethod
    def create_market_order(symbol_id: str, trade_side: TradeSide, volume: float, 
                          comment: Optional[str] = None) -> OrderRequest:
        """Create a market order request"""
        return OrderRequest(
            symbolId=symbol_id,
            orderType=OrderType.MARKET,
            tradeSide=trade_side,
            volume=volume,
            comment=comment
        )
    
    @staticmethod
    def create_limit_order(symbol_id: str, trade_side: TradeSide, volume: float,
                          limit_price: float, comment: Optional[str] = None) -> OrderRequest:
        """Create a limit order request"""
        return OrderRequest(
            symbolId=symbol_id,
            orderType=OrderType.LIMIT,
            tradeSide=trade_side,
            volume=volume,
            limitPrice=limit_price,
            comment=comment
        )
    
    @staticmethod
    def create_stop_order(symbol_id: str, trade_side: TradeSide, volume: float,
                         stop_price: float, comment: Optional[str] = None) -> OrderRequest:
        """Create a stop order request"""
        return OrderRequest(
            symbolId=symbol_id,
            orderType=OrderType.STOP,
            tradeSide=trade_side,
            volume=volume,
            stopPrice=stop_price,
            comment=comment
        )