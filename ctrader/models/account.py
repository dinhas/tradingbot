"""
Account Model for cTrader API

Data structures and models for account-related information including
account details, positions, assets, and balance information.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class AccountType(Enum):
    """Account type enumeration"""
    HEDGED = 0
    NETTED = 1
    SPREAD_BETTING = 2


class TradeSide(Enum):
    """Trade side enumeration"""
    BUY = 1
    SELL = 2


class PositionStatus(Enum):
    """Position status enumeration"""
    OPEN = 1
    CLOSED = 2
    ERROR = 3


@dataclass
class AssetInfo:
    """Asset information model"""
    assetId: str
    name: str
    displayName: str
    digits: int
    country: Optional[str] = None
    assetClass: Optional[str] = None


@dataclass
class TradeData:
    """Trade data model for positions"""
    symbolId: str
    volume: float
    tradeSide: TradeSide
    openPrice: float
    openTimestamp: int


@dataclass
class Position:
    """Position model"""
    positionId: str
    tradeData: TradeData
    positionStatus: PositionStatus
    swap: float
    price: float
    commission: float
    utcLastUpdateTimestamp: int
    stopLoss: Optional[float] = None
    takeProfit: Optional[float] = None
    marginRate: Optional[float] = None
    mirroringCommission: Optional[float] = None
    usedMargin: Optional[float] = None
    guaranteedStopLoss: bool = False
    trailingStopLoss: bool = False
    stopLossTriggerMethod: Optional[str] = None
    moneyDigits: int = 2
    
    @property
    def is_open(self) -> bool:
        """Check if position is open"""
        return self.positionStatus == PositionStatus.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if position is closed"""
        return self.positionStatus == PositionStatus.CLOSED
    
    @property
    def has_stop_loss(self) -> bool:
        """Check if position has stop loss"""
        return self.stopLoss is not None
    
    @property
    def has_take_profit(self) -> bool:
        """Check if position has take profit"""
        return self.takeProfit is not None
    
    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        if self.tradeData.tradeSide == TradeSide.BUY:
            return (self.price - self.tradeData.openPrice) * self.tradeData.volume
        else:
            return (self.tradeData.openPrice - self.price) * self.tradeData.volume
    
    @property
    def net_pnl(self) -> float:
        """Calculate net P&L including fees"""
        return self.unrealized_pnl + self.swap - self.commission
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            "positionId": self.positionId,
            "tradeData": self.tradeData.to_dict() if hasattr(self.tradeData, 'to_dict') else vars(self.tradeData),
            "positionStatus": self.positionStatus.value,
            "swap": self.swap,
            "price": self.price,
            "commission": self.commission,
            "utcLastUpdateTimestamp": self.utcLastUpdateTimestamp,
            "stopLoss": self.stopLoss,
            "takeProfit": self.takeProfit,
            "marginRate": self.marginRate,
            "mirroringCommission": self.mirroringCommission,
            "usedMargin": self.usedMargin,
            "guaranteedStopLoss": self.guaranteedStopLoss,
            "trailingStopLoss": self.trailingStopLoss,
            "stopLossTriggerMethod": self.stopLossTriggerMethod,
            "moneyDigits": self.moneyDigits
        }


@dataclass
class AccountInfo:
    """Account information model"""
    ctidTraderAccountId: str
    login: str
    balance: float
    equity: float
    margin: float
    freeMargin: float
    marginLevel: float
    brokerName: str
    accountType: AccountType
    lastUpdated: datetime
    balanceVersion: str
    leverageInCents: int
    maxLeverage: int
    depositAssetId: str
    swapFree: bool
    moneyDigits: int
    isLimitedRisk: bool
    registrationTimestamp: int
    credit: float = 0.0
    swap: float = 0.0
    nonWithdrawableBonus: Optional[float] = None
    accessRights: Optional[str] = None
    frenchRisk: Optional[bool] = None
    
    @property
    def available_margin_percentage(self) -> float:
        """Calculate available margin as percentage"""
        if self.equity == 0:
            return 0.0
        return (self.freeMargin / self.equity) * 100
    
    @property
    def margin_usage_percentage(self) -> float:
        """Calculate margin usage as percentage"""
        if self.equity == 0:
            return 0.0
        return (self.margin / self.equity) * 100
    
    @property
    def leverage_ratio(self) -> float:
        """Get leverage ratio (e.g., 100 for 1:100)"""
        return self.maxLeverage / 100.0
    
    def can_trade(self, required_margin: float) -> bool:
        """Check if account can handle required margin"""
        return self.freeMargin >= required_margin
    
    def risk_level(self) -> str:
        """Get risk level based on margin level"""
        if self.marginLevel >= 500:
            return "LOW"
        elif self.marginLevel >= 200:
            return "MEDIUM"
        elif self.marginLevel >= 100:
            return "HIGH"
        else:
            return "CRITICAL"


@dataclass
class AccountSummary:
    """Account summary model"""
    login: str
    brokerName: str
    accountType: AccountType
    balance: float
    equity: float
    margin: float
    freeMargin: float
    marginLevel: float
    positionsCount: int
    lastUpdated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "login": self.login,
            "brokerName": self.brokerName,
            "accountType": self.accountType.name,
            "balance": self.balance,
            "equity": self.equity,
            "margin": self.margin,
            "freeMargin": self.freeMargin,
            "marginLevel": self.marginLevel,
            "positionsCount": self.positionsCount,
            "lastUpdated": self.lastUpdated.isoformat()
        }


class AccountModelFactory:
    """Factory for creating account models from API data"""
    
    @staticmethod
    def create_account_info(data: Dict[str, Any]) -> AccountInfo:
        """Create AccountInfo from dictionary data"""
        return AccountInfo(
            ctidTraderAccountId=data["ctidTraderAccountId"],
            login=data["login"],
            balance=data["balance"],
            equity=data["equity"],
            margin=data["margin"],
            freeMargin=data["freeMargin"],
            marginLevel=data["marginLevel"],
            brokerName=data["brokerName"],
            accountType=AccountType(data["accountType"]) if isinstance(data["accountType"], int) else AccountType[data["accountType"]],
            lastUpdated=datetime.fromisoformat(data["lastUpdated"]),
            balanceVersion=data["balanceVersion"],
            leverageInCents=data["leverageInCents"],
            maxLeverage=data["maxLeverage"],
            depositAssetId=data["depositAssetId"],
            swapFree=data["swapFree"],
            moneyDigits=data["moneyDigits"],
            isLimitedRisk=data["isLimitedRisk"],
            registrationTimestamp=data["registrationTimestamp"],
            credit=data.get("credit", 0.0),
            swap=data.get("swap", 0.0),
            nonWithdrawableBonus=data.get("nonWithdrawableBonus"),
            accessRights=data.get("accessRights"),
            frenchRisk=data.get("frenchRisk")
        )
    
    @staticmethod
    def create_position(data: Dict[str, Any]) -> Position:
        """Create Position from dictionary data"""
        try:
            # Handle nested tradeData structure
            if 'tradeData' in data:
                trade_data_dict = data['tradeData']
            else:
                # Create trade data from position data directly
                trade_data_dict = {
                    'symbolId': data.get('symbolId', ''),
                    'volume': data.get('volume', 0),
                    'tradeSide': data.get('tradeSide', 'BUY'),
                    'openPrice': data.get('price', 0),
                    'openTimestamp': data.get('openTimestamp', 0)
                }
            
            # Parse trade side
            trade_side_val = trade_data_dict.get('tradeSide', 'BUY')
            if isinstance(trade_side_val, str):
                trade_side = TradeSide.BUY if trade_side_val == 'BUY' else TradeSide.SELL
            else:
                trade_side = TradeSide.BUY if trade_side_val == 1 else TradeSide.SELL
            
            trade_data = TradeData(
                symbolId=str(trade_data_dict.get('symbolId', '')),
                volume=float(trade_data_dict.get('volume', 0)) / 100,  # Convert from cents
                tradeSide=trade_side,
                openPrice=float(trade_data_dict.get('openPrice', 0)),
                openTimestamp=int(trade_data_dict.get('openTimestamp', 0))
            )
            
            # Parse position status
            status_val = data.get('positionStatus', 'POSITION_STATUS_OPEN')
            if isinstance(status_val, str):
                if 'OPEN' in status_val:
                    position_status = PositionStatus.OPEN
                else:
                    position_status = PositionStatus.CLOSED
            else:
                position_status = PositionStatus.OPEN if status_val == 1 else PositionStatus.CLOSED
            
            return Position(
                positionId=str(data.get('positionId', '')),
                tradeData=trade_data,
                positionStatus=position_status,
                swap=float(data.get('swap', 0)),
                price=float(data.get('price', 0)),
                commission=float(data.get('commission', 0)),
                utcLastUpdateTimestamp=int(data.get('utcLastUpdateTimestamp', 0)),
                stopLoss=data.get('stopLoss'),
                takeProfit=data.get('takeProfit'),
                marginRate=data.get('marginRate'),
                mirroringCommission=data.get('mirroringCommission'),
                usedMargin=data.get('usedMargin'),
                guaranteedStopLoss=bool(data.get('guaranteedStopLoss', False)),
                trailingStopLoss=bool(data.get('trailingStopLoss', False)),
                stopLossTriggerMethod=data.get('stopLossTriggerMethod'),
                moneyDigits=data.get('moneyDigits', 2)
            )
            
        except Exception as e:
            raise ValueError(f"Error creating position from data: {e}")
    
    @staticmethod
    def create_asset_info(data: Dict[str, Any]) -> AssetInfo:
        """Create AssetInfo from dictionary data"""
        return AssetInfo(
            assetId=data["assetId"],
            name=data["name"],
            displayName=data["displayName"],
            digits=data["digits"],
            country=data.get("country"),
            assetClass=data.get("assetClass")
        )