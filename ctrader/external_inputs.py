"""
External Input System for cTrader Trading Bot

This module provides various input sources for trading signals including:
- Webhook receivers
- File watchers
- API endpoints
- Manual input interfaces
- Strategy signals
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Callable
from enum import Enum
import json
import os
from datetime import datetime
import asyncio
from pathlib import Path


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"
    MODIFY = "MODIFY"


class SignalSource(Enum):
    """Sources of trading signals"""
    WEBHOOK = "webhook"
    FILE = "file"
    API = "api"
    MANUAL = "manual"
    STRATEGY = "strategy"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_type: SignalType
    symbol: str
    volume: float
    source: SignalSource
    timestamp: datetime
    signal_id: Optional[str] = None
    
    # Order parameters
    order_type: str = "MARKET"  # MARKET, LIMIT, STOP
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Additional parameters
    comment: Optional[str] = None
    risk_percent: Optional[float] = None
    position_id: Optional[str] = None  # For CLOSE or MODIFY signals
    
    # Metadata
    strategy_name: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['signal_type'] = self.signal_type.value
        data['source'] = self.source.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create signal from dictionary"""
        data['signal_type'] = SignalType(data['signal_type'])
        data['source'] = SignalSource(data['source'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate signal data and return list of errors"""
        errors = []
        
        if not self.symbol:
            errors.append("Symbol is required")
        
        if self.volume <= 0:
            errors.append("Volume must be positive")
        
        if self.signal_type in [SignalType.BUY, SignalType.SELL]:
            if self.order_type == "LIMIT" and not self.entry_price:
                errors.append("Entry price required for LIMIT orders")
            elif self.order_type == "STOP" and not self.entry_price:
                errors.append("Entry price required for STOP orders")
        
        if self.signal_type in [SignalType.CLOSE, SignalType.MODIFY] and not self.position_id:
            errors.append("Position ID required for CLOSE/MODIFY signals")
        
        return errors


class SignalProcessor:
    """Base class for processing trading signals"""
    
    def __init__(self, trading_service, callback: Optional[Callable] = None):
        """
        Initialize signal processor
        
        Args:
            trading_service: TradingService instance
            callback: Optional callback for signal results
        """
        self.trading_service = trading_service
        self.callback = callback
        self.processed_signals: List[TradingSignal] = []
    
    def process_signal(self, signal: TradingSignal) -> bool:
        """
        Process a trading signal
        
        Args:
            signal: TradingSignal to process
            
        Returns:
            bool: True if signal was processed successfully
        """
        # Validate signal
        errors = signal.validate()
        if errors:
            self._handle_error(signal, f"Validation errors: {', '.join(errors)}")
            return False
        
        try:
            # Process based on signal type
            if signal.signal_type == SignalType.BUY:
                return self._process_buy_signal(signal)
            elif signal.signal_type == SignalType.SELL:
                return self._process_sell_signal(signal)
            elif signal.signal_type == SignalType.CLOSE:
                return self._process_close_signal(signal)
            elif signal.signal_type == SignalType.MODIFY:
                return self._process_modify_signal(signal)
            else:
                self._handle_error(signal, f"Unknown signal type: {signal.signal_type}")
                return False
                
        except Exception as e:
            self._handle_error(signal, f"Processing error: {e}")
            return False
    
    def _process_buy_signal(self, signal: TradingSignal) -> bool:
        """Process buy signal"""
        from ctrader.models.account import TradeSide
        
        def order_callback(order, error):
            if self.callback:
                self.callback(signal, order, error)
        
        if signal.order_type == "MARKET":
            client_order_id = self.trading_service.place_market_order(
                symbol_id=signal.symbol,
                trade_side=TradeSide.BUY,
                volume=signal.volume,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=signal.comment or f"Signal {signal.signal_id}",
                callback=order_callback
            )
        elif signal.order_type == "LIMIT":
            client_order_id = self.trading_service.place_limit_order(
                symbol_id=signal.symbol,
                trade_side=TradeSide.BUY,
                volume=signal.volume,
                limit_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=signal.comment or f"Signal {signal.signal_id}",
                callback=order_callback
            )
        elif signal.order_type == "STOP":
            client_order_id = self.trading_service.place_stop_order(
                symbol_id=signal.symbol,
                trade_side=TradeSide.BUY,
                volume=signal.volume,
                stop_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=signal.comment or f"Signal {signal.signal_id}",
                callback=order_callback
            )
        else:
            self._handle_error(signal, f"Unknown order type: {signal.order_type}")
            return False
        
        if client_order_id:
            self.processed_signals.append(signal)
            return True
        return False
    
    def _process_sell_signal(self, signal: TradingSignal) -> bool:
        """Process sell signal"""
        from ctrader.models.account import TradeSide
        
        def order_callback(order, error):
            if self.callback:
                self.callback(signal, order, error)
        
        if signal.order_type == "MARKET":
            client_order_id = self.trading_service.place_market_order(
                symbol_id=signal.symbol,
                trade_side=TradeSide.SELL,
                volume=signal.volume,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=signal.comment or f"Signal {signal.signal_id}",
                callback=order_callback
            )
        elif signal.order_type == "LIMIT":
            client_order_id = self.trading_service.place_limit_order(
                symbol_id=signal.symbol,
                trade_side=TradeSide.SELL,
                volume=signal.volume,
                limit_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=signal.comment or f"Signal {signal.signal_id}",
                callback=order_callback
            )
        elif signal.order_type == "STOP":
            client_order_id = self.trading_service.place_stop_order(
                symbol_id=signal.symbol,
                trade_side=TradeSide.SELL,
                volume=signal.volume,
                stop_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                comment=signal.comment or f"Signal {signal.signal_id}",
                callback=order_callback
            )
        else:
            self._handle_error(signal, f"Unknown order type: {signal.order_type}")
            return False
        
        if client_order_id:
            self.processed_signals.append(signal)
            return True
        return False
    
    def _process_close_signal(self, signal: TradingSignal) -> bool:
        """Process close signal"""
        # This would require position service implementation
        # For now, just log the signal
        print(f"Close signal received for position {signal.position_id}")
        if self.callback:
            self.callback(signal, None, "Position closing not implemented yet")
        return True
    
    def _process_modify_signal(self, signal: TradingSignal) -> bool:
        """Process modify signal"""
        def modify_callback(result, error):
            if self.callback:
                self.callback(signal, result, error)
        
        success = self.trading_service.modify_position_sltp(
            position_id=signal.position_id,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            callback=modify_callback
        )
        
        if success:
            self.processed_signals.append(signal)
        return success
    
    def _handle_error(self, signal: TradingSignal, error_msg: str):
        """Handle signal processing error"""
        print(f"Signal processing error: {error_msg}")
        if self.callback:
            self.callback(signal, None, error_msg)


class FileSignalWatcher:
    """Watch for trading signals from files"""
    
    def __init__(self, signal_processor: SignalProcessor, watch_directory: str = "signals"):
        """
        Initialize file watcher
        
        Args:
            signal_processor: SignalProcessor instance
            watch_directory: Directory to watch for signal files
        """
        self.signal_processor = signal_processor
        self.watch_directory = Path(watch_directory)
        self.watch_directory.mkdir(exist_ok=True)
        self.processed_files: set = set()
    
    def start_watching(self):
        """Start watching for signal files"""
        print(f"👁️  Watching for signals in: {self.watch_directory}")
        
        while True:
            try:
                # Check for new JSON files
                for file_path in self.watch_directory.glob("*.json"):
                    if file_path.name not in self.processed_files:
                        self._process_signal_file(file_path)
                        self.processed_files.add(file_path.name)
                
                # Sleep for a bit before checking again
                import time
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("File watcher stopped")
                break
            except Exception as e:
                print(f"Error in file watcher: {e}")
                import time
                time.sleep(5)
    
    def _process_signal_file(self, file_path: Path):
        """Process a signal file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle both single signal and array of signals
            if isinstance(data, list):
                signals = [TradingSignal.from_dict(item) for item in data]
            else:
                signals = [TradingSignal.from_dict(data)]
            
            print(f"📄 Processing {len(signals)} signal(s) from {file_path.name}")
            
            for signal in signals:
                success = self.signal_processor.process_signal(signal)
                print(f"   {'✅' if success else '❌'} {signal.signal_type.value} {signal.symbol}")
            
            # Move processed file to processed folder
            processed_dir = self.watch_directory / "processed"
            processed_dir.mkdir(exist_ok=True)
            file_path.rename(processed_dir / file_path.name)
            
        except Exception as e:
            print(f"Error processing signal file {file_path}: {e}")
            # Move to error folder
            error_dir = self.watch_directory / "error"
            error_dir.mkdir(exist_ok=True)
            file_path.rename(error_dir / file_path.name)


class WebhookSignalReceiver:
    """Receive trading signals via webhook"""
    
    def __init__(self, signal_processor: SignalProcessor, port: int = 8080):
        """
        Initialize webhook receiver
        
        Args:
            signal_processor: SignalProcessor instance
            port: Port to listen on
        """
        self.signal_processor = signal_processor
        self.port = port
    
    async def start_server(self):
        """Start webhook server"""
        from aiohttp import web, ClientResponse
        
        async def handle_webhook(request):
            try:
                data = await request.json()
                
                # Handle both single signal and array of signals
                if isinstance(data, list):
                    signals = [TradingSignal.from_dict(item) for item in data]
                else:
                    signals = [TradingSignal.from_dict(data)]
                
                results = []
                for signal in signals:
                    success = self.signal_processor.process_signal(signal)
                    results.append({
                        'signal_id': signal.signal_id,
                        'success': success,
                        'message': 'Signal processed' if success else 'Signal failed'
                    })
                
                return web.json_response({
                    'status': 'success',
                    'processed': len(results),
                    'results': results
                })
                
            except Exception as e:
                return web.json_response({
                    'status': 'error',
                    'message': str(e)
                }, status=400)
        
        app = web.Application()
        app.router.add_post('/webhook', handle_webhook)
        app.router.add_post('/signal', handle_webhook)  # Alternative endpoint
        
        print(f"🌐 Starting webhook server on port {self.port}")
        await web._run_app(app, port=self.port)


class ManualSignalInterface:
    """Manual interface for entering trading signals"""
    
    def __init__(self, signal_processor: SignalProcessor):
        """Initialize manual interface"""
        self.signal_processor = signal_processor
    
    def start_interactive_mode(self):
        """Start interactive signal entry mode"""
        print("\n🖐️  Manual Signal Entry Mode")
        print("=" * 50)
        
        while True:
            try:
                print("\nSignal Types: BUY, SELL, CLOSE, MODIFY, QUIT")
                signal_type = input("Enter signal type: ").strip().upper()
                
                if signal_type == "QUIT":
                    break
                
                if signal_type not in ["BUY", "SELL", "CLOSE", "MODIFY"]:
                    print("❌ Invalid signal type")
                    continue
                
                signal = self._get_signal_from_user(SignalType(signal_type))
                if signal:
                    success = self.signal_processor.process_signal(signal)
                    print(f"{'✅' if success else '❌'} Signal {'processed' if success else 'failed'}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _get_signal_from_user(self, signal_type: SignalType) -> Optional[TradingSignal]:
        """Get signal details from user input"""
        try:
            symbol = input("Symbol: ").strip()
            if not symbol:
                print("❌ Symbol is required")
                return None
            
            if signal_type in [SignalType.CLOSE, SignalType.MODIFY]:
                position_id = input("Position ID: ").strip()
                if not position_id:
                    print("❌ Position ID is required for CLOSE/MODIFY")
                    return None
            else:
                position_id = None
            
            if signal_type in [SignalType.BUY, SignalType.SELL]:
                volume = float(input("Volume (e.g., 0.01): ").strip())
                order_type = input("Order type (MARKET/LIMIT/STOP) [MARKET]: ").strip().upper() or "MARKET"
                
                entry_price = None
                if order_type in ["LIMIT", "STOP"]:
                    entry_price = float(input("Entry price: ").strip())
            else:
                volume = 0.01  # Default for CLOSE/MODIFY
                order_type = "MARKET"
                entry_price = None
            
            stop_loss_str = input("Stop Loss (optional): ").strip()
            stop_loss = float(stop_loss_str) if stop_loss_str else None
            
            take_profit_str = input("Take Profit (optional): ").strip()
            take_profit = float(take_profit_str) if take_profit_str else None
            
            comment = input("Comment (optional): ").strip() or None
            
            return TradingSignal(
                signal_type=signal_type,
                symbol=symbol,
                volume=volume,
                source=SignalSource.MANUAL,
                timestamp=datetime.now(),
                signal_id=f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                order_type=order_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                comment=comment,
                position_id=position_id
            )
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            return None
        except Exception as e:
            print(f"❌ Error creating signal: {e}")
            return None