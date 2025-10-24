import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from src.indicators import calculate_donchian_channels, calculate_atr, calculate_volume_ma
from src.config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    pnl_pct: float

    def __repr__(self):
        return (f"Trade({self.direction}, entry={self.entry_price:.5f}, "
                f"exit={self.exit_price:.5f}, pnl={self.pnl:.2f})")


class BacktestEngine:
    """Core backtesting engine for forex strategies."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]

    def reset(self):
        """Reset the backtest state."""
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = [self.initial_capital]

    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on current capital (25% of capital)."""
        position_value = self.capital * POSITION_SIZE_PCT
        return position_value / price

    def apply_spread(self, price: float, direction: str, action: str) -> float:
        """
        Apply spread cost to entry/exit prices.

        Args:
            price: Base price
            direction: 'long' or 'short'
            action: 'entry' or 'exit'
        """
        spread_cost = SPREAD_PIPS * PIP_VALUE

        if direction == 'long':
            if action == 'entry':
                return price + spread_cost  # Buy at ask
            else:  # exit
                return price - spread_cost  # Sell at bid
        else:  # short
            if action == 'entry':
                return price - spread_cost  # Sell at bid
            else:  # exit
                return price + spread_cost  # Cover at ask

    def execute_trade(self, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                     direction: str, entry_price: float, exit_price: float):
        """Execute a trade and update capital."""

        # Apply spread
        entry_price_adj = self.apply_spread(entry_price, direction, 'entry')
        exit_price_adj = self.apply_spread(exit_price, direction, 'exit')

        # Calculate position size at entry
        position_size = self.calculate_position_size(entry_price_adj)

        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price_adj - entry_price_adj) * position_size
        else:  # short
            pnl = (entry_price_adj - exit_price_adj) * position_size

        pnl_pct = pnl / self.capital

        # Update capital
        self.capital += pnl
        self.equity_curve.append(self.capital)

        # Record trade
        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price_adj,
            exit_price=exit_price_adj,
            position_size=position_size,
            pnl=pnl,
            pnl_pct=pnl_pct
        )
        self.trades.append(trade)

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics from trades."""
        if not self.trades:
            return self._empty_metrics()

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)

        win_rate = num_wins / total_trades if total_trades > 0 else 0

        # P&L metrics
        total_pnl = sum(t.pnl for t in self.trades)
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        avg_win = gross_profit / num_wins if num_wins > 0 else 0
        avg_loss = gross_loss / num_losses if num_losses > 0 else 0

        risk_reward_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Return metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital

        # Calculate maximum drawdown
        peak = self.initial_capital
        max_drawdown = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        # Sharpe ratio (annualized, 5-min bars: 288 bars/day, 252 trading days)
        returns = [t.pnl_pct for t in self.trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 288)
        else:
            sharpe_ratio = 0

        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': total_trades,
            'num_wins': num_wins,
            'num_losses': num_losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_reward_ratio': risk_reward_ratio,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'sharpe_ratio': sharpe_ratio,
            'final_capital': self.capital
        }

    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict when no trades."""
        return {
            'total_return': 0, 'total_return_pct': 0, 'win_rate': 0,
            'profit_factor': 0, 'num_trades': 0, 'num_wins': 0,
            'num_losses': 0, 'avg_win': 0, 'avg_loss': 0,
            'risk_reward_ratio': 0, 'gross_profit': 0, 'gross_loss': 0,
            'max_drawdown': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0,
            'final_capital': self.initial_capital
        }


def is_in_london_session(timestamp: pd.Timestamp) -> bool:
    """Check if a timestamp is within the London trading session."""
    return LONDON_SESSION_START_HOUR <= timestamp.hour < LONDON_SESSION_END_HOUR


def backtest_method1_donchian(df: pd.DataFrame, lookback_period: int) -> Dict:
    """
    Method 1: Donchian Channel Breakout (with London Session filter)
    """
    engine = BacktestEngine()
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour

    df = calculate_donchian_channels(df, int(lookback_period))
    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        return engine.calculate_metrics()

    position = None
    entry_idx = None

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i-1]

        in_session = is_in_london_session(current['timestamp'])

        long_signal = current['close'] > previous['donchian_high']
        short_signal = current['close'] < previous['donchian_low']

        if position is None:
            if in_session and long_signal:
                position = 'long'
                entry_idx = i
            elif in_session and short_signal:
                position = 'short'
                entry_idx = i

        elif position == 'long' and short_signal:
            engine.execute_trade(
                entry_time=df.iloc[entry_idx]['timestamp'],
                exit_time=current['timestamp'],
                direction='long',
                entry_price=df.iloc[entry_idx]['close'],
                exit_price=current['close']
            )
            if in_session:
                position = 'short'
                entry_idx = i
            else:
                position = None
                entry_idx = None

        elif position == 'short' and long_signal:
            engine.execute_trade(
                entry_time=df.iloc[entry_idx]['timestamp'],
                exit_time=current['timestamp'],
                direction='short',
                entry_price=df.iloc[entry_idx]['close'],
                exit_price=current['close']
            )
            if in_session:
                position = 'long'
                entry_idx = i
            else:
                position = None
                entry_idx = None

    if position is not None and entry_idx is not None:
        engine.execute_trade(
            entry_time=df.iloc[entry_idx]['timestamp'],
            exit_time=df.iloc[-1]['timestamp'],
            direction=position,
            entry_price=df.iloc[entry_idx]['close'],
            exit_price=df.iloc[-1]['close']
        )

    metrics = engine.calculate_metrics()
    metrics['trades'] = engine.trades
    metrics['equity_curve'] = engine.equity_curve
    return metrics


def backtest_method2_atr(df: pd.DataFrame, atr_period: int, atr_multiplier: float) -> Dict:
    """
    Method 2: ATR Volatility Breakout (with London Session filter)
    """
    engine = BacktestEngine()
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour

    df = calculate_atr(df, int(atr_period))
    df['long_threshold'] = df['high'].shift(1) + (df['atr'] * atr_multiplier)
    df['short_threshold'] = df['low'].shift(1) - (df['atr'] * atr_multiplier)
    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        return engine.calculate_metrics()

    position = None
    entry_idx = None

    for i in range(1, len(df)):
        current = df.iloc[i]
        in_session = is_in_london_session(current['timestamp'])

        long_signal = current['close'] > current['long_threshold']
        short_signal = current['close'] < current['short_threshold']

        if position is None:
            if in_session and long_signal:
                position = 'long'
                entry_idx = i
            elif in_session and short_signal:
                position = 'short'
                entry_idx = i

        elif position == 'long' and short_signal:
            engine.execute_trade(
                entry_time=df.iloc[entry_idx]['timestamp'],
                exit_time=current['timestamp'],
                direction='long',
                entry_price=df.iloc[entry_idx]['close'],
                exit_price=current['close']
            )
            if in_session:
                position = 'short'
                entry_idx = i
            else:
                position = None
                entry_idx = None

        elif position == 'short' and long_signal:
            engine.execute_trade(
                entry_time=df.iloc[entry_idx]['timestamp'],
                exit_time=current['timestamp'],
                direction='short',
                entry_price=df.iloc[entry_idx]['close'],
                exit_price=current['close']
            )
            if in_session:
                position = 'long'
                entry_idx = i
            else:
                position = None
                entry_idx = None

    if position is not None and entry_idx is not None:
        engine.execute_trade(
            entry_time=df.iloc[entry_idx]['timestamp'],
            exit_time=df.iloc[-1]['timestamp'],
            direction=position,
            entry_price=df.iloc[entry_idx]['close'],
            exit_price=df.iloc[-1]['close']
        )

    metrics = engine.calculate_metrics()
    metrics['trades'] = engine.trades
    metrics['equity_curve'] = engine.equity_curve
    return metrics


def backtest_method3_volume(df: pd.DataFrame, lookback_period: int,
                           volume_threshold: float, volume_ma_period: int) -> Dict:
    """
    Method 3: Volume-Confirmed Channel Breakout (with London Session filter)
    """
    engine = BacktestEngine()
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour

    df = calculate_donchian_channels(df, int(lookback_period))
    df = calculate_volume_ma(df, int(volume_ma_period))
    df = df.dropna().reset_index(drop=True)

    if len(df) < 50:
        return engine.calculate_metrics()

    position = None
    entry_idx = None

    for i in range(1, len(df)):
        current = df.iloc[i]
        previous = df.iloc[i-1]
        in_session = is_in_london_session(current['timestamp'])

        volume_confirmed = current['volume'] > (current['volume_ma'] * volume_threshold)
        long_signal = (current['close'] > previous['donchian_high']) and volume_confirmed
        short_signal = (current['close'] < previous['donchian_low']) and volume_confirmed

        if position is None:
            if in_session and long_signal:
                position = 'long'
                entry_idx = i
            elif in_session and short_signal:
                position = 'short'
                entry_idx = i

        elif position == 'long' and short_signal:
            engine.execute_trade(
                entry_time=df.iloc[entry_idx]['timestamp'],
                exit_time=current['timestamp'],
                direction='long',
                entry_price=df.iloc[entry_idx]['close'],
                exit_price=current['close']
            )
            if in_session:
                position = 'short'
                entry_idx = i
            else:
                position = None
                entry_idx = None

        elif position == 'short' and long_signal:
            engine.execute_trade(
                entry_time=df.iloc[entry_idx]['timestamp'],
                exit_time=current['timestamp'],
                direction='short',
                entry_price=df.iloc[entry_idx]['close'],
                exit_price=current['close']
            )
            if in_session:
                position = 'long'
                entry_idx = i
            else:
                position = None
                entry_idx = None

    if position is not None and entry_idx is not None:
        engine.execute_trade(
            entry_time=df.iloc[entry_idx]['timestamp'],
            exit_time=df.iloc[-1]['timestamp'],
            direction=position,
            entry_price=df.iloc[entry_idx]['close'],
            exit_price=df.iloc[-1]['close']
        )

    metrics = engine.calculate_metrics()
    metrics['trades'] = engine.trades
    metrics['equity_curve'] = engine.equity_curve
    return metrics
