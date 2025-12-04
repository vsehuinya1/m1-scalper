#!/usr/bin/env python3
"""
Numba-accelerated backtest engine for BTC-USDT 1-minute scalper.
Single-pass simulation with realistic costs.
"""

import numpy as np
from numba import jit, float64, boolean, int64
import pandas as pd
from typing import Dict, Tuple, List
import structlog


@jit(nopython=True)
def calculate_returns(
    prices: np.ndarray,
    signals: np.ndarray,
    commissions: float = 0.00017,
    slippage_min: int = 1,
    slippage_max: int = 3,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate trade returns with commissions and slippage.
    
    Args:
        prices: Array of close prices
        signals: Array of trade signals (-1 for short, 0 for neutral, 1 for long)
        commissions: Commission rate per trade (default 0.017%)
        slippage_min: Minimum slippage in ticks
        slippage_max: Maximum slippage in ticks
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (returns array, positions array)
    """
    np.random.seed(seed)
    n = len(prices)
    returns = np.zeros(n)
    positions = np.zeros(n)
    position = 0
    entry_price = 0.0
    
    # Price per tick (assuming BTC price ~$60k, tick size = $0.01)
    tick_value = 0.01
    
    for i in range(1, n):
        # Check for signal change
        if signals[i] != signals[i-1]:
            # Close previous position
            if position != 0:
                # Calculate exit price with slippage
                slippage = np.random.randint(slippage_min, slippage_max + 1)
                exit_price = prices[i] + (tick_value * slippage * (-1 if position > 0 else 1))
                pnl = (exit_price - entry_price) * position
                returns[i] = pnl - abs(position) * exit_price * commissions
                position = 0
                entry_price = 0.0
            
            # Open new position
            if signals[i] != 0:
                slippage = np.random.randint(slippage_min, slippage_max + 1)
                entry_price = prices[i] + (tick_value * slippage * (1 if signals[i] > 0 else -1))
                position = signals[i]  # 1 for long, -1 for short
                returns[i] = -abs(position) * entry_price * commissions
        
        # Track current position
        positions[i] = position
    
    return returns, positions


@jit(nopython=True)
def backtest_single_pass(
    data: np.ndarray,
    long_signals: np.ndarray,
    short_signals: np.ndarray,
    initial_capital: float = 10000.0,
    risk_percent: float = 0.002,
    max_positions: int = 3,
    max_daily_loss: float = -0.01
) -> Dict:
    """
    Single-pass backtest engine optimized with Numba.
    
    Args:
        data: 2D array with columns: [timestamp, open, high, low, close, volume]
        long_signals: Boolean array for long entries
        short_signals: Boolean array for short entries
        initial_capital: Starting capital
        risk_percent: Risk per trade percentage
        max_positions: Maximum concurrent positions
        max_daily_loss: Maximum daily loss limit
    
    Returns:
        Dictionary with backtest results
    """
    n = len(data)
    equity = np.zeros(n)
    equity[0] = initial_capital
    daily_pnl = 0.0
    
    # Extract price data
    closes = data[:, 4]  # Close prices
    
    # Initialize tracking arrays
    active_trades = []
    daily_returns = np.zeros(n)
    trades = []
    position_sizes = np.zeros(n)
    
    for i in range(1, n):
        # Check for new day
        if i > 0 and data[i, 0] // 86400000 != data[i-1, 0] // 86400000:
            daily_pnl = 0.0
        
        # Update equity from previous trades
        equity[i] = equity[i-1]
        
        # Check if we can open new positions
        can_trade = (len(active_trades) < max_positions) and (daily_pnl > max_daily_loss)
        
        # Check for entry signals
        if can_trade:
            if long_signals[i]:
                # Calculate position size
                risk_usd = equity[i] * risk_percent
                entry_price = closes[i]
                stop_price = closes[i] * 0.995  # 0.5% stop for example
                size = risk_usd / abs(entry_price - stop_price)
                size = round(size, 3)
                
                # Record trade
                trade = {
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'size': size,
                    'side': 1  # long
                }
                active_trades.append(trade)
                position_sizes[i] += size
                
            elif short_signals[i]:
                # Calculate position size
                risk_usd = equity[i] * risk_percent
                entry_price = closes[i]
                stop_price = closes[i] * 1.005  # 0.5% stop for example
                size = risk_usd / abs(entry_price - stop_price)
                size = round(size, 3)
                
                # Record trade
                trade = {
                    'entry_idx': i,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'size': size,
                    'side': -1  # short
                }
                active_trades.append(trade)
                position_sizes[i] -= size
        
        # Update active trades
        to_remove = []
        for trade in active_trades:
            # Check for exit conditions
            exit_condition = False
            
            if trade['side'] == 1:  # long
                # Check stop loss
                if closes[i] <= trade['stop_price']:
                    exit_condition = True
                # Check take profit (example: 2R)
                elif closes[i] >= trade['entry_price'] + 2 * abs(trade['entry_price'] - trade['stop_price']):
                    exit_condition = True
            else:  # short
                # Check stop loss
                if closes[i] >= trade['stop_price']:
                    exit_condition = True
                # Check take profit (example: 2R)
                elif closes[i] <= trade['entry_price'] - 2 * abs(trade['entry_price'] - trade['stop_price']):
                    exit_condition = True
            
            if exit_condition:
                # Calculate P&L
                if trade['side'] == 1:
                    pnl = (closes[i] - trade['entry_price']) * trade['size']
                else:
                    pnl = (trade['entry_price'] - closes[i]) * trade['size']
                
                # Apply commissions (0.017%)
                pnl -= abs(trade['size']) * closes[i] * 0.00017 * 2  # Entry and exit
                
                # Update equity
                equity[i] += pnl
                daily_pnl += pnl / equity[i-1] if equity[i-1] > 0 else 0
                
                # Record trade
                trades.append({
                    'entry_idx': trade['entry_idx'],
                    'exit_idx': i,
                    'entry_price': trade['entry_price'],
                    'exit_price': closes[i],
                    'size': trade['size'],
                    'side': trade['side'],
                    'pnl': pnl,
                    'pnl_pct': pnl / (trade['entry_price'] * trade['size'])
                })
                
                to_remove.append(trade)
                position_sizes[i] -= trade['size'] if trade['side'] == 1 else -trade['size']
        
        # Remove closed trades
        for trade in to_remove:
            active_trades.remove(trade)
    
    # Close any remaining positions at final price
    for trade in active_trades:
        if trade['side'] == 1:
            pnl = (closes[-1] - trade['entry_price']) * trade['size']
        else:
            pnl = (trade['entry_price'] - closes[-1]) * trade['size']
        
        pnl -= abs(trade['size']) * closes[-1] * 0.00017 * 2
        equity[-1] += pnl
        
        trades.append({
            'entry_idx': trade['entry_idx'],
            'exit_idx': n-1,
            'entry_price': trade['entry_price'],
            'exit_price': closes[-1],
            'size': trade['size'],
            'side': trade['side'],
            'pnl': pnl,
            'pnl_pct': pnl / (trade['entry_price'] * trade['size'])
        })
    
    # Calculate returns
    returns = np.diff(equity) / equity[:-1]
    returns = np.concatenate(([0], returns))
    
    return {
        'equity': equity,
        'returns': returns,
        'trades': trades,
        'position_sizes': position_sizes,
        'num_trades': len(trades),
        'final_equity': equity[-1],
        'total_return': equity[-1] / initial_capital - 1
    }


class BacktestEngine:
    """Main backtest engine class."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.logger = structlog.get_logger()
    
    def run(
        self,
        df: pd.DataFrame,
        long_signals: pd.Series,
        short_signals: pd.Series,
        initial_capital: float = 10000.0,
        risk_percent: float = 0.002
    ) -> Dict:
        """
        Run backtest on provided data and signals.
        
        Args:
            df: DataFrame with OHLCV data
            long_signals: Boolean series for long entries
            short_signals: Boolean series for short entries
            initial_capital: Starting capital
            risk_percent: Risk per trade percentage
        
        Returns:
            Dictionary with backtest results
        """
        # Convert to numpy arrays for Numba
        data_array = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values
        long_array = long_signals.values.astype(np.int8)
        short_array = short_signals.values.astype(np.int8)
        
        # Combine signals (1 for long, -1 for short, 0 for neutral)
        signals = np.zeros(len(df), dtype=np.int8)
        signals[long_array == 1] = 1
        signals[short_array == 1] = -1
        
        # Run backtest
        self.logger.info("Starting backtest", bars=len(df))
        results = backtest_single_pass(
            data_array,
            long_signals.values,
            short_signals.values,
            initial_capital,
            risk_percent
        )
        
        self.logger.info(
            "Backtest completed",
            trades=results['num_trades'],
            final_equity=results['final_equity'],
            total_return=f"{results['total_return']*100:.2f}%"
        )
        
        return results
