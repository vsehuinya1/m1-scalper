#!/usr/bin/env python3
"""
Numba-accelerated backtest engine for BTC-USDT 1-minute scalper.
Single-pass simulation with realistic costs (0.017% commission + uniform(1,3) tick slippage).
Target: 400k bars <1.5s on M1 Air.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import structlog
from numba import njit, jit
import numba as nb


@njit(fastmath=True)
def calculate_sharpe_ratio_numba(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: float = 252 * 24 * 60  # 1-minute bars
) -> float:
    """
    Calculate Sharpe ratio for returns series.
    
    Args:
        returns: Array of periodic returns
        risk_free_rate: Risk-free rate per period
        annualization: Factor to annualize the ratio
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    # Calculate sample standard deviation (ddof=1)
    n = len(excess_returns)
    if n == 1:
        return 0.0
    variance = np.sum((excess_returns - mean_excess) ** 2) / (n - 1)
    std_excess = np.sqrt(variance)
    
    if std_excess == 0:
        return 0.0
    
    return mean_excess / std_excess * np.sqrt(annualization)


@njit(fastmath=True)
def calculate_max_drawdown_numba(
    equity_curve: np.ndarray
) -> float:
    """
    Calculate maximum drawdown from equity curve.
    
    Args:
        equity_curve: Array of equity values
    
    Returns:
        Maximum drawdown as negative percentage
    """
    if len(equity_curve) == 0:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for i in range(1, len(equity_curve)):
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        
        dd = (equity_curve[i] - peak) / peak
        if dd < max_dd:
            max_dd = dd
    
    return max_dd


@njit(fastmath=True)
def simulate_core(
    close_prices: np.ndarray,
    long_signals: np.ndarray,
    short_signals: np.ndarray,
    initial_capital: float = 10000.0,
    commission: float = 0.00017,
    slippage_min: int = 1,
    slippage_max: int = 3,
    seed: int = 42
) -> np.ndarray:
    """
    Core simulation function with Numba JIT.
    
    Args:
        close_prices: Array of closing prices
        long_signals: Boolean array for long entries
        short_signals: Boolean array for short entries
        initial_capital: Starting capital
        commission: Commission per trade (0.017%)
        slippage_min: Minimum slippage in ticks
        slippage_max: Maximum slippage in ticks
        seed: Random seed for reproducibility
    
    Returns:
        Array of equity values
    """
    np.random.seed(seed)
    n = len(close_prices)
    equity = np.zeros(n)
    equity[0] = initial_capital
    
    position = 0.0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    tick_value = 0.01  # $0.01 per tick for BTC
    
    for i in range(1, n):
        # Copy previous equity
        equity[i] = equity[i-1]
        
        # Check for position exit (stop/target logic is handled elsewhere)
        # For simplicity, we'll assume position held until opposite signal
        
        # Check for new position entry
        if position == 0:
            if long_signals[i]:
                # Apply slippage on entry
                slippage_ticks = np.random.randint(slippage_min, slippage_max + 1)
                entry_price = close_prices[i] + (tick_value * slippage_ticks)
                position = 1.0
                # Pay commission
                equity[i] -= entry_price * commission
                
            elif short_signals[i]:
                slippage_ticks = np.random.randint(slippage_min, slippage_max + 1)
                entry_price = close_prices[i] - (tick_value * slippage_ticks)
                position = -1.0
                equity[i] -= entry_price * commission
        
        # Update equity for open position
        elif position == 1:  # long
            current_price = close_prices[i]
            equity[i] += (current_price - entry_price) * 1.0  # position size = 1 for simplicity
        elif position == -1:  # short
            current_price = close_prices[i]
            equity[i] += (entry_price - current_price) * 1.0
        
        # Exit position on opposite signal (simplified)
        if position == 1 and short_signals[i]:
            # Exit long
            slippage_ticks = np.random.randint(slippage_min, slippage_max + 1)
            exit_price = close_prices[i] - (tick_value * slippage_ticks)
            equity[i] += (exit_price - entry_price) * 1.0 - exit_price * commission
            position = 0
            
        elif position == -1 and long_signals[i]:
            # Exit short
            slippage_ticks = np.random.randint(slippage_min, slippage_max + 1)
            exit_price = close_prices[i] + (tick_value * slippage_ticks)
            equity[i] += (entry_price - exit_price) * 1.0 - exit_price * commission
            position = 0
    
    return equity


def simulate(
    close: pd.Series,
    long_signals: pd.Series,
    short_signals: pd.Series,
    initial_capital: float = 10000.0,
    commission: float = 0.00017,
    slippage_min: int = 1,
    slippage_max: int = 3,
    seed: int = 42
) -> pd.Series:
    """
    Simulate trading with given signals, returning equity curve.
    
    Args:
        close: Series of closing prices
        long_signals: Boolean series for long entries
        short_signals: Boolean series for short entries
        initial_capital: Starting capital
        commission: Commission per trade (0.017%)
        slippage_min: Minimum slippage in ticks
        slippage_max: Maximum slippage in ticks
        seed: Random seed for reproducibility
    
    Returns:
        Series of equity values
    """
    equity = simulate_core(
        close.values,
        long_signals.values.astype(np.bool_),
        short_signals.values.astype(np.bool_),
        initial_capital,
        commission,
        slippage_min,
        slippage_max,
        seed
    )
    return pd.Series(equity, index=close.index)


# Python wrapper functions that accept pandas Series or numpy arrays
def calculate_sharpe_ratio(
    returns,
    risk_free_rate: float = 0.0,
    annualization: float = 252 * 24 * 60
) -> float:
    """Calculate Sharpe ratio for returns series (pandas wrapper)."""
    if isinstance(returns, pd.Series):
        values = returns.values
    else:
        values = returns
    return calculate_sharpe_ratio_numba(
        values,
        risk_free_rate,
        annualization
    )


def calculate_max_drawdown(
    equity_curve
) -> float:
    """Calculate maximum drawdown from equity curve (pandas wrapper)."""
    if isinstance(equity_curve, pd.Series):
        values = equity_curve.values
    else:
        values = equity_curve
    return calculate_max_drawdown_numba(values)


# Export functions for direct use
__all__ = ['calculate_sharpe_ratio', 'calculate_max_drawdown', 'simulate']


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
        risk_percent: float = 0.002,
        max_positions: int = 3,
        max_daily_loss: float = -0.01
    ) -> Dict:
        """
        Run backtest on provided data and signals.
        
        Args:
            df: DataFrame with OHLCV data
            long_signals: Boolean series for long entries
            short_signals: Boolean series for short entries
            initial_capital: Starting capital
            risk_percent: Risk per trade percentage
            max_positions: Maximum concurrent positions
            max_daily_loss: Maximum daily loss limit
        
        Returns:
            Dictionary with backtest results
        """
        # Extract price data
        close_prices = df['close'].values
        long_arr = long_signals.values.astype(np.bool_)
        short_arr = short_signals.values.astype(np.bool_)
        
        # Run simulation
        self.logger.info("Starting backtest", bars=len(df))
        
        equity = simulate_core(
            close_prices,
            long_arr,
            short_arr,
            initial_capital,
            commission=0.00017,
            slippage_min=1,
            slippage_max=3,
            seed=self.seed
        )
        
        # Calculate returns
        returns = np.diff(equity) / equity[:-1]
        returns = np.concatenate(([0], returns))
        
        # Calculate statistics
        total_return = equity[-1] / initial_capital - 1
        sharpe = calculate_sharpe_ratio(returns)
        max_dd = calculate_max_drawdown(equity)
        
        # Count trades (simplified)
        position_changes = np.diff(long_arr.astype(np.int8) - short_arr.astype(np.int8))
        num_trades = np.sum(np.abs(position_changes)) // 2
        
        self.logger.info(
            "Backtest completed",
            trades=num_trades,
            final_equity=equity[-1],
            total_return=f"{total_return*100:.2f}%",
            sharpe=f"{sharpe:.2f}",
            max_dd=f"{max_dd*100:.2f}%"
        )
        
        return {
            'equity': pd.Series(equity, index=df.index),
            'returns': pd.Series(returns, index=df.index),
            'num_trades': num_trades,
            'final_equity': equity[-1],
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': 0.0,  # Would need trade-level tracking
            'avg_r': 0.0,     # Would need trade-level tracking
            'trades': []      # Empty for now
        }
