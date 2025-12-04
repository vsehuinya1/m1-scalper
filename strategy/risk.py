#!/usr/bin/env python3
"""
Risk management for BTC-USDT 1-minute scalper.
Pure, vectorized functions for position sizing and risk filters.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def position_size(
    equity: float,
    entry: float,
    stop: float,
    risk_pct: float = 0.002,
    min_size: float = 0.001,
    round_decimals: int = 3
) -> float:
    """
    Calculate position size based on fixed percentage risk.
    
    Args:
        equity: Current account equity
        entry: Entry price
        stop: Stop loss price
        risk_pct: Risk per trade as decimal (default 0.2%)
        min_size: Minimum position size in BTC (default 0.001)
        round_decimals: Round size to this many decimals
    
    Returns:
        Position size in BTC
    """
    risk_usd = equity * risk_pct
    size = risk_usd / abs(entry - stop)
    size = max(round(size, round_decimals), min_size)
    return size


def risk_filter(
    long: pd.Series,
    short: pd.Series,
    open_trades: pd.DataFrame,
    pnl: pd.Series,
    max_positions: int = 3,
    max_daily_loss: float = -0.01
) -> Tuple[pd.Series, pd.Series]:
    """
    Filter entry signals based on position and loss limits.
    
    Args:
        long: Long entry signals
        short: Short entry signals
        open_trades: DataFrame of open positions
        pnl: Series of cumulative daily PnL
        max_positions: Maximum concurrent positions allowed
        max_daily_loss: Maximum daily drawdown allowed (-1% default)
    
    Returns:
        Tuple of (filtered_long, filtered_short) boolean Series
    """
    # Position limit filter
    pos_count = len(open_trades)
    pos_filter = pos_count < max_positions
    
    # Daily loss limit filter 
    pnl_ok = pnl > max_daily_loss
    
    # Combine filters
    long_allowed = long & pos_filter & pnl_ok
    short_allowed = short & pos_filter & pnl_ok
    
    return long_allowed, short_allowed


def calculate_stop(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    volatility: pd.Series,
    side: str,
    atr_mult: float = 1.5
) -> pd.Series:
    """
    Calculate adaptive stop loss levels based on volatility.
    
    Args:
        close: Series of closing prices
        high: Series of high prices
        low: Series of low prices
        volume: Series of volume
        volatility: Series of volatility measures
        side: 'long' or 'short'
        atr_mult: ATR multiplier for stop distance
    
    Returns:
        Series of stop loss prices
    """
    # Use volatility-adjusted ATR for stop distance
    atr = volatility.rolling(20).mean()
    stop_distance = atr * atr_mult
    
    if side == 'long':
        # Long stop below entry
        stop = close - stop_distance
        # Round down to recent swing low if nearby
        recent_low = low.rolling(5).min()
        stop = np.where(
            (recent_low - stop).abs() < stop_distance,
            recent_low,
            stop
        )
    else:
        # Short stop above entry  
        stop = close + stop_distance
        # Round up to recent swing high if nearby
        recent_high = high.rolling(5).max()
        stop = np.where(
            (recent_high - stop).abs() < stop_distance,
            recent_high,
            stop
        )
    
    return pd.Series(stop, index=close.index)


def validate_risk_reward(
    entry: float,
    stop: float,
    side: str,
    min_r: float = 3.0
) -> bool:
    """
    Validate if trade setup meets minimum reward-to-risk ratio.
    
    Args:
        entry: Entry price
        stop: Stop loss price
        side: 'long' or 'short'
        min_r: Minimum reward-to-risk ratio required
    
    Returns:
        Boolean indicating if trade meets R:R requirement
    """
    risk = abs(entry - stop)
    if side == 'long':
        reward = (entry * (1 + min_r * (risk/entry))) - entry
    else:
        reward = entry - (entry * (1 - min_r * (risk/entry)))
        
    return (reward / risk) >= min_r
