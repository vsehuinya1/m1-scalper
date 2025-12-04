#!/usr/bin/env python3
"""
Entry signal generation for the BTC-USDT 1-minute scalper.
Pure, vectorized functions for core entry logic and signal composition.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def core_long(
    close: pd.Series,
    vwap: pd.Series,
    bid_size: pd.Series,
    ask_size: pd.Series,
    ratio: float = 1.8
) -> pd.Series:
    """
    Core long entry signal based on VWAP and order book imbalance.
    
    Args:
        close: Series of closing prices
        vwap: Series of volume-weighted average prices
        bid_size: Series of bid sizes
        ask_size: Series of ask sizes
        ratio: Required bid/ask size ratio (default 1.8)
    
    Returns:
        Boolean series of long entry signals
    """
    return (close > vwap) & (bid_size > ask_size * ratio)


def core_short(
    close: pd.Series,
    vwap: pd.Series,
    bid_size: pd.Series,
    ask_size: pd.Series,
    ratio: float = 1.8
) -> pd.Series:
    """
    Core short entry signal based on VWAP and order book imbalance.
    
    Args:
        close: Series of closing prices
        vwap: Series of volume-weighted average prices
        bid_size: Series of bid sizes
        ask_size: Series of ask sizes
        ratio: Required bid/ask size ratio (default 1.8)
    
    Returns:
        Boolean series of short entry signals
    """
    return (close < vwap) & (ask_size > bid_size * ratio)


def anti_chase(
    close: pd.Series,
    prev_range: pd.Series,
    pct: float = 0.3,
    side: str = 'long',
    low: pd.Series = None,
    high: pd.Series = None
) -> pd.Series:
    """
    Anti-chase filter to prevent entering after extended moves.
    
    Args:
        close: Series of closing prices
        prev_range: Previous bar's high-low range
        pct: Percentage of range to use as threshold (default 0.3)
        side: Entry direction ('long' or 'short')
        low: Series of low prices (required for long side)
        high: Series of high prices (required for short side)
    
    Returns:
        Boolean series indicating where entry is allowed
    """
    if side == 'long':
        if low is None:
            raise ValueError("low series required for long side anti-chase")
        return close <= low.shift(1) + pct * prev_range
    else:
        if high is None:
            raise ValueError("high series required for short side anti-chase")
        return close >= high.shift(1) - pct * prev_range


def build_signal(
    df: pd.DataFrame,
    params: dict
) -> Tuple[pd.Series, pd.Series]:
    """
    Construct final entry signals combining core signals, regime filter,
    and anti-chase logic.
    
    Args:
        df: DataFrame with OHLCV and regime data
        params: Dictionary of signal parameters:
            - bid_ask_ratio: Minimum bid/ask size ratio
            - range_pct: Percentage of range for anti-chase
    
    Returns:
        Tuple of (long_signals, short_signals) as boolean Series
    """
    # Extract parameters with defaults
    bid_ask_ratio = params.get('bid_ask_ratio', 1.8)
    range_pct = params.get('range_pct', 0.3)
    
    # Calculate previous bar range
    prev_range = df.high.shift(1) - df.low.shift(1)
    
    # Generate core signals
    core_long_signal = core_long(
        df.close,
        df.vwap,
        df.bid_size,
        df.ask_size,
        bid_ask_ratio
    )
    
    core_short_signal = core_short(
        df.close,
        df.vwap,
        df.bid_size,
        df.ask_size,
        bid_ask_ratio
    )
    
    # Apply regime filters
    regime_long = (df.regime == "low_vol_up")
    regime_short = (df.regime == "high_vol_down")
    
    # Apply anti-chase filters
    chase_long = anti_chase(df.close, prev_range, range_pct, 'long', low=df.low)
    chase_short = anti_chase(df.close, prev_range, range_pct, 'short', high=df.high)
    
    # Combine all conditions
    long_signal = core_long_signal & regime_long & chase_long
    short_signal = core_short_signal & regime_short & chase_short
    
    return long_signal, short_signal
