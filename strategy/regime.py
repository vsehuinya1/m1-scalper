#!/usr/bin/env python3
"""
Regime detection for BTC-USDT 1-minute scalper.
Pure, vectorized functions for market regime classification.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def rolling_percentile(
    series: pd.Series,
    window: int,
    percentile: float,
    min_periods: int = None
) -> pd.Series:
    """
    Calculate rolling percentile with NumPy backend.
    
    Args:
        series: Input data series
        window: Rolling window size
        percentile: Percentile to calculate (0-100)
        min_periods: Minimum periods required for calculation
    
    Returns:
        Series of rolling percentile values
    """
    if min_periods is None:
        min_periods = window
        
    return series.rolling(window, min_periods=min_periods).apply(
        lambda x: np.nanpercentile(x, percentile), raw=True
    )


def micro_trend(close: pd.Series, open_: pd.Series) -> pd.Series:
    """
    Calculate micro-trend with volatility filter.
    
    Args:
        close: Series of closing prices
        open_: Series of opening prices
    
    Returns:
        Series of normalized trend values (positive for uptrend,
        negative for downtrend, zero for neutral)
    """
    # Raw price change
    raw = (close - open_) / open_
    
    # Rolling volatility (5-period standard deviation)
    vol = close.pct_change().rolling(5).std()
    
    # Trend calculation with 20-period rolling mean
    trend = raw.rolling(20).mean().fillna(0)
    
    # Apply volatility gate: set trend to zero when volatility is high
    vol_threshold = vol.rolling(20).quantile(0.6)
    return trend.where(vol < vol_threshold, 0)


def volume_gate(
    volume: pd.Series,
    mult: float,
    min_bars: int = 20
) -> pd.Series:
    """
    Filter for volume confirmation using median comparison.
    
    Args:
        volume: Series of volume values
        mult: Multiplier for median threshold
        min_bars: Minimum bars for rolling median calculation
    
    Returns:
        Boolean series where volume is above threshold for 2 consecutive bars
    """
    # Calculate rolling median volume
    med = volume.rolling(min_bars, min_periods=min_bars).median()
    
    # Check if volume exceeds threshold
    above = volume > med * mult
    
    # Require 2-bar confirmation
    return above & above.shift(1).fillna(False)


def classify(
    trend: pd.Series,
    vol_cat: pd.Series,
    vol_ok: pd.Series
) -> pd.Series:
    """
    Classify market regime based on trend, volatility, and volume.
    
    Args:
        trend: Series of trend values from micro_trend()
        vol_cat: Series of volatility categories ('low', 'mid', 'high')
        vol_ok: Series of volume gate booleans
    
    Returns:
        Series of regime labels
    """
    # Initialize with neutral
    regime = pd.Series('neutral', index=trend.index)
    
    # Define classification rules
    mask_up = (np.sign(trend) == 1) & (vol_cat == 'low') & vol_ok
    mask_down = (np.sign(trend) == -1) & (vol_cat == 'high') & vol_ok
    
    regime[mask_up] = 'low_vol_up'
    regime[mask_down] = 'high_vol_down'
    
    return regime


def detect_regime(
    df: pd.DataFrame,
    params: dict
) -> pd.DataFrame:
    """
    Main regime detection pipeline.
    
    Args:
        df: DataFrame with OHLCV data
        params: Dictionary of parameters:
            - trend_window: Window for trend calculation (default 20)
            - vol_window: Window for volatility calculation (default 5)
            - vol_mult: Volume multiplier for gate (default 1.2)
            - vol_bins: Number of volatility bins (default 3)
    
    Returns:
        DataFrame with added 'regime' column
    """
    # Extract parameters with defaults
    trend_window = params.get('trend_window', 20)
    vol_window = params.get('vol_window', 5)
    vol_mult = params.get('vol_mult', 1.2)
    vol_bins = params.get('vol_bins', 3)
    
    # Calculate volatility (rolling standard deviation of returns)
    volatility = df.close.pct_change().rolling(vol_window).std()
    
    # Calculate micro trend
    trend = micro_trend(df.close, df.open)
    
    # Apply volume gate
    vol_ok = volume_gate(df.volume, vol_mult)
    
    # Categorize volatility into bins
    vol_cat = pd.cut(
        volatility,
        bins=vol_bins,
        labels=['low', 'mid', 'high'],
        include_lowest=True
    )
    
    # Classify regime
    regime = classify(trend, vol_cat, vol_ok)
    
    # Add results to dataframe
    result = df.copy()
    result['volatility'] = volatility
    result['trend'] = trend
    result['vol_cat'] = vol_cat
    result['vol_ok'] = vol_ok
    result['regime'] = regime
    
    return result
