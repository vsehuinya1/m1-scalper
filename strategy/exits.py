#!/usr/bin/env python3
"""
Scale-out exit logic for BTC-USDT 1-minute scalper.
Pure, vectorized functions for calculating take-profit and trailing stop levels.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ScaleOutLeg:
    """Data class representing a single leg of a scaled exit."""
    parent_id: str
    leg_id: str
    size: float
    entry: float
    stop: float
    target: Optional[float]
    trail_start: Optional[float]
    trail_offset: Optional[float]


def make_exit_prices(
    entry: float,
    stop: float,
    side: str,
    risk_multiple: float = 3.0
) -> Dict[str, float]:
    """
    Calculate take-profit and trailing stop trigger prices.
    
    Args:
        entry: Entry price
        stop: Initial stop loss price
        side: Trade direction ('long' or 'short')
        risk_multiple: Target reward-to-risk ratio (default 3.0)
    
    Returns:
        Dictionary of key price levels:
            - t2: 2R target price
            - t3: 3R target price
            - trail_trigger: Price to activate trailing stop
    """
    risk = abs(entry - stop)
    
    if side == 'long':
        t2 = entry + 2 * risk  # 2R target
        t3 = entry + 3 * risk  # 3R target
        trail_trigger = entry + 1 * risk  # 1R trigger for trailing
    else:
        t2 = entry - 2 * risk
        t3 = entry - 3 * risk
        trail_trigger = entry - 1 * risk
        
    return {
        't2': t2,
        't3': t3,
        'trail_trigger': trail_trigger
    }


def split_position(
    trade_id: str,
    size: float,
    entry: float,
    stop: float,
    side: str
) -> List[ScaleOutLeg]:
    """
    Split position into three legs with different exit strategies.
    
    Args:
        trade_id: Unique identifier for parent trade
        size: Total position size
        entry: Entry price
        stop: Initial stop loss price
        side: Trade direction ('long' or 'short')
    
    Returns:
        List of ScaleOutLeg objects with exit parameters
    """
    # Calculate target prices
    exits = make_exit_prices(entry, stop, side)
    
    # Split ratios (50% @ 2R, 25% @ 3R, 25% trailing)
    leg_sizes = [size * ratio for ratio in [0.5, 0.25, 0.25]]
    
    # Create legs
    legs = [
        # Leg 1: 50% @ 2R target
        ScaleOutLeg(
            parent_id=trade_id,
            leg_id=f"{trade_id}_1",
            size=leg_sizes[0],
            entry=entry,
            stop=stop,
            target=exits['t2'],
            trail_start=None,
            trail_offset=None
        ),
        # Leg 2: 25% @ 3R target
        ScaleOutLeg(
            parent_id=trade_id,
            leg_id=f"{trade_id}_2",
            size=leg_sizes[1],
            entry=entry,
            stop=stop,
            target=exits['t3'],
            trail_start=None,
            trail_offset=None
        ),
        # Leg 3: 25% trail from 1R
        ScaleOutLeg(
            parent_id=trade_id,
            leg_id=f"{trade_id}_3",
            size=leg_sizes[2],
            entry=entry,
            stop=stop,
            target=None,
            trail_start=exits['trail_trigger'],
            trail_offset=abs(entry - stop) * 0.5  # 0.5R offset
        )
    ]
    
    return legs


def update_trailing_stop(
    current_price: float,
    high: pd.Series,
    low: pd.Series,
    entry: float,
    stop: float,
    trail_start: float,
    trail_offset: float,
    side: str
) -> float:
    """
    Update trailing stop price based on price movement.
    
    Args:
        current_price: Current market price
        high: Series of high prices
        low: Series of low prices
        entry: Entry price
        stop: Current stop loss
        trail_start: Price level to start trailing
        trail_offset: Distance to maintain behind price
        side: Trade direction ('long' or 'short')
    
    Returns:
        Updated stop loss price
    """
    # Check if trailing should be active
    if side == 'long':
        if current_price < trail_start:
            return stop
        # Trail up with high prices
        new_stop = high.rolling(window=5).max() - trail_offset
        return max(stop, new_stop.iloc[-1])
    else:
        if current_price > trail_start:
            return stop
        # Trail down with low prices
        new_stop = low.rolling(window=5).min() + trail_offset
        return min(stop, new_stop.iloc[-1])


def check_exits(
    legs: List[ScaleOutLeg],
    current_price: float,
    high: pd.Series,
    low: pd.Series
) -> Tuple[List[str], List[ScaleOutLeg]]:
    """
    Check for exit triggers across all position legs.
    
    Args:
        legs: List of active ScaleOutLeg objects
        current_price: Current market price
        high: Series of high prices
        low: Series of low prices
    
    Returns:
        Tuple of (closed_leg_ids, updated_legs)
    """
    closed_legs = []
    updated_legs = []
    
    for leg in legs:
        # Check fixed targets
        if leg.target is not None:
            if (leg.side == 'long' and current_price >= leg.target) or \
               (leg.side == 'short' and current_price <= leg.target):
                closed_legs.append(leg.leg_id)
                continue
                
        # Update trailing stops
        if leg.trail_start is not None:
            new_stop = update_trailing_stop(
                current_price,
                high,
                low,
                leg.entry,
                leg.stop,
                leg.trail_start,
                leg.trail_offset,
                leg.side
            )
            
            # Check if stopped out
            if (leg.side == 'long' and current_price <= new_stop) or \
               (leg.side == 'short' and current_price >= new_stop):
                closed_legs.append(leg.leg_id)
            else:
                # Update stop and keep leg active
                leg.stop = new_stop
                updated_legs.append(leg)
        else:
            # Check fixed stops
            if (leg.side == 'long' and current_price <= leg.stop) or \
               (leg.side == 'short' and current_price >= leg.stop):
                closed_legs.append(leg.leg_id)
            else:
                updated_legs.append(leg)
                
    return closed_legs, updated_legs
