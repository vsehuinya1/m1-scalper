#!/usr/bin/env python3
"""
Quick evaluation of the walk-forward optimization.
"""

import pandas as pd
import numpy as np
import structlog
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.regime import detect_regime
from strategy.entry import build_signal
from backtest.engine import BacktestEngine

logger = structlog.get_logger()

def load_data(start_date=None, end_date=None):
    data_dir = Path("data/binance_1m")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found. Run 'make download' first.")
    
    parquet_files = list(data_dir.glob("*.parquet"))
    dfs = []
    for f in parquet_files:
        dfs.append(pd.read_parquet(f))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]
    return df

def main():
    logger.info("Starting quick evaluation")
    
    # Load data with the required range
    df = load_data(start_date='2020-01-01', end_date='2024-12-31')
    # If no data for that range, use available data
    try:
        df = load_data(start_date='2020-01-01', end_date='2024-12-31')
    
    if len(df) == 0:
        logger.error("No data available for the specified date range.")
        return
    
    # Use the best parameters from the grid (first combo)
    best_params = {
        'ema_len': 15,
        'percentile_thr': 55,
        'bid_ask_ratio': 1.3,
        'range_pct': 0.25,
        'volume_mult': 1.05
    }
    
    # Run backtest with these parameters
    from strategy.regime import detect_regime
    from strategy.entry import build_signal
    from backtest.engine import BacktestEngine
    
    # Detect regime
    regime_params = {
        'trend_window': 20,
        'vol_window': 5,
        'vol_mult': 1.2,
        'vol_bins': 3
    }
    df = detect_regime(df, regime_params)
    
    # Generate signals
    signal_params = {
        'bid_ask_ratio': best_params['bid_ask_ratio'],
        'range_pct': best_params['range_pct']
    }
    long_signals, short_signals = build_signal(df, signal_params)
    
    # Run backtest engine
    engine = BacktestEngine(seed=42)
    results = engine.run(
        df,
        long_signals,
        short_signals,
        initial_capital=10000.0,
        risk_percent=0.002
    )
    
    # Extract results
    num_trades = results.get('num_trades', 0)
    sharpe = results.get('sharpe_ratio', 0.0)
    
    print(f"Best parameters: {best_params}")
    print(f"Trade count: {num_trades}")
    print(f"Sharpe ratio: {sharpe:.4f}")
    
    logger.info("Quick evaluation completed")

if __name__ == "__main__":
    main()
