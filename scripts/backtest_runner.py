#!/usr/bin/env python3
"""
Backtest runner for BTC-USDT 1-minute scalper.
Loads data, runs regime detection, generates signals, and executes backtest.
"""

import sys
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path
import structlog

# Add parent directory to path to import strategy modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy.regime import detect_regime
from strategy.entry import build_signal
from backtest.engine import BacktestEngine

logger = structlog.get_logger()

def load_data(start_date=None, end_date=None) -> pd.DataFrame:
    """Load all downloaded data from parquet files, optionally filter by date range, and exclude 00-04 UTC."""
    data_dir = Path("data/binance_1m")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found. Run 'make download' first.")
    
    # Load all parquet files in the directory
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    
    dfs = []
    for file in parquet_files:
        df_chunk = pd.read_parquet(file)
        dfs.append(df_chunk)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Filter by date range if provided
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]
    
    # Exclude 00-04 UTC hours
    df = df[~df['timestamp'].dt.hour.isin([0, 1, 2, 3])]
    
    logger.info("Loaded data", rows=len(df), columns=df.columns.tolist())
    logger.info(f"Date range: {start_date} to {end_date}, excluded 00-04 UTC")
    return df

def main():
    """Main execution function."""
    logger.info("Starting backtest runner")
    
    # Load data
    df = load_data()
    
    # Detect regime with original MVP values
    regime_params = {
        'trend_window': 20,
        'vol_window': 5,
        'vol_mult': 1.2,
        'vol_bins': 3
    }
    df = detect_regime(df, regime_params)
    
    # Generate signals with original MVP values
    signal_params = {
        'bid_ask_ratio': 1.8,
        'range_pct': 0.3
    }
    long_signals, short_signals = build_signal(df, signal_params)
    
    # Convert timestamp to milliseconds integer (as expected by backtest engine)
    df = df.copy()
    df['timestamp'] = (df['timestamp'].astype('int64') // 10**6).astype('int64')  # nanoseconds to milliseconds
    
    # Ensure numeric columns are float64
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'bid_size', 'ask_size', 'quote_asset_volume', 'vwap']
    for col in numeric_cols:
        df[col] = df[col].astype('float64')
    
    # Run backtest
    engine = BacktestEngine()
    results = engine.run(
        df,
        long_signals,
        short_signals,
        initial_capital=10000.0,
        risk_percent=0.002
    )
    
    # Print equity curve (first 10 values)
    equity = results['equity']
    print("Equity curve (first 10 values):")
    for i in range(min(10, len(equity))):
        print(f"{i}: {equity[i]:.2f}")
    
    # Print summary
    print(f"\nTotal trades: {results['num_trades']}")
    print(f"Final equity: {results['final_equity']:.2f}")
    print(f"Total return: {results['total_return']*100:.2f}%")
    
    logger.info("Backtest runner completed")

if __name__ == "__main__":
    main()
