#!/usr/bin/env python3
"""
Data pipeline for downloading Binance BTC-USDT 1-minute data.
- REST back-fill → data/binance_1m/btcusdt_<year>.parquet (Snappy, monthly partitions)
- WebSocket append → live file btcusdt_current.parquet
"""

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from binance.client import Client
import pyarrow as pa
import pyarrow.parquet as pq
import structlog

logger = structlog.get_logger()

def setup_binance_client():
    """Initialize Binance client with API keys from env vars."""
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    return Client(api_key, api_secret)

def get_klines(client, symbol, interval, start_str, end_str=None):
    """Get historical klines/candlestick data."""
    klines = client.get_historical_klines(
        symbol=symbol,
        interval=interval,
        start_str=start_str,
        end_str=end_str,
        limit=1000
    )
    return pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])

def process_klines(df):
    """Process raw klines data into required format."""
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Convert string values to float
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']:
        df[col] = df[col].astype(float)
    
    # Calculate bid/ask size from taker data
    df['bid_size'] = df['volume'] * (1 - df['taker_buy_base'] / df['volume'])
    df['ask_size'] = df['volume'] - df['bid_size']
    
    # Calculate VWAP
    df['vwap'] = df['quote_asset_volume'] / df['volume']
    
    # Select and reorder columns
    return df[[
        'timestamp', 'open', 'high', 'low', 'close',
        'volume', 'bid_size', 'ask_size', 'quote_asset_volume', 'vwap'
    ]]

def download_year(client, symbol, year, output_dir):
    """Download and save one year of data."""
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    
    # Create monthly partitions
    schema = pa.schema([
        ('timestamp', pa.timestamp('ns')),
        ('open', pa.float64()),
        ('high', pa.float64()),
        ('low', pa.float64()),
        ('close', pa.float64()),
        ('volume', pa.float64()),
        ('bid_size', pa.float64()),
        ('ask_size', pa.float64()),
        ('quote_asset_volume', pa.float64()),
        ('vwap', pa.float64())
    ])
    
    partitioning = pa.dataset.partitioning(
        pa.schema([("month", pa.int32())]),
        flavor="hive"
    )
    
    # Download and process data month by month
    current = start
    while current < end:
        next_month = current + pd.offsets.MonthEnd(1)
        
        df = get_klines(
            client,
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            start_str=current.strftime('%Y-%m-%d'),
            end_str=next_month.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            logger.warning("No data for period", start=current, end=next_month)
            current = next_month + timedelta(days=1)
            continue
            
        # Process data
        df = process_klines(df)
        
        # Add month partition column
        df['month'] = df['timestamp'].dt.month
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df, schema=schema)
        
        # Save partition
        output_file = output_dir / f"btcusdt_{year}.parquet"
        pq.write_to_dataset(
            table,
            root_path=str(output_file),
            partition_cols=['month'],
            partitioning=partitioning,
            compression='snappy',
            existing_data_behavior='delete_matching'
        )
        
        logger.info(
            "Saved partition",
            year=year,
            month=current.month,
            rows=len(df)
        )
        
        current = next_month + timedelta(days=1)

def main():
    """Main execution function."""
    # Setup
    client = setup_binance_client()
    output_dir = Path("data/binance_1m")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download historical data year by year
    current_year = datetime.now().year
    for year in range(2020, current_year + 1):
        logger.info("Downloading year", year=year)
        download_year(client, "BTCUSDT", year, output_dir)
    
    logger.info("Download completed")

if __name__ == "__main__":
    main()
