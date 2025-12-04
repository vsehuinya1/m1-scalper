#!/usr/bin/env python3
"""
Paper trading engine for BTC-USDT 1-minute scalper.
Real-time simulation with WebSocket data feed.
"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

import pandas as pd
from binance import AsyncClient, BinanceSocketManager
import structlog

from strategy.entry import build_signal
from strategy.risk import position_size, risk_filter
from strategy.exits import split_position
from strategy.regime import detect_regime


class PaperTradeEngine:
    """Paper trading engine for real-time simulation."""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = Decimal(str(initial_capital))
        self.capital = self.initial_capital
        self.positions = []
        self.trade_history = []
        self.current_prices = {}
        self.logger = structlog.get_logger()
        
        # Trading parameters
        self.params = {
            'bid_ask_ratio': 1.8,
            'range_pct': 0.3,
            'volume_mult': 1.2,
            'ema_len': 20,
            'percentile_thr': 60,
            'trend_window': 20,
            'vol_window': 5,
            'vol_bins': 3
        }
        
        # Risk parameters
        self.risk_params = {
            'max_positions': 3,
            'max_daily_loss': -0.01,
            'risk_pct': 0.002
        }
    
    async def connect_to_binance(self, api_key: str = "", api_secret: str = ""):
        """Connect to Binance WebSocket."""
        self.client = await AsyncClient.create(api_key, api_secret)
        self.bm = BinanceSocketManager(self.client)
        
        # Subscribe to BTCUSDT 1-minute klines
        self.conn_key = self.bm.kline_socket('BTCUSDT', AsyncClient.KLINE_INTERVAL_1MINUTE)
        
        self.logger.info("Connected to Binance WebSocket")
    
    async def process_candle(self, candle_data: Dict):
        """Process incoming candle data."""
        # Parse candle data
        candle = candle_data['k']
        timestamp = pd.Timestamp(candle['t'], unit='ms')
        
        # Create dataframe row
        df_row = pd.DataFrame({
            'timestamp': [timestamp],
            'open': [Decimal(candle['o'])],
            'high': [Decimal(candle['h'])],
            'low': [Decimal(candle['l'])],
            'close': [Decimal(candle['c'])],
            'volume': [Decimal(candle['v'])],
            'quote_asset_volume': [Decimal(candle['q'])],
            'taker_buy_base': [Decimal(candle['V'])],
            'taker_buy_quote': [Decimal(candle['Q'])]
        })
        
        # Calculate VWAP
        df_row['vwap'] = df_row['quote_asset_volume'] / df_row['volume']
        
        # Calculate bid/ask size (simplified from taker data)
        df_row['bid_size'] = df_row['volume'] * (Decimal('1') - df_row['taker_buy_base'] / df_row['volume'])
        df_row['ask_size'] = df_row['volume'] - df_row['bid_size']
        
        # Update current prices
        self.current_prices['BTCUSDT'] = {
            'bid': Decimal(candle['c']) * Decimal('0.9999'),  # Simulated bid
            'ask': Decimal(candle['c']) * Decimal('1.0001'),  # Simulated ask
            'last': Decimal(candle['c'])
        }
        
        # Add to rolling window (in production, would maintain a DataFrame buffer)
        await self.update_position(df_row)
    
    async def update_position(self, df_row: pd.DataFrame):
        """Update positions based on new price data."""
        # Detect regime
        regime_df = detect_regime(df_row, self.params)
        
        # Generate signals
        long_signals, short_signals = build_signal(regime_df, self.params)
        
        # Apply risk filters
        open_trades_df = pd.DataFrame(self.positions)
        daily_pnl = self.calculate_daily_pnl()
        
        filtered_long, filtered_short = risk_filter(
            long_signals,
            short_signals,
            open_trades_df,
            daily_pnl,
            self.risk_params['max_positions'],
            self.risk_params['max_daily_loss']
        )
        
        # Check for entry opportunities
        if filtered_long.iloc[0]:
            await self.enter_position('long', df_row)
        elif filtered_short.iloc[0]:
            await self.enter_position('short', df_row)
        
        # Update existing positions
        await self.update_existing_positions(df_row)
    
    async def enter_position(self, side: str, df_row: pd.DataFrame):
        """Enter a new position."""
        price = df_row['close'].iloc[0]
        stop_price = price * (Decimal('0.995') if side == 'long' else Decimal('1.005'))
        
        # Calculate position size
        size = position_size(
            float(self.capital),
            float(price),
            float(stop_price),
            self.risk_params['risk_pct']
        )
        
        if size <= 0:
            return
        
        # Create trade
        trade_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{side}"
        
        trade = {
            'id': trade_id,
            'entry_time': datetime.now(),
            'side': side,
            'entry_price': price,
            'stop_price': stop_price,
            'size': Decimal(str(size)),
            'status': 'open',
            'legs': []
        }
        
        # Split into scale-out legs
        legs = split_position(trade_id, size, float(price), float(stop_price), side)
        
        for leg in legs:
            leg_dict = {
                'leg_id': leg.leg_id,
                'size': Decimal(str(leg.size)),
                'target_price': Decimal(str(leg.target)) if leg.target else None,
                'trail_start': Decimal(str(leg.trail_start)) if leg.trail_start else None,
                'trail_offset': Decimal(str(leg.trail_offset)) if leg.trail_offset else None,
                'status': 'open'
            }
            trade['legs'].append(leg_dict)
        
        self.positions.append(trade)
        self.logger.info(
            "Entered position",
            trade_id=trade_id,
            side=side,
            size=size,
            price=float(price)
        )
    
    async def update_existing_positions(self, df_row: pd.DataFrame):
        """Update and check exit conditions for existing positions."""
        current_price = df_row['close'].iloc[0]
        
        for trade in self.positions:
            if trade['status'] != 'open':
                continue
            
            for leg in trade['legs']:
                if leg['status'] != 'open':
                    continue
                
                # Check target exit
                if leg['target_price']:
                    if (trade['side'] == 'long' and current_price >= leg['target_price']) or \
                       (trade['side'] == 'short' and current_price <= leg['target_price']):
                        await self.exit_leg(trade, leg, current_price, 'target')
                
                # Check trailing stop
                elif leg['trail_start']:
                    # Update trailing stop (simplified)
                    if trade['side'] == 'long':
                        trail_price = max(leg['trail_start'], current_price - leg['trail_offset'])
                    else:
                        trail_price = min(leg['trail_start'], current_price + leg['trail_offset'])
                    
                    if (trade['side'] == 'long' and current_price <= trail_price) or \
                       (trade['side'] == 'short' and current_price >= trail_price):
                        await self.exit_leg(trade, leg, current_price, 'trail')
                
                # Check stop loss
                elif current_price <= trade['stop_price'] if trade['side'] == 'long' else \
                     current_price >= trade['stop_price']:
                    await self.exit_leg(trade, leg, current_price, 'stop')
    
    async def exit_leg(self, trade: Dict, leg: Dict, exit_price: Decimal, reason: str):
        """Exit a position leg."""
        # Calculate P&L
        if trade['side'] == 'long':
            pnl = (exit_price - trade['entry_price']) * leg['size']
        else:
            pnl = (trade['entry_price'] - exit_price) * leg['size']
        
        # Apply commission (0.017%)
        commission = leg['size'] * exit_price * Decimal('0.00017')
        net_pnl = pnl - commission
        
        # Update capital
        self.capital += net_pnl
        
        # Update leg status
        leg['status'] = 'closed'
        leg['exit_price'] = exit_price
        leg['exit_time'] = datetime.now()
        leg['pnl'] = net_pnl
        leg['exit_reason'] = reason
        
        self.logger.info(
            "Exited leg",
            trade_id=trade['id'],
            leg_id=leg['leg_id'],
            reason=reason,
            pnl=float(net_pnl)
        )
        
        # Check if all legs are closed
        if all(l['status'] == 'closed' for l in trade['legs']):
            trade['status'] = 'closed'
            trade['exit_time'] = datetime.now()
            self.trade_history.append(trade)
            self.positions.remove(trade)
    
    def calculate_daily_pnl(self) -> pd.Series:
        """Calculate daily P&L for risk filtering."""
        today = datetime.now().date()
        daily_trades = [t for t in self.trade_history 
                       if t.get('exit_time', datetime.now()).date() == today]
        
        daily_pnl = sum(float(t.get('total_pnl', 0)) for t in daily_trades)
        
        # Create series for compatibility with risk_filter
        return pd.Series([daily_pnl])
    
    def get_statistics(self) -> Dict:
        """Get paper trading statistics."""
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history 
                            if sum(float(l['pnl']) for l in t['legs'] if 'pnl' in l) > 0])
        
        total_pnl = sum(sum(float(l['pnl']) for l in t['legs'] if 'pnl' in l) 
                       for t in self.trade_history)
        
        return {
            'initial_capital': float(self.initial_capital),
            'current_capital': float(self.capital),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'return_pct': (float(self.capital) / float(self.initial_capital) - 1) * 100,
            'open_positions': len(self.positions)
        }
    
    async def run(self):
        """Main paper trading loop."""
        try:
            # Start WebSocket connection
            await self.connect_to_binance()
            
            async with self.conn_key as stream:
                while True:
                    response = await stream.recv()
                    
                    if response['e'] == 'kline':
                        await self.process_candle(response)
                    
                    # Log statistics every 100 candles
                    if len(self.trade_history) % 100 == 0:
                        stats = self.get_statistics()
                        self.logger.info("Paper trading statistics", **stats)
                    
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            self.logger.info("Paper trading stopped")
        finally:
            if hasattr(self, 'client'):
                await self.client.close_connection()


async def main():
    """Main entry point."""
    engine = PaperTradeEngine(initial_capital=10000.0)
    
    try:
        await engine.run()
    except KeyboardInterrupt:
        engine.logger.info("Paper trading interrupted by user")


if __name__ == "__main__":
    asyncio.run(main())
