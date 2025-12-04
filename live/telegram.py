#!/usr/bin/env python3
"""
Telegram reporting for BTC-USDT 1-minute scalper.
Send daily summaries, trade alerts, and system status updates.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import aiohttp
import structlog


@dataclass
class TradeAlert:
    """Trade alert data structure."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    stop_price: float
    target_prices: List[float]
    size: float
    timestamp: datetime
    reason: str


@dataclass
class DailySummary:
    """Daily trading summary."""
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    daily_return: float
    max_drawdown: float
    sharpe_ratio: float
    open_positions: int


class TelegramReporter:
    """Telegram bot for reporting trading activity."""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram reporter.
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = structlog.get_logger()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self):
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession()
        self.logger.info("Telegram reporter initialized", chat_id=self.chat_id)
    
    async def disconnect(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.logger.info("Telegram reporter disconnected")
    
    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Send a message to Telegram.
        
        Args:
            text: Message text
            parse_mode: Parse mode (Markdown or HTML)
        
        Returns:
            Success status
        """
        if not self.session:
            await self.connect()
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    self.logger.debug("Message sent successfully")
                    return True
                else:
                    error_text = await response.text()
                    self.logger.error(
                        "Failed to send message",
                        status=response.status,
                        error=error_text
                    )
                    return False
        except Exception as e:
            self.logger.error("Error sending message", error=str(e))
            return False
    
    async def send_trade_alert(self, trade: TradeAlert):
        """Send trade alert to Telegram."""
        side_emoji = "🟢" if trade.side == "long" else "🔴"
        side_text = "LONG" if trade.side == "long" else "SHORT"
        
        message = f"""
*{side_emoji} TRADE ALERT: {trade.symbol} {side_text}*
────────────────────────────
• *Entry*: ${trade.entry_price:,.2f}
• *Stop*: ${trade.stop_price:,.2f}
• *Targets*: {', '.join(f'${p:,.2f}' for p in trade.target_prices)}
• *Size*: {trade.size:.4f} BTC
• *Reason*: {trade.reason}
────────────────────────────
*Time*: {trade.timestamp.strftime('%H:%M:%S')}
"""
        
        await self.send_message(message)
        self.logger.info("Trade alert sent", symbol=trade.symbol, side=trade.side)
    
    async def send_daily_summary(self, summary: DailySummary):
        """Send daily trading summary to Telegram."""
        win_rate_color = "🟢" if summary.win_rate >= 0.5 else "🔴"
        pnl_color = "🟢" if summary.total_pnl >= 0 else "🔴"
        
        message = f"""
*📊 DAILY TRADING SUMMARY: {summary.date}*
────────────────────────────
• *Trades*: {summary.total_trades}
• *Win Rate*: {summary.win_rate:.1%} {win_rate_color}
• *Wins/Losses*: {summary.winning_trades}/{summary.losing_trades}
• *PnL*: ${summary.total_pnl:+,.2f} {pnl_color}
• *Return*: {summary.daily_return:+.2%}
• *Max DD*: {summary.max_drawdown:.2%}
• *Sharpe*: {summary.sharpe_ratio:.2f}
• *Open Positions*: {summary.open_positions}
────────────────────────────
*Performance*: {'✅ Profitable' if summary.total_pnl > 0 else '❌ Loss'} day
"""
        
        await self.send_message(message)
        self.logger.info("Daily summary sent", date=summary.date)
    
    async def send_system_status(self, status_data: Dict):
        """Send system status update."""
        status_emoji = "✅" if status_data.get("healthy", False) else "⚠️"
        
        message = f"""
*⚙️ SYSTEM STATUS UPDATE*
────────────────────────────
• *Status*: {status_emoji} {status_data.get('status', 'Unknown')}
• *Uptime*: {status_data.get('uptime', 'N/A')}
• *Last Trade*: {status_data.get('last_trade_time', 'N/A')}
• *Queue Size*: {status_data.get('queue_size', 0)}
• *Memory Usage*: {status_data.get('memory_mb', 0):.1f} MB
• *CPU Load*: {status_data.get('cpu_percent', 0):.1f}%
────────────────────────────
*Alerts*: {status_data.get('alerts', 0)} pending
*Errors*: {status_data.get('errors', 0)} in last 24h
"""
        
        await self.send_message(message)
        self.logger.info("System status sent")
    
    async def send_error_alert(self, error_msg: str, context: Dict = None):
        """Send error alert to Telegram."""
        message = f"""
*🚨 ERROR ALERT*
────────────────────────────
{error_msg}
────────────────────────────
"""
        if context:
            for key, value in context.items():
                message += f"• *{key}*: {value}\n"
        
        await self.send_message(message)
        self.logger.error("Error alert sent", error=error_msg, context=context)
    
    async def send_performance_report(self, performance_data: Dict):
        """Send performance report (weekly/monthly)."""
        period = performance_data.get("period", "Weekly")
        
        message = f"""
*📈 {period.upper()} PERFORMANCE REPORT*
────────────────────────────
• *Period*: {performance_data.get('start_date')} to {performance_data.get('end_date')}
• *Total Trades*: {performance_data.get('total_trades', 0)}
• *Win Rate*: {performance_data.get('win_rate', 0):.1%}
• *Total PnL*: ${performance_data.get('total_pnl', 0):+,.2f}
• *Return*: {performance_data.get('total_return', 0):+.2%}
• *Sharpe Ratio*: {performance_data.get('sharpe_ratio', 0):.2f}
• *Max Drawdown*: {performance_data.get('max_drawdown', 0):.2%}
• *Avg Win*: ${performance_data.get('avg_win', 0):,.2f}
• *Avg Loss*: ${performance_data.get('avg_loss', 0):,.2f}
• *Profit Factor*: {performance_data.get('profit_factor', 0):.2f}
────────────────────────────
"""
        
        # Add best/worst trade if available
        if "best_trade" in performance_data:
            message += f"*Best Trade*: ${performance_data['best_trade']:+,.2f}\n"
        if "worst_trade" in performance_data:
            message += f"*Worst Trade*: ${performance_data['worst_trade']:+,.2f}\n"
        
        await self.send_message(message)
        self.logger.info("Performance report sent", period=period)
    
    async def monitor_and_report(self, check_interval: int = 300):
        """
        Continuous monitoring and reporting loop.
        
        Args:
            check_interval: Seconds between checks
        """
        self.logger.info("Starting Telegram monitoring loop")
        
        while True:
            try:
                # Check system health
                status = await self.check_system_health()
                await self.send_system_status(status)
                
                # Send daily summary if it's end of day
                now = datetime.now()
                if now.hour == 23 and now.minute >= 55:  # 5 minutes before midnight
                    summary = await self.generate_daily_summary()
                    await self.send_daily_summary(summary)
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(60)  # Wait a minute before retry
    
    async def check_system_health(self) -> Dict:
        """Check system health status."""
        # In production, this would check actual system metrics
        return {
            "healthy": True,
            "status": "Operational",
            "uptime": str(timedelta(seconds=3600)),  # Example
            "last_trade_time": datetime.now().strftime("%H:%M:%S"),
            "queue_size": 0,
            "memory_mb": 128.5,
            "cpu_percent": 15.2,
            "alerts": 0,
            "errors": 0
        }
    
    async def generate_daily_summary(self) -> DailySummary:
        """Generate daily trading summary."""
        # In production, this would query the database/trade history
        return DailySummary(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_trades=24,
            winning_trades=11,
            losing_trades=13,
            win_rate=0.458,
            total_pnl=245.67,
            daily_return=0.0246,
            max_drawdown=0.0185,
            sharpe_ratio=1.92,
            open_positions=2
        )


async def main():
    """Main entry point for Telegram reporting."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")
    
    reporter = TelegramReporter(bot_token, chat_id)
    
    try:
        await reporter.connect()
        
        # Send startup message
        await reporter.send_message(
            f"*🤖 BTC-USDT Scalper Bot Started*\n"
            f"────────────────────────────\n"
            f"• *Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"• *Version*: 1.0.0\n"
            f"• *Mode*: Paper Trading\n"
            f"────────────────────────────\n"
            f"Monitoring active. All systems nominal."
        )
        
        # Start monitoring loop
        await reporter.monitor_and_report()
        
    except KeyboardInterrupt:
        await reporter.send_message(
            f"*👋 BTC-USDT Scalper Bot Stopped*\n"
            f"────────────────────────────\n"
            f"• *Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"• *Reason*: Manual shutdown\n"
            f"────────────────────────────\n"
            f"Goodbye!"
        )
    finally:
        await reporter.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
