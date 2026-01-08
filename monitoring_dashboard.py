"""
Real-Time Monitoring Dashboard
Provides live display of account status, PDT compliance, risk levels, and performance.
"""
import threading
import time
import os
import sys
from datetime import datetime
from loguru import logger


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


class MonitoringDashboard:
    """Real-time monitoring dashboard for local display."""

    def __init__(self, bot):
        """Initialize dashboard.

        Args:
            bot: IntelligentTradingBot instance
        """
        self.bot = bot
        self.running = False
        self.update_interval = 5  # seconds
        self.thread = None

    def start(self):
        """Start dashboard in background thread."""
        self.running = True
        self.thread = threading.Thread(target=self._display_loop, daemon=True)
        self.thread.start()
        logger.info("Dashboard started")

    def stop(self):
        """Stop dashboard."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            logger.info("Dashboard stopped")

    def _display_loop(self):
        """Continuously update dashboard."""
        while self.running:
            try:
                self._display()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                time.sleep(self.update_interval)

    def _display(self):
        """Display current status."""
        clear_screen()

        try:
            # Get current data
            account = self.bot.client.get_account()
            positions = self.bot.client.get_positions()

            # Try to get PDT stats (may not be available yet)
            try:
                day_trades_used = self.bot.client.get_day_trades_used()
                day_trades_remaining = self.bot.client.get_day_trades_remaining()
            except:
                day_trades_used = 0
                day_trades_remaining = 3

            # Calculate portfolio P&L
            total_pl = sum(p['unrealized_pl'] for p in positions) if positions else 0

            # Get risk status
            can_trade = True
            risk_msg = "OK"
            try:
                can_trade, risk_msg = self.bot.risk_manager.check_intraday_risk(account['portfolio_value'])
            except:
                pass

            # Display header
            print("=" * 80)
            print("🤖 INTELLIGENT TRADING BOT - MONITORING DASHBOARD")
            print("=" * 80)
            print(f"🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()

            # Account Status
            print("💰 ACCOUNT STATUS")
            print("-" * 80)
            print(f"  Equity:           ${account['portfolio_value']:,.2f}")
            print(f"  Cash:             ${account['cash']:,.2f}")
            print(f"  Buying Power:     ${account['buying_power']:,.2f}")
            print(f"  Portfolio P&L:    ${total_pl:+,.2f}")
            try:
                print(f"  Starting Equity:  ${self.bot.risk_manager.starting_equity:,.2f}")
            except:
                print(f"  Starting Equity:  $N/A")
            print()

            # PDT Status
            print("📊 PATTERN DAY TRADING (PDT) STATUS")
            print("-" * 80)
            if day_trades_remaining == float('inf'):
                print(f"  Status:           ✅ UNLIMITED (Account >= $25,000)")
            else:
                print(f"  Status:           ⚠️  PDT RESTRICTED")
                print(f"  Day Trades Used:  {day_trades_used}/3")
                print(f"  Remaining:        {day_trades_remaining} trades (5-day window)")

            # Show last 5 day trades
            if hasattr(self.bot.client, 'day_trade_history') and self.bot.client.day_trade_history:
                print(f"  Last Trade:       {self.bot.client.day_trade_history[-1].strftime('%Y-%m-%d')}")
            print()

            # Risk Status
            print("🛡️  RISK MANAGEMENT")
            print("-" * 80)
            if can_trade:
                print(f"  Status:           ✅ SAFE")
                try:
                    print(f"  Intraday P&L:     ${self.bot.risk_manager.intraday_pnl:+,.2f}")
                    print(f"  Max Drawdown:     {self.bot.risk_manager.intraday_max_drawdown*100:.2f}%")
                except:
                    print(f"  Intraday P&L:     $N/A")
                    print(f"  Max Drawdown:     N/A")
            else:
                print(f"  Status:           🚨 BLOCKED")
                print(f"  Reason:           {risk_msg}")
            print()

            # Market Status
            print("📈 MARKET STATUS")
            print("-" * 80)
            try:
                session = self.bot.client.get_market_session()
                can_trade_time, time_reason = self.bot.client.is_good_time_to_trade()
                print(f"  Session:          {session}")
                print(f"  Can Trade:        {'✅ YES' if can_trade_time else '❌ NO'}")
                if not can_trade_time:
                    print(f"  Reason:           {time_reason}")
            except Exception as e:
                print(f"  Status:           Error checking market status: {e}")
            print()

            # Active Positions
            print(f"📦 ACTIVE POSITIONS ({len(positions)} positions)")
            print("-" * 80)
            if positions:
                for i, pos in enumerate(positions[:5], 1):  # Show first 5
                    symbol = pos['symbol']
                    qty = pos['qty']
                    price = pos['current_price']
                    pl = pos['unrealized_pl']
                    pl_pct = pos['unrealized_plpc']
                    emoji = "✅" if pl >= 0 else "❌"

                    print(f"  {emoji} {symbol}: {qty} shares @ ${price:.2f} | P&L: ${pl:+,.2f} ({pl_pct:+.2f}%)")

                if len(positions) > 5:
                    print(f"  ... and {len(positions) - 5} more")
            else:
                print("  (No open positions)")
            print()

            # Recent Trades
            print("📜 RECENT TRADES")
            print("-" * 80)
            try:
                recent_trades = self.bot.performance_tracker.trade_history[-5:]
                if recent_trades:
                    for trade in reversed(recent_trades):
                        symbol = trade['symbol']
                        pnl = trade['pnl']
                        pnl_pct = trade['pnl_pct']
                        reason = trade['exit_reason']
                        date = trade['exit_time'][:10] if trade.get('exit_time') else 'N/A'

                        emoji = "✅" if pnl and pnl > 0 else "❌"
                        print(f"  {emoji} {date} | {symbol} | ${pnl:+,.2f} ({pnl_pct:+.2f}%) | {reason}")
                else:
                    print("  (No trades recorded yet)")
            except Exception as e:
                print(f"  (Cannot load trades: {e})")
            print()

            # Performance Stats
            print("📊 PERFORMANCE STATS")
            print("-" * 80)
            try:
                metrics = self.bot.performance_tracker.calculate_metrics()
                print(f"  Total Trades:     {metrics.get('total_trades', 0)}")
                print(f"  Win Rate:         {metrics.get('win_rate', 0)*100:.1f}%")
                print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0)*100:.2f}%")
            except Exception as e:
                print(f"  (Cannot load stats: {e})")
            print()

            print("=" * 80)
            print("Press Ctrl+C to stop the bot and dashboard")
            print("=" * 80)

        except Exception as e:
            print(f"Error updating dashboard: {e}")
            logger.error(f"Dashboard display error: {e}")


if __name__ == "__main__":
    # Test dashboard without bot
    print("Monitoring Dashboard Module")
    print("Import this module and use with IntelligentTradingBot instance")
