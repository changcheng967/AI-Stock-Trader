"""
⚡ INTELLIGENT ADAPTIVE TRADING SYSTEM
Professional-grade algorithmic trading with advanced risk management,
smart signal analysis, and dynamic portfolio optimization.
"""
import time
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from loguru import logger
import multiprocessing

from alpaca_client import AlpacaClient
from strategies import get_strategy
from stock_screener import turbo_screener
from market_data import get_market_data
from config import settings

# Import smart systems
from risk_manager import RiskManager
from signal_analyzer import SignalAnalyzer
from performance_tracker import PerformanceTracker


class IntelligentTradingBot:
    """
    Advanced AI-powered trading bot with:
    - Dynamic risk management
    - Smart signal analysis
    - Portfolio optimization
    - Performance tracking
    - Market regime adaptation
    """

    def __init__(self, max_workers: int = None, use_all_stocks: bool = True):
        """Initialize intelligent trading system."""
        logger.info("=" * 80)
        logger.info("⚡⚡⚡ INTELLIGENT ADAPTIVE TRADING SYSTEM ⚡⚡⚡")
        logger.info("=" * 80)

        # System components
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = max_workers or max(10, cpu_count * 2)

        self.client = AlpacaClient()
        self.risk_manager = RiskManager()
        self.signal_analyzer = SignalAnalyzer()
        self.performance_tracker = PerformanceTracker()
        self.strategy = get_strategy("ensemble")

        self.use_all_stocks = use_all_stocks
        self.running = False

        # Track positions with entry data
        self.position_tracker = {}  # symbol: {'entry_time': datetime, 'entry_price': float, 'original_stop': float}

        # Performance stats
        self.stats = {
            "total_trades": 0,
            "stocks_scanned": 0,
            "opportunities_found": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "start_time": datetime.now(),
            "last_scan_time": None,
            "market_regime": "SIDEWAYS"
        }

        # Initialize risk manager with starting equity
        account = self.client.get_account()
        self.risk_manager.starting_equity = account['portfolio_value']
        self.risk_manager.peak_equity = account['portfolio_value']
        self.performance_tracker.starting_equity = account['portfolio_value']

        # Initialize position_tracker with existing positions
        existing_positions = self.client.get_positions()
        if existing_positions:
            logger.info(f"Found {len(existing_positions)} existing position(s)")
            for pos in existing_positions:
                symbol = pos['symbol']
                # For existing positions, we don't know the exact entry time or confidence
                # so we use defaults
                self.position_tracker[symbol] = {
                    'entry_time': None,  # Unknown for pre-existing positions
                    'entry_price': pos['avg_entry_price'],
                    'original_stop': pos['avg_entry_price'] * 0.97,
                    'confidence': 0.5  # Default confidence for unknown trades
                }

        logger.info(f"⚡ System initialized with {self.max_workers} workers")
        logger.info(f"⚡ Starting Equity: ${account['portfolio_value']:,.2f}")
        logger.info(f"⚡ Strategy: Multi-Strategy Ensemble with AI Enhancement")
        logger.info("")

    def start(self):
        """Start the intelligent trading system."""
        self.running = True

        # Wait for market open
        self._wait_for_market_open()

        logger.info("✅ Intelligent trading system started. Press Ctrl+C to stop.\n")

        try:
            while self.running:
                # Check if market is open
                market_status = self.client.get_market_status()
                if not market_status["is_open"]:
                    logger.info("Market closed. Waiting for next open...")
                    self._wait_for_market_open()
                    # Reset daily tracker on new market open
                    self.risk_manager.reset_daily_tracker()
                    continue

                # Run intelligent trading cycle
                self.run_intelligent_cycle()

                # Wait for next cycle
                logger.info(
                    f"⏰ Waiting {settings.check_interval_minutes} minutes until next cycle..."
                )
                time.sleep(settings.check_interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        finally:
            self.shutdown()

    def run_intelligent_cycle(self):
        """Run one intelligent trading cycle with advanced decision making."""

        logger.info("=" * 80)
        logger.info(f"⚡ INTELLIGENT TRADING CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info("")

        # 1. Get account status
        account = self.client.get_account()
        logger.info(
            f"💰 Portfolio: ${account['portfolio_value']:,.2f} | "
            f"Cash: ${account['cash']:,.2f} | "
            f"Buying Power: ${account['buying_power']:,.2f}"
        )

        # Update equity curve
        self.performance_tracker.record_equity(
            account['portfolio_value'],
            account['cash']
        )

        # Update peak equity
        if account['portfolio_value'] > self.risk_manager.peak_equity:
            self.risk_manager.peak_equity = account['portfolio_value']

        logger.info("")

        # 2. CHECK INTRADAY RISK FIRST (Auto-shutdown protection)
        can_trade, risk_msg = self.risk_manager.check_intraday_risk(account['portfolio_value'])
        if not can_trade:
            logger.critical(f"🛑 RISK LIMIT BREACHED: {risk_msg}")
            logger.critical("🛑 AUTO-SHUTDOWN INITIATED")

            # Log shutdown if compliance logger exists
            if hasattr(self, 'compliance_logger'):
                self.compliance_logger.log_auto_shutdown(risk_msg)

            # Close all positions
            logger.info("Closing all positions due to risk breach...")
            positions = self.client.get_positions()
            for pos in positions:
                self.client.close_position(pos['symbol'])
                logger.info(f"Closed {pos['symbol']}")

            # Cancel all orders
            logger.info("Cancelling all open orders...")
            self.client.cancel_all_orders()

            # Stop bot
            self.running = False
            return

        # 3. Check if good time to trade (market hours, volatility)
        can_trade_time, time_reason = self.client.is_good_time_to_trade()
        if not can_trade_time:
            logger.info(f"⏰ Waiting: {time_reason}")
            return

        logger.info(f"✅ Good time to trade: {time_reason}")

        # 4. Check PDT compliance before any new trades
        can_day_trade, pdt_msg = self.client.can_day_trade()
        if not can_day_trade:
            logger.warning(f"⚠️ {pdt_msg}")
            logger.info("Will manage existing positions but won't open new ones")
            # Continue to position management but skip new trades

        logger.info("")

        # 5. Intelligent position management (manage existing positions)
        self.intelligent_position_management()

        logger.info("")

        # 6. Market regime detection
        market_regime = self._detect_market_regime()
        self.stats['market_regime'] = market_regime

        regime_emoji = "🐂" if market_regime == "BULL" else "🐻" if market_regime == "BEAR" else "😐"
        logger.info(f"📊 Market Regime: {regime_emoji} {market_regime}")

        logger.info("")

        # 7. Smart market scanning with regime awareness
        opportunities = self.smart_market_scan(account, market_regime)

        logger.info("")

        # 8. Intelligent trade execution with risk management
        # Only open new positions if PDT allows
        if can_day_trade:
            self.intelligent_execute_trades(opportunities, account, market_regime)
        else:
            logger.info("⏸️ Skipping new trades due to PDT limits")

        logger.info("")
        logger.info("=" * 80)
        logger.info("")

    def intelligent_position_management(self):
        """Actively manage positions with smart exit decisions."""

        positions = self.client.get_positions()

        if not positions:
            logger.info("📭 No open positions")
            return

        logger.info(f"⚡ ACTIVELY MANAGING {len(positions)} POSITION(S):")
        logger.info("-" * 80)

        for pos in positions:
            symbol = pos['symbol']
            current_price = pos['current_price']
            entry_price = pos['avg_entry_price']
            quantity = pos['qty']
            unrealized_pl = pos['unrealized_pl']
            unrealized_pl_pct = pos['unrealized_plpc']

            # Calculate hold time
            entry_time = self.position_tracker.get(symbol, {}).get('entry_time')
            hold_minutes = 0
            if entry_time:
                hold_minutes = (datetime.now() - entry_time).total_seconds() / 60

            # Get original stop loss
            original_stop = self.position_tracker.get(symbol, {}).get('original_stop', entry_price * 0.97)

            # Determine status
            status_emoji = "✅" if unrealized_pl >= 0 else "❌"
            status_text = "PROFIT" if unrealized_pl >= 0 else "LOSS"

            # Calculate trailing stop
            trailing_stop = self.risk_manager.calculate_trailing_stop(
                entry_price, current_price, original_stop, unrealized_pl_pct / 100
            )

            # Show position details
            logger.info(
                f"  {status_emoji} {symbol}: {quantity} shares @ ${current_price:.2f} "
                f"(${pos['market_value']:,.2f}) | "
                f"P&L: ${unrealized_pl:,.2f} ({unrealized_pl_pct:.2f}%) - {status_text} | "
                f"Held: {hold_minutes:.0f}min"
            )

            # Get current market data for signal analysis
            try:
                data = get_market_data([symbol], days=30)
                if not data.empty:
                    symbol_data = data[data['symbol'] == symbol]

                    # Smart exit decision
                    should_exit, exit_reason = self.signal_analyzer.should_exit_position(
                        pos, symbol_data, hold_minutes
                    )

                    # Check for trailing stop upgrade
                    if trailing_stop > original_stop and unrealized_pl_pct > 3:
                        logger.info(f"    📈 Trailing stop raised: ${trailing_stop:.2f} (was ${original_stop:.2f})")

                    # Exit signals
                    if should_exit:
                        logger.warning(f"    ⚠️ EXIT SIGNAL: {exit_reason}")

                        # Close position
                        if self.client.close_position(symbol):
                            logger.info(f"    ✓ Closed {symbol}")

                            # Calculate execution costs (estimate if not tracked)
                            position_info = self.position_tracker.get(symbol, {})
                            estimated_execution_cost = position_info.get('estimated_execution_cost', unrealized_pl * 0.001)  # Default 0.1%
                            slippage_pct = position_info.get('estimated_slippage', 0.001)

                            # Record trade with execution cost tracking
                            self.performance_tracker.record_trade(
                                symbol=symbol,
                                side='buy',
                                entry_price=entry_price,
                                exit_price=current_price,
                                quantity=int(quantity),
                                entry_time=entry_time,
                                exit_time=datetime.now(),
                                pnl=unrealized_pl,
                                pnl_pct=unrealized_pl_pct,
                                exit_reason=exit_reason,
                                confidence=position_info.get('confidence', 0.5),
                                execution_cost=estimated_execution_cost,
                                slippage_pct=slippage_pct
                            )

                            # Update risk manager
                            self.risk_manager.update_daily_pnl(unrealized_pl)
                            self.risk_manager.update_intraday_pnl(unrealized_pl)

                            # Track PDT compliance (day trade if opened and closed same day)
                            if entry_time:
                                hours_held = (datetime.now() - entry_time).total_seconds() / 3600
                                if hours_held < 6.5:  # Less than typical trading day
                                    self.client.track_day_trade(symbol)
                                    logger.info(f"    📊 PDT: Day trade tracked ({hours_held:.1f} hours held)")

                            # Log compliance event if logger exists
                            if hasattr(self, 'compliance_logger'):
                                self.compliance_logger.log_position_closed(
                                    symbol, int(quantity), entry_price, current_price,
                                    unrealized_pl, exit_reason
                                )

                            # Update stats
                            self.stats['total_trades'] += 1
                            if unrealized_pl > 0:
                                self.stats['winning_trades'] += 1
                            else:
                                self.stats['losing_trades'] += 1

                            # Remove from tracker
                            if symbol in self.position_tracker:
                                del self.position_tracker[symbol]

                        else:
                            logger.error(f"    ✗ Failed to close {symbol}")

                    # Warning zones
                    elif unrealized_pl_pct < -settings.stop_loss_pct * 100 * 0.7:
                        logger.warning(
                            f"    ⚠️ WARNING: Approaching stop loss! "
                            f"({unrealized_pl_pct:.2f}% vs {-settings.stop_loss_pct * 100:.2f}%)"
                        )

                    # Profit alerts
                    elif unrealized_pl_pct > settings.take_profit_pct * 100 * 0.85:
                        logger.info(
                            f"    💰 Near take profit! "
                            f"({unrealized_pl_pct:.2f}% vs {settings.take_profit_pct * 100:.2f}%)"
                        )

            except Exception as e:
                logger.warning(f"    Could not analyze {symbol}: {e}")

        # Summary
        positions = self.client.get_positions()  # Refresh after closes
        if positions:
            total_pl = sum(p['unrealized_pl'] for p in positions)
            logger.info("-" * 80)
            logger.info(f"Total Portfolio P&L: ${total_pl:,.2f}")

            if total_pl >= 0:
                logger.success(f"Net Status: ✅ PROFIT (+${total_pl:,.2f})")
            else:
                logger.warning(f"Net Status: ❌ LOSS (-${abs(total_pl):,.2f})")

    def _detect_market_regime(self) -> str:
        """Detect current market regime using multiple indicators."""
        try:
            # Use SPY or QQQ as market proxy
            market_data = get_market_data(['SPY'], days=50)

            if market_data.empty:
                return "SIDEWAYS"

            spy_data = market_data[market_data['symbol'] == 'SPY'].sort_index()

            if len(spy_data) < 20:
                return "SIDEWAYS"

            # Calculate indicators
            recent = spy_data.tail(20)
            price_now = spy_data.iloc[-1]['close']
            price_20_ago = spy_data.iloc[-20]['close']

            # Price trend
            price_change = (price_now - price_20_ago) / price_20_ago

            # Volume trend
            recent_volume = recent['volume'].mean()
            older_volume = spy_data.tail(50).head(30)['volume'].mean()
            volume_ratio = recent_volume / older_volume if older_volume > 0 else 1

            # Determine regime
            if price_change > 0.02 and volume_ratio > 1.0:
                return "BULL"
            elif price_change < -0.02 and volume_ratio > 1.0:
                return "BEAR"
            else:
                return "SIDEWAYS"

        except Exception as e:
            logger.warning(f"Could not detect market regime: {e}")
            return "SIDEWAYS"

    def smart_market_scan(self, account: Dict, market_regime: str) -> List[Dict]:
        """Intelligent market scanning with regime awareness."""

        logger.info("🔍 SMART MARKET SCAN")
        logger.info("-" * 80)
        logger.info("")

        # Get all tradeable stocks
        if self.use_all_stocks:
            logger.info("⚡ Fetching ALL tradeable stocks...")
            all_stocks = turbo_screener.get_all_tradeable_stocks()

            if not all_stocks:
                logger.warning("No stocks found!")
                return []

            logger.info(f"✓ Found {len(all_stocks):,} stocks")
            logger.info("")

            # Quick filter
            logger.info("⚡ Quick filtering...")
            filtered = turbo_screener._apply_quick_filters(all_stocks)
            logger.info(f"✓ Quick filtered to {len(filtered):,} stocks")
            logger.info("")

            # Smart filtering with regime awareness
            logger.info("⚡ Smart filtering with data & regime analysis...")

            # Get liquidity-sorted stocks
            sorted_stocks = turbo_screener.filter_with_data_parallel(
                filtered,
                self.max_workers
            )

            logger.info(f"✓ Filtered to {len(sorted_stocks):,} liquid stocks")
            logger.info("")

            # Analyze opportunities
            logger.info("⚡ Analyzing opportunities...")

            opportunities = []

            for i, stock in enumerate(sorted_stocks[:100], 1):  # Top 100 by liquidity
                if i % 20 == 0:
                    logger.info(f"  Scanned {i}/{min(100, len(sorted_stocks))}...")

                # Handle both string symbols and dict objects
                if isinstance(stock, str):
                    symbol = stock
                elif isinstance(stock, dict):
                    symbol = stock.get('symbol', '')
                else:
                    continue

                try:
                    # Get historical data
                    data = get_market_data([symbol], days=100)

                    if data.empty:
                        continue

                    # Get current price from data
                    symbol_data = data[data['symbol'] == symbol]
                    if symbol_data.empty:
                        continue

                    price = float(symbol_data.iloc[-1]['close'])

                    # Calculate volatility from data
                    returns = symbol_data['close'].pct_change().dropna()
                    volatility = float(returns.std()) if len(returns) > 0 else 0.03

                    # Analyze signal
                    analysis = self.signal_analyzer.analyze_entry_signal(
                        symbol, data, market_regime
                    )

                    # Only consider high-confidence signals
                    if analysis['confidence'] >= self.signal_analyzer.min_confidence:

                        # Adjust confidence based on market regime
                        if market_regime == "BEAR" and 'BUY' in analysis['signal']:
                            analysis['confidence'] *= 0.6
                        elif market_regime == "BULL" and 'SELL' in analysis['signal']:
                            analysis['confidence'] *= 0.6

                        opportunities.append({
                            'symbol': symbol,
                            'price': price,
                            'volatility': volatility,
                            'signal': analysis['signal'],
                            'confidence': analysis['confidence'],
                            'strength': analysis['strength'],
                            'reason': analysis['reason'],
                            'indicators': analysis['indicators']
                        })

                except Exception as e:
                    logger.debug(f"Error analyzing {symbol}: {e}")
                    continue

            # Sort by confidence
            opportunities.sort(key=lambda x: x['confidence'], reverse=True)

            # Early exit with enough opportunities
            if len(opportunities) >= 10:
                logger.info(f"✓ Early exit - Found {len(opportunities)} high-confidence opportunities")
            else:
                logger.info(f"✓ Found {len(opportunities)} opportunities")

            self.stats['stocks_scanned'] = len(sorted_stocks[:100])

            return opportunities

        return []

    def intelligent_execute_trades(
        self,
        opportunities: List[Dict],
        account: Dict,
        market_regime: str
    ):
        """Execute trades with intelligent risk management."""

        if not opportunities:
            logger.info("📭 No high-confidence opportunities found")
            return

        logger.info("💡 INTELLIGENT TRADE EXECUTION")
        logger.info("-" * 80)
        logger.info("")

        successful_trades = []
        blocked_trades = []
        closed_for_rebalance = []

        # Get current positions
        positions = self.client.get_positions()
        current_positions = len(positions)

        logger.info(f"Current positions: {current_positions}/{settings.max_positions}")
        logger.info("")

        # Check if at max positions - consider rebalancing
        if current_positions >= settings.max_positions:
            logger.info("🔄 At max positions - checking for rebalancing opportunities...")

            if opportunities:
                best_new = opportunities[0]
                best_new_score = best_new['confidence']

                # Find weakest position
                weakest_pos = None
                weakest_score = float('inf')

                for pos in positions:
                    pl_pct = pos.get('unrealized_plpc', 0)

                    # Combine P&L with signal strength (if we could get it)
                    position_score = pl_pct / 100  # Convert to decimal

                    if position_score < weakest_score:
                        weakest_score = position_score
                        weakest_pos = pos

                # Rebalance if significantly better opportunity
                if weakest_pos and best_new_score > abs(weakest_score) + 0.2:
                    logger.info(
                        f"🔄 REBALANCING: {best_new['symbol']} (conf: {best_new_score:.2f}) "
                        f"vs {weakest_pos['symbol']} (P&L: {weakest_score:.2f}%)"
                    )

                    # Close weakest position
                    if self.client.close_position(weakest_pos['symbol']):
                        closed_for_rebalance.append(weakest_pos['symbol'])
                        positions = [p for p in positions if p['symbol'] != weakest_pos['symbol']]
                        current_positions -= 1
                        logger.info(f"  ✓ Closed {weakest_pos['symbol']}, freed up 1 slot")
                    else:
                        logger.warning(f"  ✗ Failed to close {weakest_pos['symbol']}")
                        logger.info("Rebalancing failed - keeping current positions")
                        return
                else:
                    logger.info("Current positions are strong - no rebalancing needed")
                    return

        # Execute trades on available opportunities
        available_slots = settings.max_positions - current_positions

        for i, opp in enumerate(opportunities[:available_slots], 1):
            symbol = opp['symbol']
            price = opp['price']
            confidence = opp['confidence']
            volatility = opp.get('volatility', 0.03)
            signal = opp['signal']

            logger.info(f"\n{i}. Trading {symbol}...")
            logger.info(f"   Signal: {signal} | Confidence: {confidence:.1%} | Strength: {opp['strength']}")
            logger.info(f"   Reason: {opp['reason']}")

            # Check portfolio risk
            allowed, risk_reason = self.risk_manager.check_portfolio_risk(
                positions, symbol
            )

            if not allowed:
                logger.warning(f"   ⚠️ RISK CHECK FAILED: {risk_reason}")
                blocked_trades.append(symbol)
                continue

            # Calculate smart stop loss and ADAPTIVE take profit
            stop_loss = price * (1 - settings.stop_loss_pct)

            # Use adaptive take profit based on volatility (much smarter!)
            take_profit = self.risk_manager.calculate_adaptive_take_profit(
                entry_price=price,
                stop_loss_price=stop_loss,
                volatility=volatility,
                symbol=symbol
            )

            shares, size_reason = self.risk_manager.calculate_position_size(
                symbol=symbol,
                entry_price=price,
                stop_loss=stop_loss,
                portfolio_value=account['portfolio_value'],
                volatility=volatility,
                confidence=confidence
            )

            if shares < 1:
                logger.warning(f"   ⚠️ Insufficient calculated position size: {size_reason}")
                blocked_trades.append(symbol)
                continue

            # Estimate slippage
            slippage = self.risk_manager.estimate_slippage(symbol, shares, 'buy', price)
            expected_fill_price = self.risk_manager.adjust_price_for_slippage(price, slippage, 'buy')

            # Calculate expected execution cost
            estimated_execution_cost = (expected_fill_price - price) * shares

            logger.info(f"   📊 Slippage Estimate: {slippage*100:.2f}% (~${estimated_execution_cost:.2f})")
            logger.info(f"   📊 Expected Fill: ${expected_fill_price:.2f} vs Ask: ${price:.2f}")

            # Adjust for market regime
            shares = self.risk_manager.get_risk_adjusted_position_limit(
                shares, market_regime, volatility
            )

            if shares < 1:
                logger.warning(f"   ⚠️ Position size reduced to 0 due to market conditions")
                blocked_trades.append(symbol)
                continue

            # Place bracket order (buying power validation happens inside)
            result = self.client.bracket_order(
                symbol=symbol,
                qty=shares,
                side="buy",
                take_profit_price=take_profit,
                stop_loss_price=stop_loss
            )

            if result:
                logger.info(
                    f"   ✓ ORDER PLACED: Buy {shares} {symbol} @ ${price:.2f}\n"
                    f"     Stop Loss: ${stop_loss:.2f} (-{settings.stop_loss_pct*100:.0f}%)\n"
                    f"     Take Profit: ${take_profit:.2f} (+{settings.take_profit_pct*100:.0f}%)\n"
                    f"     Position Value: ${shares * price:,.2f}\n"
                    f"     Est. Execution Cost: ${estimated_execution_cost:.2f}"
                )

                # Track position entry with execution cost info
                self.position_tracker[symbol] = {
                    'entry_time': datetime.now(),
                    'entry_price': price,
                    'original_stop': stop_loss,
                    'confidence': confidence,
                    'estimated_slippage': slippage,
                    'estimated_execution_cost': estimated_execution_cost
                }

                # Log compliance event if logger exists
                if hasattr(self, 'compliance_logger'):
                    self.compliance_logger.log_position_opened(
                        symbol, shares, price, f"Signal: {signal}"
                    )

                successful_trades.append(symbol)
                self.stats["total_trades"] += 1
                self.stats["opportunities_found"] += 1

            else:
                logger.warning(f"   ✗ Order blocked or failed")
                blocked_trades.append(symbol)

            time.sleep(1)

        # Summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 EXECUTION SUMMARY")
        logger.info("=" * 80)

        if closed_for_rebalance:
            logger.info(f"🔄 Rebalanced: Closed {len(closed_for_rebalance)} {closed_for_rebalance}")

        if successful_trades:
            logger.success(f"✅ Successful: {len(successful_trades)} {successful_trades}")

        if blocked_trades:
            logger.warning(f"⚠️ Blocked: {len(blocked_trades)} {blocked_trades}")

        logger.info("=" * 80)

    def _wait_for_market_open(self):
        """Wait for market to open."""
        while self.running:
            market_status = self.client.get_market_status()

            if market_status["is_open"]:
                logger.info("⚡ Market is OPEN - Starting intelligent trading!")
                break

            next_open = market_status.get("next_open")
            if next_open:
                wait_seconds = (next_open - datetime.now(timezone.utc)).total_seconds()
                wait_minutes = int(wait_seconds / 60)
                logger.info(f"Market closed. Opens in {wait_minutes} minutes...")
                time.sleep(min(wait_seconds, 300))
            else:
                time.sleep(60)

    def shutdown(self):
        """Shutdown the trading system."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("⚡ SHUTDOWN SUMMARY")
        logger.info("=" * 80)

        # Cancel all open orders
        self.client.cancel_all_orders()

        # Calculate runtime
        elapsed = (datetime.now() - self.stats["start_time"]).total_seconds() / 60

        logger.info(f"Runtime: {elapsed:.1f} minutes")
        logger.info(f"Stocks Scanned: {self.stats['stocks_scanned']}")
        logger.info(f"Total Trades: {self.stats['total_trades']}")
        logger.info(f"Winning Trades: {self.stats['winning_trades']}")
        logger.info(f"Losing Trades: {self.stats['losing_trades']}")

        # Generate performance report
        logger.info(self.performance_tracker.generate_report())

        # Final account status
        account = self.client.get_account()
        positions = self.client.get_positions()

        logger.info(f"Final Portfolio: ${account['portfolio_value']:,.2f}")
        logger.info(f"Open Positions: {len(positions)}")

        for pos in positions:
            logger.info(
                f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['current_price']:.2f} | "
                f"P&L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.2f}%)"
            )

        # Export trade history
        if self.stats['total_trades'] > 0:
            self.performance_tracker.export_to_csv()

        logger.info("=" * 80)
        logger.info("✅ Intelligent trading system shutdown complete")
