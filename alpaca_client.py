"""
Streamlined Alpaca API client for paper trading.
Based on Alpaca Trading API documentation.
"""
from typing import List, Dict, Optional, Literal, Tuple
from datetime import datetime, timedelta, timezone
from loguru import logger
from collections import deque
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLossRequest, TakeProfitRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import pandas as pd

from config import settings


def round_price(price: float) -> float:
    """Round price to appropriate decimal places based on stock price.

    Args:
        price: The price to round

    Returns:
        Rounded price according to Alpaca's pricing rules:
        - Stocks >= $1: round to 2 decimal places
        - Stocks < $1: round to 4 decimal places
    """
    if price >= 1.0:
        return round(price, 2)
    else:
        return round(price, 4)


class AlpacaClient:
    """Simplified Alpaca client for paper trading."""

    def __init__(self):
        """Initialize Alpaca trading and data clients."""
        # Trading client (for orders, positions, account)
        self.trading_client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,  # Paper trading mode
            url_override=settings.alpaca_base_url  # Explicit paper trading endpoint
        )

        # Data client (for market data)
        self.data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key
        )

        logger.info(f"Alpaca paper trading client initialized: {settings.alpaca_base_url}")

        # Day trade tracking for PDT compliance
        self.day_trade_count = 0
        self.day_trade_history = []  # List of dates when day trades occurred

        # Rate limiting (200 requests/minute)
        self.request_timestamps = deque()
        self.rate_limit = 200  # requests per minute
        self.rate_window = 60  # seconds

        # Check PDT status and provide guidance
        self._check_pdt_status()

    def _check_pdt_status(self):
        """Check Pattern Day Trading protection status and provide guidance."""
        try:
            # Get current account configuration
            account = self.trading_client.get_account()

            # Check account equity vs PDT threshold ($25,000)
            equity = float(account.equity) if hasattr(account, 'equity') else 0
            pdt_threshold = settings.pdt_threshold

            logger.info(f"Account Equity: ${equity:,.2f} | PDT Threshold: ${pdt_threshold:,.2f}")

            if equity >= pdt_threshold:
                logger.info("✓ Account is ABOVE PDT threshold - unlimited day trades allowed")
            else:
                logger.warning(
                    f"⚠️ Account is BELOW PDT threshold (${equity:,.2f} < ${pdt_threshold:,.2f})\n"
                    f"   Limited to 3 day trades per 5-day period.\n\n"
                    f"   Bot is configured to respect PDT limits (max_positions=3).\n"
                    f"   To remove PDT restrictions, increase paper trading balance to ${pdt_threshold:,.0f}+.\n"
                    f"   Contact Alpaca support to reset your paper trading account."
                )

        except Exception as e:
            logger.warning(f"Could not check PDT protection status: {e}")

    def get_account(self) -> Dict:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            "id": account.id,
            "buying_power": float(account.buying_power),
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity) if hasattr(account, 'last_equity') else float(account.equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "initial_margin": float(account.initial_margin) if hasattr(account, 'initial_margin') else 0.0,
            "multiplier": float(account.multiplier) if hasattr(account, 'multiplier') else 1.0,
            "currency": account.currency if hasattr(account, 'currency') else 'USD',
        }

    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "side": pos.side,
                    "change_today": float(pos.change_today) if hasattr(pos, 'change_today') else 0.0,
                    "exchange": pos.exchange if hasattr(pos, 'exchange') else 'N/A',
                }
                for pos in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def get_market_status(self) -> Dict:
        """Check if market is open."""
        try:
            clock = self.trading_client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
                "timestamp": clock.timestamp,
            }
        except Exception as e:
            logger.error(f"Error getting market status: {e}")
            return {"is_open": False}

    def market_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"]
    ) -> Optional[Dict]:
        """Place a market order."""
        try:
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )

            order = self.trading_client.submit_order(order_data)

            logger.info(f"✓ Market order: {side.upper()} {qty} {symbol}")

            return {
                "id": order.id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": float(order.qty),
                "status": order.status,
            }

        except Exception as e:
            logger.error(f"Error placing market order for {symbol}: {e}")
            return None

    def bracket_order(
        self,
        symbol: str,
        qty: float,
        side: Literal["buy", "sell"],
        take_profit_price: float,
        stop_loss_price: float
    ) -> Optional[Dict]:
        """Place a bracket order with stop loss and take profit.

        Prices are automatically rounded to comply with Alpaca's pricing rules:
        - Stocks >= $1: rounded to 2 decimal places
        - Stocks < $1: rounded to 4 decimal places

        Validates buying power before placing order.
        """
        try:
            # Validate buying power first (for buy orders)
            if side == "buy":
                # Estimate price as the average of TP and SL, or use TP
                estimated_price = (take_profit_price + stop_loss_price) / 2
                valid, msg = self.validate_order_funds(symbol, int(qty), side, estimated_price)
                if not valid:
                    logger.error(f"Order validation failed for {symbol}: {msg}")
                    return None

            # Round prices to appropriate precision
            tp_rounded = round_price(take_profit_price)
            sl_rounded = round_price(stop_loss_price)

            logger.info(
                f"Placing bracket order for {symbol}: "
                f"TP: ${tp_rounded} (was ${take_profit_price}), "
                f"SL: ${sl_rounded} (was ${stop_loss_price})"
            )

            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=tp_rounded),
                stop_loss=StopLossRequest(stop_price=sl_rounded)
            )

            order = self.trading_client.submit_order(order_data)

            logger.info(
                f"✓ Bracket order submitted: {side.upper()} {qty} {symbol} | "
                f"TP: ${tp_rounded:.2f} | SL: ${sl_rounded:.2f}"
            )

            return {
                "id": order.id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "take_profit": tp_rounded,
                "stop_loss": sl_rounded,
                "status": "submitted",
            }

        except Exception as e:
            error_str = str(e)
            # Check if it's a PDT error
            if "pattern day trading" in error_str.lower():
                logger.warning(
                    f"⚠️ PDT protection blocked order for {symbol}. "
                    f"Disable PDT in Alpaca Dashboard to allow unlimited day trades."
                )
            else:
                logger.error(f"Error placing bracket order for {symbol}: {e}")
            return None

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        try:
            self.trading_client.cancel_orders()
            logger.info("All open orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """Close a position."""
        try:
            self.trading_client.close_position(symbol_or_asset_id=symbol)
            logger.info(f"✓ Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all positions."""
        try:
            self.trading_client.close_all_positions()
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return False

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        days: int = 100
    ) -> pd.DataFrame:
        """Get historical bar data."""
        try:
            # Map timeframe string to TimeFrame enum
            timeframe_map = {
                "1Minute": TimeFrame.Minute,
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
            }

            tf = timeframe_map.get(timeframe, TimeFrame.Day)

            # Calculate start date
            from datetime import timedelta
            end = datetime.now()
            start = end - timedelta(days=days * 2)  # Buffer for weekends

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
                end=end,
            )

            bars = self.data_client.get_stock_bars(request)

            # Convert to DataFrame
            df_list = []
            for symbol in symbols:
                if symbol not in bars:
                    continue
                for bar in bars[symbol]:
                    df_list.append({
                        "symbol": symbol,
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    })

            df = pd.DataFrame(df_list)
            if not df.empty:
                df = df.set_index("timestamp")
                df.index = pd.to_datetime(df.index)
                # Only take the last N bars per symbol
                df = df.groupby("symbol").tail(days)

            return df

        except Exception as e:
            logger.error(f"Error getting bars: {e}")
            return pd.DataFrame()

    def get_orders(self, limit: int = None, status: str = None) -> List[Dict]:
        """Get all orders with optional filtering.

        Args:
            limit: Maximum number of orders to return
            status: Filter by order status (e.g., 'open', 'closed', 'all')

        Returns:
            List of order dictionaries
        """
        try:
            # Get orders with optional status filter
            if status:
                orders = self.trading_client.get_orders(filter=status)
            else:
                orders = self.trading_client.get_orders()

            # Convert to list and apply limit if specified
            orders_list = list(orders)
            if limit:
                orders_list = orders_list[:limit]

            return [
                {
                    "id": order.id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": float(order.qty) if order.qty else 0,
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0,
                    "status": order.status,
                    "type": order.type,
                    "limit_price": float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
                    "stop_price": float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None,
                    "created_at": order.created_at.strftime('%Y-%m-%d %H:%M:%S') if order.created_at else 'N/A',
                    "filled_at": order.filled_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(order, 'filled_at') and order.filled_at else 'N/A',
                    "updated_at": order.updated_at.strftime('%Y-%m-%d %H:%M:%S') if hasattr(order, 'updated_at') and order.updated_at else 'N/A',
                }
                for order in orders_list
            ]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []

    def get_all_assets(self) -> List[Dict]:
        """Get all tradeable assets from Alpaca."""
        try:
            assets = self.trading_client.get_all_assets()
            return [
                {
                    "id": asset.id,
                    "symbol": asset.symbol,
                    "class": asset.asset_class,
                    "exchange": asset.exchange,
                    "status": asset.status,
                    "tradable": asset.tradable,
                    "marginable": asset.marginable,
                    "shortable": asset.shortable,
                    "easy_to_borrow": asset.easy_to_borrow,
                    "fractionable": asset.fractionable,
                }
                for asset in assets
            ]
        except Exception as e:
            logger.error(f"Error getting assets: {e}")
            return []

    # ==============================================================================
    # PDT COMPLIANCE METHODS
    # ==============================================================================

    def track_day_trade(self, symbol: str):
        """Track a day trade for PDT compliance.

        Call this method when a position is opened and closed on the same day.

        Args:
            symbol: The stock symbol that was day traded
        """
        today = datetime.now().date()
        self.day_trade_history.append(today)
        self._clean_old_day_trades()
        self.day_trade_count = len(self.day_trade_history)

        logger.info(
            f"Day trade tracked for {symbol} | "
            f"Total day trades (5-day window): {self.day_trade_count}/3"
        )

    def _clean_old_day_trades(self):
        """Remove day trades older than 5 days from tracking."""
        cutoff = datetime.now().date() - timedelta(days=5)
        self.day_trade_history = [d for d in self.day_trade_history if d >= cutoff]

    def can_day_trade(self) -> Tuple[bool, str]:
        """Check if account can place a day trade.

        Returns:
            Tuple of (can_trade: bool, message: str)
        """
        account = self.get_account()
        equity = account['portfolio_value']

        # Above PDT threshold - unlimited day trades
        if equity >= 25000:
            return True, "Above PDT threshold - unlimited day trades allowed"

        # Below PDT threshold - check day trade count
        self._clean_old_day_trades()
        self.day_trade_count = len(self.day_trade_history)

        if self.day_trade_count >= 3:
            return False, f"PDT limit reached: {self.day_trade_count}/3 day trades used (5-day window)"

        remaining = 3 - self.day_trade_count
        return True, f"{remaining} day trades remaining (5-day window)"

    def get_day_trades_used(self) -> int:
        """Get number of day trades used in current 5-day window."""
        self._clean_old_day_trades()
        self.day_trade_count = len(self.day_trade_history)
        return self.day_trade_count

    def get_day_trades_remaining(self) -> int:
        """Get number of day trades remaining in 5-day window.

        Returns:
            Number of remaining day trades, or float('inf') if unlimited
        """
        account = self.get_account()
        if account['portfolio_value'] >= 25000:
            return float('inf')  # Unlimited
        self._clean_old_day_trades()
        return max(0, 3 - len(self.day_trade_history))

    # ==============================================================================
    # BUYING POWER VALIDATION
    # ==============================================================================

    def validate_order_funds(self, symbol: str, qty: int, side: str, price: float = None) -> Tuple[bool, str]:
        """Validate if sufficient buying power for order.

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            price: Optional price (if not provided, will fetch latest quote)

        Returns:
            Tuple of (valid: bool, message: str)
        """
        account = self.get_account()
        buying_power = account['buying_power']
        cash = account['cash']

        # Get current price if not provided
        if price is None:
            try:
                # For simplicity, estimate from recent bars
                bars = self.get_bars([symbol], days=1)
                if bars.empty:
                    return False, f"Cannot get price for {symbol}"
                price = float(bars['close'].iloc[-1])
            except Exception as e:
                return False, f"Error getting price for {symbol}: {e}"

        # Calculate required amount
        required = price * qty

        # For margin accounts, check buying power
        # For cash accounts, check cash
        if account['multiplier'] > 1:
            available = buying_power
            account_type = "margin"
        else:
            available = cash
            account_type = "cash"

        if required > available:
            return False, (
                f"Insufficient funds ({account_type} account): "
                f"need ${required:,.2f}, have ${available:,.2f}"
            )

        return True, f"Sufficient funds (${required:,.2f} of ${available:,.2f} used)"

    # ==============================================================================
    # RATE LIMITING & RETRY LOGIC
    # ==============================================================================

    def _check_rate_limit(self):
        """Check if we're within rate limits, wait if needed."""
        now = time.time()

        # Remove old timestamps outside the window
        cutoff = now - self.rate_window
        while self.request_timestamps and self.request_timestamps[0] < cutoff:
            self.request_timestamps.popleft()

        # Check if we're at the limit
        if len(self.request_timestamps) >= self.rate_limit:
            wait_time = self.rate_window - (now - self.request_timestamps[0])
            if wait_time > 0:
                logger.warning(f"Rate limit approaching, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._check_rate_limit()  # Recursive check after waiting

        # Record this request
        self.request_timestamps.append(now)

    def _api_call_with_retry(self, func, *args, max_retries: int = 3, **kwargs):
        """Execute API call with retry logic.

        Args:
            func: The API function to call
            *args: Positional arguments for the function
            max_retries: Maximum number of retry attempts
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the function call

        Raises:
            Exception: If all retries are exhausted
        """
        from alpaca.common.exceptions import APIError

        for attempt in range(max_retries):
            self._check_rate_limit()

            try:
                return func(*args, **kwargs)
            except APIError as e:
                # Handle rate limit errors (429)
                if hasattr(e, 'code') and e.code == 429:
                    wait = (2 ** attempt) * 1  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Rate limited by API, waiting {wait}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                # Handle server errors (5xx)
                elif hasattr(e, 'code') and e.code >= 500:
                    wait = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                    logger.warning(
                        f"Server error ({e.code}), retrying in {wait}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                # Don't retry client errors (4xx except 429)
                else:
                    raise

        raise Exception(f"API call failed after {max_retries} retries")

    # ==============================================================================
    # MARKET SESSION DETECTION
    # ==============================================================================

    def get_market_session(self) -> str:
        """Get current market session.

        Returns:
            Session name: 'PRE_MARKET', 'REGULAR', 'AFTER_HOURS', or 'CLOSED'
        """
        try:
            clock = self.trading_client.get_clock()
            now = clock.timestamp
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            else:
                now = now.astimezone(timezone.utc)

            # Convert to Eastern Time
            et_zone = timezone(timedelta(hours=-4))  # EDT (UTC-4)
            # Note: This doesn't handle EST (UTC-5), but for simplicity using EDT
            now_et = now.astimezone(et_zone)
            hour = now_et.hour + now_et.minute / 60

            # Pre-market: 4:00 - 9:30 ET
            if 4 <= hour < 9.5:
                return "PRE_MARKET"
            # Regular: 9:30 - 16:00 ET
            elif 9.5 <= hour < 16:
                return "REGULAR"
            # After-hours: 16:00 - 20:00 ET
            elif 16 <= hour < 20:
                return "AFTER_HOURS"
            # Closed
            else:
                return "CLOSED"

        except Exception as e:
            logger.error(f"Error detecting market session: {e}")
            return "CLOSED"

    def is_good_time_to_trade(self) -> Tuple[bool, str]:
        """Check if current time is good for trading.

        Returns:
            Tuple of (can_trade: bool, reason: str)
        """
        session = self.get_market_session()

        # Avoid extended hours - low liquidity
        if session == "PRE_MARKET":
            return False, "Pre-market trading - low liquidity, avoid"
        if session == "AFTER_HOURS":
            return False, "After-hours trading - low liquidity, avoid"
        if session == "CLOSED":
            return False, "Market closed"

        # For regular hours, avoid first/last 30 minutes (high volatility)
        if session == "REGULAR":
            try:
                clock = self.trading_client.get_clock()
                now = clock.timestamp
                if now.tzinfo is None:
                    now = now.replace(tzinfo=timezone.utc)

                et_zone = timezone(timedelta(hours=-4))
                now_et = now.astimezone(et_zone)
                hour = now_et.hour + now_et.minute / 60

                # First 30 minutes
                if 9.5 <= hour < 10:
                    return False, "Opening volatility (first 30 min) - avoid trading"
                # Last 30 minutes
                if 15.5 <= hour < 16:
                    return False, "Closing volatility (last 30 min) - avoid trading"

                return True, "Regular hours - good to trade"

            except Exception as e:
                logger.error(f"Error checking trading time: {e}")
                return False, f"Cannot verify trading time: {e}"

        return False, f"Market session: {session}"

    # ==============================================================================
    # ORDER STATUS & PARTIAL FILL HANDLING
    # ==============================================================================

    def get_order_status(self, order_id: str) -> Dict:
        """Get detailed order status including fill information.

        Args:
            order_id: The order ID

        Returns:
            Dictionary with order status, filled quantity, fill price, etc.
        """
        try:
            order = self.trading_client.get_order_by_id(order_id)

            return {
                'id': order.id,
                'symbol': order.symbol,
                'status': order.status,  # NEW, PARTIALLY_FILLED, FILLED, etc.
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'qty': float(order.qty) if order.qty else 0,
                'fill_price': float(order.filled_avg_price) if hasattr(order, 'filled_avg_price') and order.filled_avg_price else None,
                'limit_price': float(order.limit_price) if hasattr(order, 'limit_price') and order.limit_price else None,
                'stop_price': float(order.stop_price) if hasattr(order, 'stop_price') and order.stop_price else None,
                'side': order.side,
                'type': order.type,
                'created_at': order.created_at,
                'filled_at': order.filled_at if hasattr(order, 'filled_at') else None,
                'updated_at': order.updated_at if hasattr(order, 'updated_at') else None,
            }

        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return {}

    def check_order_fills(self, symbol: str) -> Dict:
        """Check all open orders for a symbol and return fill status.

        Args:
            symbol: Stock symbol to check

        Returns:
            Dictionary mapping order IDs to their status
        """
        try:
            orders = self.get_orders(status='open')

            results = {}
            for order in orders:
                if order['symbol'] == symbol:
                    status = self.get_order_status(order['id'])
                    results[order['id']] = status

            return results

        except Exception as e:
            logger.error(f"Error checking order fills for {symbol}: {e}")
            return {}

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order.

        Args:
            order_id: The order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"✓ Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False

