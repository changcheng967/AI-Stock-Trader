"""
Compliance Logging System
Provides detailed logging for PDT compliance, orders, risk events, and shutdowns.
"""
from loguru import logger
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional


class ComplianceLogger:
    """Detailed logging system for compliance and audit trail."""

    def __init__(self, log_dir: str = "logs"):
        """Initialize compliance logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Separate log files for different events
        self.compliance_log = self.log_dir / "compliance.log"
        self.order_log = self.log_dir / "orders.log"
        self.risk_log = self.log_dir / "risk.log"

        logger.info("Compliance logger initialized")

    def log_day_trade(self, symbol: str, action: str, count: int, remaining: int):
        """Log day trade event for PDT compliance.

        Args:
            symbol: Stock symbol
            action: Action taken (e.g., 'OPENED', 'CLOSED')
            count: Day trades used in 5-day window
            remaining: Day trades remaining in 5-day window
        """
        logger.info(
            f"PDT EVENT | {datetime.now()} | {symbol} | {action} | "
            f"Trades used: {count}/3 | Remaining: {remaining}"
        )
        logger.bind(compliance=True).info(
            f"DAY_TRADE | symbol={symbol} | action={action} | "
            f"count={count} | remaining={remaining}"
        )

    def log_order_placed(self, order: Dict):
        """Log order placement.

        Args:
            order: Order dictionary with details
        """
        limit_price = order.get('limit_price', 'MARKET')
        if isinstance(limit_price, (int, float)):
            price_str = f"${limit_price:.2f}"
        else:
            price_str = str(limit_price)

        logger.info(
            f"ORDER PLACED | {order.get('symbol', 'N/A')} | {order.get('side', 'N/A')} | "
            f"{order.get('qty', 0)} shares @ {price_str}"
        )
        logger.bind(order=True).info(
            f"ORDER_PLACE | {json.dumps(order)}"
        )

    def log_order_fill(self, order_id: str, filled_qty: int, fill_price: float):
        """Log order fill.

        Args:
            order_id: Order ID
            filled_qty: Quantity filled
            fill_price: Fill price
        """
        logger.info(
            f"ORDER FILL | ID={order_id[:8]} | Filled: {filled_qty} shares @ ${fill_price:.2f}"
        )
        logger.bind(order=True).info(
            f"ORDER_FILL | order_id={order_id} | filled_qty={filled_qty} | "
            f"fill_price={fill_price}"
        )

    def log_partial_fill(self, order_id: str, filled_qty: int, total_qty: int):
        """Log partial fill.

        Args:
            order_id: Order ID
            filled_qty: Quantity filled so far
            total_qty: Total order quantity
        """
        logger.warning(
            f"PARTIAL FILL | ID={order_id[:8]} | {filled_qty}/{total_qty} shares filled"
        )
        logger.bind(order=True).warning(
            f"PARTIAL_FILL | order_id={order_id} | filled={filled_qty}/{total_qty}"
        )

    def log_order_cancelled(self, order_id: str, reason: str = "User requested"):
        """Log order cancellation.

        Args:
            order_id: Order ID
            reason: Reason for cancellation
        """
        logger.info(
            f"ORDER CANCELLED | ID={order_id[:8]} | Reason: {reason}"
        )
        logger.bind(order=True).info(
            f"ORDER_CANCEL | order_id={order_id} | reason={reason}"
        )

    def log_risk_breach(self, breach_type: str, current_value: float, limit: float):
        """Log risk limit breach.

        Args:
            breach_type: Type of breach (e.g., 'intraday_loss', 'max_drawdown')
            current_value: Current value
            limit: Limit that was breached
        """
        logger.error(
            f"🚨 RISK BREACH | Type: {breach_type} | "
            f"Current: {current_value:.2f} | Limit: {limit:.2f}"
        )
        logger.bind(risk=True).error(
            f"RISK_BREACH | type={breach_type} | current={current_value} | limit={limit}"
        )

    def log_auto_shutdown(self, reason: str):
        """Log automatic trading shutdown.

        Args:
            reason: Reason for shutdown
        """
        logger.critical(
            f"🛑 AUTO-SHUTDOWN | Reason: {reason}"
        )
        logger.bind(shutdown=True).critical(
            f"AUTO_SHUTDOWN | reason={reason}"
        )

    def log_position_opened(self, symbol: str, qty: int, price: float, strategy: str = "N/A"):
        """Log position opened.

        Args:
            symbol: Stock symbol
            qty: Quantity
            price: Entry price
            strategy: Strategy used
        """
        logger.bind(compliance=True).info(
            f"POSITION_OPEN | symbol={symbol} | qty={qty} | price={price} | strategy={strategy}"
        )

    def log_position_closed(self, symbol: str, qty: int, entry_price: float,
                           exit_price: float, pnl: float, reason: str):
        """Log position closed.

        Args:
            symbol: Stock symbol
            qty: Quantity
            entry_price: Entry price
            exit_price: Exit price
            pnl: Profit/loss
            reason: Exit reason
        """
        logger.bind(compliance=True).info(
            f"POSITION_CLOSE | symbol={symbol} | qty={qty} | entry_price={entry_price} | "
            f"exit_price={exit_price} | pnl={pnl} | reason={reason}"
        )

    def get_daily_summary(self) -> Dict:
        """Get summary of today's compliance events.

        Returns:
            Dictionary with compliance metrics
        """
        # This would parse logs and return summary
        # For now, return placeholder
        return {
            "date": datetime.now().date().isoformat(),
            "day_trades_used": 0,
            "day_trades_remaining": 3,
            "orders_placed": 0,
            "orders_filled": 0,
            "risk_breaches": 0,
            "shutdowns": 0
        }
