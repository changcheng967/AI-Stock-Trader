"""
Advanced Risk Management System
Implements portfolio-level risk, correlation analysis, and smart position sizing.
"""
import numpy as np
from typing import List, Dict, Tuple
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd

from config import settings
from market_data import get_market_data


class RiskManager:
    """Professional-grade risk management system."""

    def __init__(self):
        self.max_portfolio_risk = 0.02  # Max 2% portfolio risk per trade
        self.max_sector_exposure = 0.40  # Max 40% in one sector
        self.max_correlation = 0.70  # Avoid highly correlated positions
        self.daily_loss_limit = 0.05  # Stop trading if down 5% in a day
        self.max_drawdown = 0.15  # Stop trading if down 15% from peak

        self.starting_equity = None
        self.peak_equity = None
        self.daily_trades = []
        self.daily_pnl = 0

        # Intraday risk tracking
        self.intraday_pnl = 0.0
        self.intraday_max_drawdown = 0.0
        self.peak_intraday_equity = 0.0
        self.last_check_time = None

    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        portfolio_value: float,
        volatility: float = None,
        confidence: float = 1.0
    ) -> Tuple[int, str]:
        """
        Calculate optimal position size using multiple risk constraints.

        Uses:
        1. Risk-based sizing (Kelly Criterion simplified)
        2. Volatility adjustment
        3. Liquidity-based constraints
        4. Confidence weighting
        5. Portfolio heat management

        Returns:
            Tuple of (shares, reason)
        """
        # Risk per share
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            logger.warning(f"Invalid risk for {symbol}: stop loss >= entry price")
            return 0, "Invalid risk parameters"

        try:
            # Get liquidity data
            data = get_market_data([symbol], days=30)
            if data.empty:
                logger.warning(f"No market data for {symbol}, using basic sizing")
                return 0, "No market data available"

            symbol_data = data[data['symbol'] == symbol]

            # Constraint 1: Risk-based sizing
            max_risk_amount = portfolio_value * self.max_portfolio_risk
            shares_by_risk = int(max_risk_amount / risk_per_share)

            # Constraint 2: Liquidity-based sizing (max 1% of daily volume)
            avg_daily_volume = symbol_data['volume'].tail(20).mean()
            shares_by_liquidity = int(avg_daily_volume * 0.01)

            # Constraint 3: Dollar volume limit (max 0.1% of daily dollar volume)
            avg_dollar_volume = (symbol_data['close'] * symbol_data['volume']).tail(20).mean()
            max_dollar_amount = avg_dollar_volume * 0.001
            shares_by_dollar_vol = int(max_dollar_amount / entry_price)

            # Constraint 4: Max position size (10% of portfolio)
            max_position_value = portfolio_value * settings.max_position_size
            shares_by_position_limit = int(max_position_value / entry_price)

            # Take minimum of all constraints
            shares = min(shares_by_risk, shares_by_liquidity, shares_by_dollar_vol, shares_by_position_limit)

            # Apply volatility adjustment
            if volatility:
                vol_adjustment = min(1.5, max(0.5, 0.20 / volatility))
                shares = int(shares * vol_adjustment)

            # Apply confidence adjustment
            shares = int(shares * confidence)

            # Minimum order size
            if shares < 1:
                return 0, "Position too small (minimum 1 share)"

            # Determine which constraint was binding
            constraints = {
                "risk": shares_by_risk,
                "liquidity": shares_by_liquidity,
                "dollar_volume": shares_by_dollar_vol,
                "position_limit": shares_by_position_limit
            }

            binding_constraint = min(constraints, key=constraints.get)
            reason = f"Limited by {binding_constraint} ({shares} shares)"

            logger.info(
                f"Position sizing {symbol}: {shares} shares "
                f"(${shares * entry_price:,.2f}) | "
                f"Risk: ${risk_per_share * shares:,.2f} ({(risk_per_share/entry_price)*100:.2f}%) | "
                f"{reason}"
            )

            return shares, reason

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0, f"Error: {str(e)}"

    def check_portfolio_risk(
        self,
        positions: List[Dict],
        new_symbol: str,
        new_sector: str = None
    ) -> Tuple[bool, str]:
        """
        Check if adding a new position would violate portfolio risk rules.

        Returns:
            (allowed, reason)
        """
        reasons = []

        # 1. Check sector concentration
        if new_sector:
            sector_exposure = sum(
                p['market_value'] for p in positions
                if p.get('sector') == new_sector
            )

            if sector_exposure > settings.max_position_size * settings.max_positions * self.max_sector_exposure:
                reasons.append(f"Over-concentrated in {new_sector} sector")

        # 2. Check correlation (simplified - in real trading, use correlation matrix)
        # For now, just warn if too many positions in same sector
        if positions and new_sector:
            same_sector_count = sum(
                1 for p in positions
                if p.get('sector') == new_sector
            )

            if same_sector_count >= 2:
                reasons.append(f"Already have {same_sector_count} positions in {new_sector}")

        # 3. Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit * settings.initial_capital:
            reasons.append(f"Daily loss limit reached (-${abs(self.daily_pnl):,.2f})")

        # 4. Check max drawdown
        if self.peak_equity:
            current_equity = self.peak_equity + self.daily_pnl
            drawdown = (self.peak_equity - current_equity) / self.peak_equity

            if drawdown > self.max_drawdown:
                reasons.append(f"Max drawdown reached ({drawdown*100:.1f}%)")

        allowed = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else "Pass"

        return allowed, reason

    def should_trading_halt(self) -> Tuple[bool, str]:
        """Check if trading should be halted due to risk limits."""
        # Daily loss limit check
        if abs(self.daily_pnl) > self.daily_loss_limit * settings.initial_capital:
            return True, f"Daily loss limit exceeded: -${abs(self.daily_pnl):,.2f}"

        return False, ""

    def update_daily_pnl(self, realized_pnl: float):
        """Update daily P&L after a trade closes."""
        self.daily_trades.append({
            'time': datetime.now(),
            'pnl': realized_pnl
        })
        self.daily_pnl += realized_pnl

        logger.info(f"Daily P&L updated: ${self.daily_pnl:,.2f}")

    def reset_daily_tracker(self):
        """Reset daily tracking at start of new trading day."""
        self.daily_pnl = 0
        self.daily_trades = []
        logger.info("Daily risk tracker reset")

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        original_stop: float,
        profit_pct: float
    ) -> float:
        """
        Calculate SMART trailing stop loss to maximize profits.

        Advanced Rules (momentum-based):
        - Below 2% profit: Keep original tight stop (2.5%)
        - 2-4% profit: Move stop to breakeven + 0.5% (lock in small profit)
        - 4-8% profit: Trail by 1.5% below current price (aggressive)
        - Above 8% profit: Trail by 1% below current price (very aggressive - let it run!)
        """
        if profit_pct < 0.02:
            # Still near entry - keep original stop
            return original_stop

        elif profit_pct < 0.04:
            # Small profit - lock in breakeven + small gain
            return entry_price * 1.005

        elif profit_pct < 0.08:
            # Good profit - trail aggressively by 1.5%
            return current_price * 0.985

        else:
            # Big profit - trail very aggressively by 1% (let winners run!)
            return current_price * 0.99

    def assess_market_regime(self, market_data: Dict) -> str:
        """
        Assess current market regime (bull/bear/sideways).

        Uses:
        - Price vs moving averages
        - Volume trends
        - Volatility levels
        """
        # Simplified regime detection
        # In production, use more sophisticated indicators

        price_trend = market_data.get('price_trend', 0)
        volume_trend = market_data.get('volume_trend', 0)
        volatility = market_data.get('volatility', 0.02)

        if price_trend > 0.02 and volume_trend > 0:
            return "BULL"
        elif price_trend < -0.02 and volume_trend > 0:
            return "BEAR"
        else:
            return "SIDEWAYS"

    def get_risk_adjusted_position_limit(
        self,
        base_limit: int,
        market_regime: str,
        volatility: float
    ) -> int:
        """
        Adjust position size based on market conditions.

        Bull market + low vol = full size
        Bear market or high vol = reduce size
        """
        multiplier = 1.0

        if market_regime == "BULL" and volatility < 0.025:
            multiplier = 1.2  # Can be slightly aggressive in bullish low-vol

        elif market_regime == "BEAR":
            multiplier = 0.5  # Reduce size in bear market

        elif volatility > 0.04:
            multiplier = 0.6  # Reduce size in high volatility

        return int(base_limit * multiplier)

    def calculate_adaptive_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        volatility: float,
        symbol: str = None
    ) -> float:
        """
        Calculate ADAPTIVE take profit based on volatility and ATR.

        Smart Logic:
        - High volatility stocks: Wider take profit (need more room)
        - Low volatility stocks: Tighter take profit (quick profits)
        - Uses risk-reward ratio of at least 3:1
        - Adjusts based on market conditions
        """
        # Calculate risk (distance to stop loss)
        risk_per_share = entry_price - stop_loss_price
        risk_pct = risk_per_share / entry_price

        # Base take profit: 3x risk (3:1 reward-risk ratio)
        base_reward = risk_per_share * 3
        base_take_profit = entry_price + base_reward

        # Adjust for volatility
        if volatility < 0.015:
            # Low volatility - stock doesn't move much, be conservative
            # Use 2.5x reward ratio
            take_profit = entry_price + (risk_per_share * 2.5)

        elif volatility < 0.03:
            # Normal volatility - use standard 3x
            take_profit = base_take_profit

        elif volatility < 0.05:
            # High volatility - needs more room, use 4x
            take_profit = entry_price + (risk_per_share * 4)

        else:
            # Very high volatility - use 5x (let it breathe!)
            take_profit = entry_price + (risk_per_share * 5)

        # Calculate percentage gain
        take_profit_pct = (take_profit - entry_price) / entry_price

        # Sanity checks
        if take_profit_pct < 0.04:
            # Minimum 4% target
            take_profit = entry_price * 1.04
        elif take_profit_pct > 0.20:
            # Maximum 20% target (don't be greedy - take profits!)
            take_profit = entry_price * 1.20

        logger.info(
            f"Adaptive TP for {symbol or 'stock'}: ${take_profit:.2f} "
            f"({take_profit_pct*100:.1f}%) | Vol: {volatility*100:.2f}%"
        )

        return take_profit

    def calculate_portfolio_metrics(self, positions: List[Dict]) -> Dict:
        """Calculate portfolio-level risk metrics."""
        if not positions:
            return {
                'total_value': 0,
                'total_unrealized_pnl': 0,
                'num_positions': 0,
                'avg_pnl_pct': 0,
                'best_position': None,
                'worst_position': None,
                'sector_exposure': {}
            }

        total_value = sum(p['market_value'] for p in positions)
        total_pnl = sum(p['unrealized_pl'] for p in positions)
        avg_pnl_pct = sum(p['unrealized_plpc'] for p in positions) / len(positions)

        # Find best and worst positions
        sorted_pos = sorted(positions, key=lambda x: x['unrealized_plpc'], reverse=True)
        best = sorted_pos[0] if sorted_pos else None
        worst = sorted_pos[-1] if sorted_pos else None

        # Sector exposure
        sectors = {}
        for p in positions:
            sector = p.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + p['market_value']

        return {
            'total_value': total_value,
            'total_unrealized_pnl': total_pnl,
            'num_positions': len(positions),
            'avg_pnl_pct': avg_pnl_pct,
            'best_position': best['symbol'] if best else None,
            'worst_position': worst['symbol'] if worst else None,
            'sector_exposure': sectors,
            'portfolio_beta': self._calculate_portfolio_beta(positions)
        }

    def _calculate_portfolio_beta(self, positions: List[Dict]) -> float:
        """
        Calculate portfolio beta (simplified).
        In production, use actual beta values from data provider.
        """
        # Placeholder - assumes avg beta of 1.0
        if not positions:
            return 1.0
        return 1.0

    # ==============================================================================
    # SLIPPAGE ESTIMATION
    # ==============================================================================

    def estimate_slippage(self, symbol: str, qty: int, side: str, price: float) -> float:
        """Estimate slippage based on order size and liquidity.

        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            price: Current price

        Returns:
            Estimated slippage as a decimal (e.g., 0.005 = 0.5%)
        """
        try:
            # Get recent market data
            data = get_market_data([symbol], days=30)
            if data.empty:
                return 0.005  # Default 0.5% slippage

            symbol_data = data[data['symbol'] == symbol]
            avg_daily_volume = symbol_data['volume'].tail(20).mean()

            # Calculate order size as % of daily volume
            order_value = price * qty
            avg_dollar_volume = (symbol_data['close'] * symbol_data['volume']).tail(20).mean()
            volume_ratio = order_value / avg_dollar_volume

            # Slippage increases with order size
            # Base slippage: 0.1% for small orders
            # Up to 2% for orders that are 5% of daily volume
            base_slippage = 0.001
            max_slippage = 0.02

            # Linear interpolation
            slippage = base_slippage + (max_slippage - base_slippage) * min(volume_ratio / 0.05, 1.0)

            # Additional slippage for low-priced stocks
            if price < 10:
                slippage *= 1.5
            elif price < 5:
                slippage *= 2.0

            logger.info(f"Estimated slippage for {symbol}: {slippage*100:.2f}%")

            return slippage

        except Exception as e:
            logger.error(f"Error estimating slippage for {symbol}: {e}")
            return 0.005  # Default 0.5% on error

    def adjust_price_for_slippage(self, price: float, slippage: float, side: str) -> float:
        """Adjust expected price for slippage.

        Args:
            price: Current price
            slippage: Slippage as decimal (e.g., 0.005 = 0.5%)
            side: 'buy' or 'sell'

        Returns:
            Adjusted price
        """
        if side == 'buy':
            return price * (1 + slippage)  # Pay more
        else:
            return price * (1 - slippage)  # Receive less

    # ==============================================================================
    # INTRADAY RISK MONITORING
    # ==============================================================================

    def check_intraday_risk(self, current_equity: float) -> Tuple[bool, str]:
        """Check if intraday risk limits are breached.

        Args:
            current_equity: Current portfolio equity

        Returns:
            Tuple of (can_trade: bool, message: str)
        """
        now = datetime.now()

        if self.last_check_time is None:
            # First check - initialize tracking
            self.last_check_time = now
            self.peak_intraday_equity = current_equity
            return True, "First check - initialized"

        # Update peak equity
        if current_equity > self.peak_intraday_equity:
            self.peak_intraday_equity = current_equity

        # Calculate intraday drawdown
        intraday_dd = (self.peak_intraday_equity - current_equity) / self.peak_intraday_equity
        self.intraday_max_drawdown = max(self.intraday_max_drawdown, intraday_dd)

        # Check intraday loss limit (3%)
        if intraday_dd > 0.03:
            return False, f"Intraday loss limit hit: {intraday_dd*100:.2f}% > 3%"

        # Check total drawdown (15%)
        if self.starting_equity:
            total_dd = (self.starting_equity - current_equity) / self.starting_equity
            if total_dd > self.max_drawdown:
                return False, f"Max drawdown exceeded: {total_dd*100:.2f}% > {self.max_drawdown*100:.2f}%"

        return True, "Risk limits OK"

    def update_intraday_pnl(self, pnl: float):
        """Update intraday P&L.

        Args:
            pnl: Profit/loss amount to add
        """
        self.intraday_pnl += pnl
        logger.debug(f"Intraday P&L updated: ${self.intraday_pnl:,.2f}")

    def reset_intraday_tracker(self):
        """Reset intraday tracking (call at start of each day)."""
        self.intraday_pnl = 0.0
        self.intraday_max_drawdown = 0.0
        self.peak_intraday_equity = 0.0
        self.last_check_time = None
        logger.info("Intraday risk tracker reset")

