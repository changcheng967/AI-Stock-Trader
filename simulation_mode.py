"""
Simulation Mode for Testing
Allows testing the bot when market is closed by simulating market conditions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from loguru import logger
import random


class MarketSimulator:
    """Simulate market conditions for testing when market is closed."""

    def __init__(self, seed: int = 42):
        """Initialize simulator with reproducible random seed."""
        random.seed(seed)
        np.random.seed(seed)
        logger.info(f"Market Simulator initialized with seed={seed}")

    def generate_mock_data(self, symbols: List[str], days: int = 30) -> pd.DataFrame:
        """Generate realistic mock market data.

        Args:
            symbols: List of stock symbols
            days: Number of days of data to generate

        Returns:
            DataFrame with mock market data
        """
        data_list = []
        base_date = datetime.now() - timedelta(days=days)

        for symbol in symbols:
            # Generate realistic price movements
            base_price = random.uniform(10, 200)

            for day in range(days):
                date = base_date + timedelta(days=day)

                # Skip weekends
                if date.weekday() >= 5:
                    continue

                # Random walk with drift
                daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily drift, 2% volatility

                # Generate OHLC
                open_price = base_price * (1 + np.random.normal(0, 0.005))
                high = open_price * (1 + abs(np.random.normal(0, 0.01)))
                low = open_price * (1 - abs(np.random.normal(0, 0.01)))
                close = open_price * (1 + daily_return)

                # Ensure high >= open, close and low <= open, close
                high = max(high, open_price, close)
                low = min(low, open_price, close)

                # Generate volume (log-normal distribution)
                volume = int(np.random.lognormal(15, 0.5))  # Mean ~3.3M shares

                data_list.append({
                    'symbol': symbol,
                    'timestamp': date,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close, 2),
                    'volume': volume
                })

                # Update base price for next day
                base_price = close

        df = pd.DataFrame(data_list)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        logger.info(f"Generated {len(data_list)} bars for {len(symbols)} symbols")
        return df

    def generate_mock_account(self) -> Dict:
        """Generate mock account data for testing.

        Returns:
            Dictionary with account information
        """
        account = {
            'id': 'test-account',
            'buying_power': 100000.00,
            'cash': 50000.00,
            'portfolio_value': 100000.00,
            'equity': 100000.00,
            'last_equity': 100000.00,
            'long_market_value': 50000.00,
            'short_market_value': 0.00,
            'initial_margin': 25000.00,
            'multiplier': 2.0,
            'currency': 'USD'
        }

        logger.info("Generated mock account data")
        return account

    def generate_mock_positions(self, num_positions: int = 3) -> List[Dict]:
        """Generate mock positions for testing.

        Args:
            num_positions: Number of mock positions to create

        Returns:
            List of position dictionaries
        """
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'JPM', 'JNJ', 'V']
        positions = []

        for i in range(min(num_positions, len(symbols))):
            symbol = symbols[i]
            qty = random.randint(50, 200)
            entry_price = random.uniform(50, 200)
            current_price = entry_price * (1 + random.uniform(-0.05, 0.05))

            unrealized_pl = (current_price - entry_price) * qty
            unrealized_plpc = (unrealized_pl / (entry_price * qty)) * 100

            positions.append({
                'symbol': symbol,
                'qty': float(qty),
                'avg_entry_price': round(entry_price, 2),
                'current_price': round(current_price, 2),
                'market_value': round(current_price * qty, 2),
                'unrealized_pl': round(unrealized_pl, 2),
                'unrealized_plpc': round(unrealized_plpc, 2),
                'side': 'long',
                'change_today': round(random.uniform(-2, 2), 2),
                'exchange': 'NASDAQ'
            })

        logger.info(f"Generated {len(positions)} mock positions")
        return positions


class TestRunner:
    """Test runner for validating bot functionality without live market."""

    def __init__(self):
        """Initialize test runner."""
        self.simulator = MarketSimulator()
        self.passed_tests = 0
        self.failed_tests = 0

    def test_pdt_compliance(self):
        """Test PDT compliance tracking."""
        logger.info("=" * 80)
        logger.info("TEST 1: PDT Compliance Tracking")
        logger.info("=" * 80)

        try:
            from alpaca_client import AlpacaClient

            client = AlpacaClient()

            # Test day trade tracking
            client.track_day_trade('TEST')
            assert client.day_trade_count == 1, "Day trade not tracked"

            client.track_day_trade('TEST2')
            assert client.day_trade_count == 2, "Second day trade not tracked"

            # Test can_day_trade
            can_trade, msg = client.can_day_trade()
            assert isinstance(can_trade, bool), "can_day_trade should return bool"
            assert isinstance(msg, str), "can_day_trade should return message"

            # Test getters
            used = client.get_day_trades_used()
            remaining = client.get_day_trades_remaining()
            assert used == 2, f"Expected 2 trades used, got {used}"

            # Check if account is above PDT threshold (paper account has $100k)
            account = client.get_account()
            if account['portfolio_value'] >= 25000:
                # Above PDT threshold - unlimited trades
                assert remaining == float('inf'), f"Expected unlimited (inf) remaining, got {remaining}"
            else:
                # Below PDT threshold - should have 1 remaining
                assert remaining == 1, f"Expected 1 remaining, got {remaining}"

            logger.success("✅ PDT Compliance Tracking: PASSED")
            self.passed_tests += 1
            return True

        except Exception as e:
            logger.error(f"❌ PDT Compliance Tracking: FAILED - {e}")
            self.failed_tests += 1
            return False

    def test_buying_power_validation(self):
        """Test buying power validation."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 2: Buying Power Validation")
        logger.info("=" * 80)

        try:
            from alpaca_client import AlpacaClient

            client = AlpacaClient()

            # Test validate_order_funds
            valid, msg = client.validate_order_funds('AAPL', 10, 'buy', 150.0)
            assert isinstance(valid, bool), "validate_order_funds should return bool"
            assert isinstance(msg, str), "validate_order_funds should return message"
            logger.info(f"Validation result: {msg}")

            logger.success("✅ Buying Power Validation: PASSED")
            self.passed_tests += 1
            return True

        except Exception as e:
            logger.error(f"❌ Buying Power Validation: FAILED - {e}")
            self.failed_tests += 1
            return False

    def test_slippage_estimation(self):
        """Test slippage estimation."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 3: Slippage Estimation")
        logger.info("=" * 80)

        try:
            from risk_manager import RiskManager

            rm = RiskManager()

            # Small order should have low slippage
            slippage_small = rm.estimate_slippage('AAPL', 100, 'buy', 150.0)
            assert 0 <= slippage_small <= 0.01, f"Small order slippage too high: {slippage_small}"
            logger.info(f"Small order (100 shares): {slippage_small*100:.2f}% slippage")

            # Large order should have higher slippage
            slippage_large = rm.estimate_slippage('AAPL', 10000, 'buy', 150.0)
            assert slippage_large >= slippage_small, "Large order should have more slippage"
            logger.info(f"Large order (10,000 shares): {slippage_large*100:.2f}% slippage")

            # Test price adjustment
            adjusted_buy = rm.adjust_price_for_slippage(150.0, 0.005, 'buy')
            adjusted_sell = rm.adjust_price_for_slippage(150.0, 0.005, 'sell')

            assert adjusted_buy > 150.0, "Buy price should increase with slippage"
            assert adjusted_sell < 150.0, "Sell price should decrease with slippage"
            logger.info(f"Price adjustments: Buy ${adjusted_buy:.2f}, Sell ${adjusted_sell:.2f}")

            logger.success("✅ Slippage Estimation: PASSED")
            self.passed_tests += 1
            return True

        except Exception as e:
            logger.error(f"❌ Slippage Estimation: FAILED - {e}")
            self.failed_tests += 1
            return False

    def test_risk_management(self):
        """Test risk management features."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 4: Risk Management")
        logger.info("=" * 80)

        try:
            from risk_manager import RiskManager

            rm = RiskManager()
            rm.starting_equity = 100000.0
            rm.peak_equity = 100000.0

            # Test intraday risk check (should pass first time)
            can_trade, msg = rm.check_intraday_risk(100000.0)
            assert can_trade, "First risk check should pass"
            logger.info(f"Risk check: {msg}")

            # Test with loss (use 96999 for slightly more than 3% to trigger > 0.03 check)
            can_trade, msg = rm.check_intraday_risk(96999.0)  # ~3.001% loss
            assert not can_trade, "Should fail at >3% loss"
            assert "Intraday loss limit" in msg, "Should mention intraday loss limit"
            logger.info(f"Loss check correctly blocked: {msg}")

            # Test position sizing
            shares, reason = rm.calculate_position_size(
                'AAPL', 150.0, 145.5, 100000.0, volatility=0.02, confidence=0.8
            )
            assert shares > 0, "Should calculate positive position size"
            assert isinstance(reason, str), "Should return reason string"
            logger.info(f"Position sizing: {shares} shares - {reason}")

            logger.success("✅ Risk Management: PASSED")
            self.passed_tests += 1
            return True

        except Exception as e:
            logger.error(f"❌ Risk Management: FAILED - {e}")
            self.failed_tests += 1
            return False

    def test_market_session_detection(self):
        """Test market session detection."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 5: Market Session Detection")
        logger.info("=" * 80)

        try:
            from alpaca_client import AlpacaClient

            client = AlpacaClient()

            # Test get_market_session
            session = client.get_market_session()
            assert session in ['PRE_MARKET', 'REGULAR', 'AFTER_HOURS', 'CLOSED'], \
                f"Invalid session: {session}"
            logger.info(f"Current session: {session}")

            # Test is_good_time_to_trade
            can_trade, reason = client.is_good_time_to_trade()
            assert isinstance(can_trade, bool), "Should return bool"
            assert isinstance(reason, str), "Should return reason"
            logger.info(f"Can trade: {can_trade} - {reason}")

            logger.success("✅ Market Session Detection: PASSED")
            self.passed_tests += 1
            return True

        except Exception as e:
            logger.error(f"❌ Market Session Detection: FAILED - {e}")
            self.failed_tests += 1
            return False

    def test_performance_tracking(self):
        """Test performance tracking with execution costs."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 6: Performance Tracking")
        logger.info("=" * 80)

        try:
            from performance_tracker import PerformanceTracker

            tracker = PerformanceTracker()

            # Get initial trade count (tracker loads existing trades from file)
            initial_count = len(tracker.trade_history)

            # Test record_trade with execution costs
            tracker.record_trade(
                symbol='AAPL',
                side='buy',
                entry_price=150.0,
                exit_price=155.0,
                quantity=100,
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                pnl=500.0,
                pnl_pct=3.33,
                exit_reason='Take Profit',
                confidence=0.75,
                execution_cost=5.50,
                slippage_pct=0.003
            )

            # Verify trade was recorded
            assert len(tracker.trade_history) == initial_count + 1, f"Trade should be recorded (expected {initial_count + 1}, got {len(tracker.trade_history)})"
            trade = tracker.trade_history[-1]  # Get the last trade (the one we just recorded)

            assert trade['execution_cost'] == 5.50, "Execution cost not recorded"
            assert trade['slippage_pct'] == 0.003, "Slippage not recorded"
            assert trade['net_pnl'] == 494.50, "Net P&L not calculated correctly"
            logger.info(f"Trade recorded: Gross ${trade['pnl']:.2f}, Net ${trade['net_pnl']:.2f}")

            logger.success("✅ Performance Tracking: PASSED")
            self.passed_tests += 1
            return True

        except Exception as e:
            logger.error(f"❌ Performance Tracking: FAILED - {e}")
            self.failed_tests += 1
            return False

    def test_compliance_logging(self):
        """Test compliance logging."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("TEST 7: Compliance Logging")
        logger.info("=" * 80)

        try:
            from compliance_logger import ComplianceLogger

            comp_logger = ComplianceLogger()

            # Test various log methods (should not crash)
            comp_logger.log_day_trade('TEST', 'TEST', 1, 2)
            logger.info("✓ Day trade logged")

            comp_logger.log_order_placed({'symbol': 'AAPL', 'side': 'buy', 'qty': 100})
            logger.info("✓ Order logged")

            comp_logger.log_order_fill('test_id', 100, 150.0)
            logger.info("✓ Order fill logged")

            comp_logger.log_risk_breach('intraday_loss', 0.04, 0.03)
            logger.info("✓ Risk breach logged")

            logger.success("✅ Compliance Logging: PASSED")
            self.passed_tests += 1
            return True

        except Exception as e:
            logger.error(f"❌ Compliance Logging: FAILED - {e}")
            self.failed_tests += 1
            return False

    def run_all_tests(self):
        """Run all tests and print summary."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("🧪 RUNNING COMPREHENSIVE TEST SUITE")
        logger.info("=" * 80)
        logger.info("")

        # Run all tests
        self.test_pdt_compliance()
        self.test_buying_power_validation()
        self.test_slippage_estimation()
        self.test_risk_management()
        self.test_market_session_detection()
        self.test_performance_tracking()
        self.test_compliance_logging()

        # Print summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Tests:  {self.passed_tests + self.failed_tests}")
        logger.success(f"Passed:       {self.passed_tests}")
        if self.failed_tests > 0:
            logger.error(f"Failed:       {self.failed_tests}")
        else:
            logger.success("All tests passed!")
        logger.info("=" * 80)

        return self.failed_tests == 0


if __name__ == "__main__":
    logger.info("Starting Test Runner...")
    tester = TestRunner()
    all_passed = tester.run_all_tests()

    if all_passed:
        logger.success("✅ All tests passed! Bot is ready for trading.")
        exit(0)
    else:
        logger.error("❌ Some tests failed. Please review the errors above.")
        exit(1)
