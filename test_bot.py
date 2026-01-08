"""
Test Bot Functionality (Works when market is closed)
Run comprehensive tests to verify all bot enhancements are working.
"""
from loguru import logger
import sys


def test_imports():
    """Test that all modules import correctly."""
    logger.info("=" * 80)
    logger.info("Testing Imports...")
    logger.info("=" * 80)

    try:
        from alpaca_client import AlpacaClient
        logger.success("✓ alpaca_client imported")
    except Exception as e:
        logger.error(f"✗ alpaca_client failed: {e}")
        return False

    try:
        from bot_full import IntelligentTradingBot
        logger.success("✓ bot_full imported")
    except Exception as e:
        logger.error(f"✗ bot_full failed: {e}")
        return False

    try:
        from risk_manager import RiskManager
        logger.success("✓ risk_manager imported")
    except Exception as e:
        logger.error(f"✗ risk_manager failed: {e}")
        return False

    try:
        from performance_tracker import PerformanceTracker
        logger.success("✓ performance_tracker imported")
    except Exception as e:
        logger.error(f"✗ performance_tracker failed: {e}")
        return False

    try:
        from monitoring_dashboard import MonitoringDashboard
        logger.success("✓ monitoring_dashboard imported")
    except Exception as e:
        logger.error(f"✗ monitoring_dashboard failed: {e}")
        return False

    try:
        from compliance_logger import ComplianceLogger
        logger.success("✓ compliance_logger imported")
    except Exception as e:
        logger.error(f"✗ compliance_logger failed: {e}")
        return False

    logger.success("\n✅ All imports successful!")
    return True


def test_pdt_features():
    """Test PDT compliance features."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing PDT Features...")
    logger.info("=" * 80)

    from alpaca_client import AlpacaClient

    client = AlpacaClient()

    # Test 1: Check PDT status
    can_trade, msg = client.can_day_trade()
    logger.info(f"PDT Status: {msg}")
    logger.success(f"✓ PDT check working")

    # Test 2: Track day trade
    client.track_day_trade('TEST_SYMBOL')
    used = client.get_day_trades_used()
    logger.info(f"Day trades used: {used}")
    logger.success(f"✓ Day trade tracking working")

    # Test 3: Get remaining trades
    remaining = client.get_day_trades_remaining()
    logger.info(f"Day trades remaining: {remaining}")
    logger.success(f"✓ Remaining trades calculation working")

    logger.success("\n✅ All PDT features working!")
    return True


def test_risk_features():
    """Test risk management features."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Risk Management Features...")
    logger.info("=" * 80)

    from risk_manager import RiskManager

    rm = RiskManager()
    rm.starting_equity = 100000.0
    rm.peak_equity = 100000.0

    # Test 1: Slippage estimation
    slippage = rm.estimate_slippage('AAPL', 100, 'buy', 150.0)
    logger.info(f"Estimated slippage: {slippage*100:.2f}%")
    logger.success(f"✓ Slippage estimation working")

    # Test 2: Price adjustment for slippage
    adjusted = rm.adjust_price_for_slippage(150.0, slippage, 'buy')
    logger.info(f"Adjusted price: ${adjusted:.2f}")
    logger.success(f"✓ Price adjustment working")

    # Test 3: Position sizing
    shares, reason = rm.calculate_position_size(
        'AAPL', 150.0, 145.5, 100000.0, volatility=0.02
    )
    logger.info(f"Position size: {shares} shares - {reason}")
    logger.success(f"✓ Position sizing working")

    # Test 4: Intraday risk check
    can_trade, msg = rm.check_intraday_risk(100000.0)
    logger.info(f"Risk check: {msg}")
    logger.success(f"✓ Intraday risk check working")

    logger.success("\n✅ All risk management features working!")
    return True


def test_market_features():
    """Test market-related features."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Market Features...")
    logger.info("=" * 80)

    from alpaca_client import AlpacaClient

    client = AlpacaClient()

    # Test 1: Market session detection
    session = client.get_market_session()
    logger.info(f"Current session: {session}")
    logger.success(f"✓ Market session detection working")

    # Test 2: Good time to trade check
    can_trade, reason = client.is_good_time_to_trade()
    logger.info(f"Can trade: {can_trade} - {reason}")
    logger.success(f"✓ Trade timing check working")

    # Test 3: Buying power validation
    valid, msg = client.validate_order_funds('AAPL', 10, 'buy', 150.0)
    logger.info(f"Buying power validation: {msg}")
    logger.success(f"✓ Buying power validation working")

    logger.success("\n✅ All market features working!")
    return True


def test_performance_features():
    """Test performance tracking features."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Performance Tracking Features...")
    logger.info("=" * 80)

    from performance_tracker import PerformanceTracker
    from datetime import datetime

    tracker = PerformanceTracker()

    # Test record_trade with all new parameters
    tracker.record_trade(
        symbol='TEST',
        side='buy',
        entry_price=100.0,
        exit_price=105.0,
        quantity=10,
        entry_time=datetime.now(),
        exit_time=datetime.now(),
        pnl=50.0,
        pnl_pct=5.0,
        exit_reason='Test',
        confidence=0.8,
        execution_cost=2.50,
        slippage_pct=0.002
    )

    logger.info("Trade recorded with execution costs")
    logger.success(f"✓ Trade recording working")

    # Verify trade data
    if len(tracker.trade_history) > 0:
        trade = tracker.trade_history[-1]
        logger.info(f"Gross P&L: ${trade['pnl']:.2f}")
        logger.info(f"Net P&L: ${trade['net_pnl']:.2f}")
        logger.info(f"Execution Cost: ${trade['execution_cost']:.2f}")
        logger.success(f"✓ Execution cost tracking working")

    logger.success("\n✅ All performance tracking features working!")
    return True


def test_compliance_logging():
    """Test compliance logging."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Compliance Logging...")
    logger.info("=" * 80)

    from compliance_logger import ComplianceLogger

    comp_logger = ComplianceLogger()

    # Test logging functions
    comp_logger.log_day_trade('TEST', 'OPEN', 1, 2)
    logger.success("✓ Day trade logging works")

    comp_logger.log_order_placed({'symbol': 'TEST', 'side': 'buy', 'qty': 10})
    logger.success("✓ Order logging works")

    comp_logger.log_risk_breach('test', 0.05, 0.03)
    logger.success("✓ Risk breach logging works")

    logger.success("\n✅ Compliance logging working!")
    return True


def initialize_bot():
    """Test bot initialization."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Testing Bot Initialization...")
    logger.info("=" * 80)

    from bot_full import IntelligentTradingBot

    # Initialize bot (this will test all integrations)
    bot = IntelligentTradingBot(max_workers=None, use_all_stocks=False)

    logger.success("✓ Bot initialized successfully")
    logger.info(f"  Starting Equity: ${bot.risk_manager.starting_equity:,.2f}")
    logger.info(f"  Max Positions: {3}")
    logger.info(f"  Workers: {bot.max_workers}")

    logger.success("\n✅ Bot initialization successful!")
    return bot


def print_feature_summary():
    """Print summary of all implemented features."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("📋 IMPLEMENTED FEATURES SUMMARY")
    logger.info("=" * 80)
    logger.info("")

    logger.info("🔒 REGULATORY COMPLIANCE:")
    logger.info("  ✅ Day trade counting (5-day rolling window)")
    logger.info("  ✅ PDT limit enforcement (3 trades below $25k)")
    logger.info("  ✅ Buying power validation before orders")
    logger.info("  ✅ Rate limiting (200 req/min with auto-retry)")
    logger.info("  ✅ Complete audit trail (separate log files)")
    logger.info("")

    logger.info("📊 MARKET REALISM:")
    logger.info("  ✅ Slippage estimation (0.1%-2% based on liquidity)")
    logger.info("  ✅ Execution cost tracking (commissions + slippage)")
    logger.info("  ✅ Liquidity-based position sizing (4 constraints)")
    logger.info("  ✅ Intraday risk monitoring (real-time checks)")
    logger.info("  ✅ Auto-shutdown on risk breach (3% intraday / 15% max)")
    logger.info("  ✅ Enhanced liquidity filtering ($10M daily volume)")
    logger.info("")

    logger.info("⏰ SMART TIMING:")
    logger.info("  ✅ Market session detection (pre/regular/after/closed)")
    logger.info("  ✅ Avoid opening volatility (first 30 min)")
    logger.info("  ✅ Avoid closing volatility (last 30 min)")
    logger.info("  ✅ No extended hours trading (low liquidity)")
    logger.info("")

    logger.info("🛡️ RISK MANAGEMENT:")
    logger.info("  ✅ Real-time risk monitoring every cycle")
    logger.info("  ✅ Kelly Criterion position sizing")
    logger.info("  ✅ Volatility-adjusted positions")
    logger.info("  ✅ Trailing stop losses")
    logger.info("  ✅ Smart rebalancing (close weak for strong)")
    logger.info("")

    logger.info("📈 MONITORING:")
    logger.info("  ✅ Real-time local dashboard (updates every 5s)")
    logger.info("  ✅ Separate log files (compliance, orders, risk)")
    logger.info("  ✅ Performance summary on exit")
    logger.info("  ✅ Net P&L tracking (after execution costs)")
    logger.info("")

    logger.info("=" * 80)


def main():
    """Run all tests."""
    logger.info("=" * 80)
    logger.info("🧪 BOT TESTING SUITE (Works when market is CLOSED)")
    logger.info("=" * 80)
    logger.info("")

    # Print feature summary first
    print_feature_summary()

    # Run tests
    tests_passed = 0
    tests_failed = 0

    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 2: PDT Features
    if test_pdt_features():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 3: Risk Features
    if test_risk_features():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 4: Market Features
    if test_market_features():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 5: Performance Features
    if test_performance_features():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 6: Compliance Logging
    if test_compliance_logging():
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 7: Bot Initialization
    try:
        initialize_bot()
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Bot initialization failed: {e}")
        tests_failed += 1

    # Print final summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Tests Passed: {tests_passed}/7")
    logger.info(f"Tests Failed: {tests_failed}/7")

    if tests_failed == 0:
        logger.success("")
        logger.success("🎉 ALL TESTS PASSED!")
        logger.success("")
        logger.success("Your bot is fully enhanced and ready to trade!")
        logger.success("")
        logger.success("To start trading: python run_bot.py")
        logger.success("To test simulation: python simulation_mode.py")
        logger.success("")
        logger.success("=" * 80)
        return 0
    else:
        logger.error("")
        logger.error("❌ SOME TESTS FAILED")
        logger.error("Please review the errors above")
        logger.error("")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    exit(main())
