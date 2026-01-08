"""
⚡ INTELLIGENT TRADING BOT - Local Runner
Run the bot locally with real-time monitoring dashboard.
"""
from bot_full import IntelligentTradingBot
from monitoring_dashboard import MonitoringDashboard
from compliance_logger import ComplianceLogger
from loguru import logger
import sys


def main():
    """Run the intelligent trading bot locally with monitoring."""
    logger.info("=" * 80)
    logger.info("⚡ STARTING INTELLIGENT TRADING BOT")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Press Ctrl+C to stop the bot at any time")
    logger.info("")

    # Initialize bot
    bot = IntelligentTradingBot(
        max_workers=None,  # Auto-detect CPU cores
        use_all_stocks=True  # Scan all tradeable stocks
    )

    # Initialize compliance logger
    compliance_logger = ComplianceLogger()

    # Configure loguru for separate log files
    logger.add(
        "logs/compliance_{time}.log",
        filter=lambda record: "compliance" in record["extra"],
        rotation="1 day"
    )
    logger.add(
        "logs/orders_{time}.log",
        filter=lambda record: "order" in record["extra"],
        rotation="1 day"
    )
    logger.add(
        "logs/risk_{time}.log",
        filter=lambda record: "risk" in record["extra"],
        rotation="1 day"
    )

    # Attach compliance logger to bot
    bot.compliance_logger = compliance_logger

    # Start monitoring dashboard
    dashboard = MonitoringDashboard(bot)
    dashboard.start()

    try:
        # Start trading
        bot.start()

    except KeyboardInterrupt:
        logger.info("\n" + "=" * 80)
        logger.info("✅ Bot stopped by user")
        logger.info("=" * 80)

        # Print performance summary
        logger.info("")
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("-" * 80)

        try:
            stats = bot.stats
            logger.info(f"Total Trades:     {stats.get('total_trades', 0)}")
            logger.info(f"Winning Trades:   {stats.get('winning_trades', 0)}")
            logger.info(f"Losing Trades:    {stats.get('losing_trades', 0)}")

            if stats.get('total_trades', 0) > 0:
                win_rate = stats.get('winning_trades', 0) / stats['total_trades'] * 100
                logger.info(f"Win Rate:         {win_rate:.1f}%")

            # Get performance metrics
            sharpe = bot.performance_tracker.get_sharpe_ratio()
            max_dd = bot.performance_tracker.get_max_drawdown()

            logger.info(f"Sharpe Ratio:     {sharpe:.2f}")
            logger.info(f"Max Drawdown:     {max_dd*100:.2f}%")

        except Exception as e:
            logger.warning(f"Could not generate performance summary: {e}")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        raise

    finally:
        # Always stop dashboard
        dashboard.stop()


if __name__ == "__main__":
    main()
