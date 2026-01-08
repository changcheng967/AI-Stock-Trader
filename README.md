# ⚡ Intelligent Trading Bot

Professional-grade algorithmic trading system with advanced risk management, smart signal analysis, and dynamic portfolio optimization.

## 🚀 Quick Start

```bash
# Run the bot
python run_bot.py
```

Press `Ctrl+C` to stop and see your performance report.

## 📁 Core Files

- `run_bot.py` - Main entry point
- `bot_full.py` - Intelligent trading bot
- `risk_manager.py` - Advanced risk management
- `signal_analyzer.py` - Smart signal analysis
- `performance_tracker.py` - Performance analytics
- `alpaca_client.py` - Alpaca API client
- `config.py` - Configuration

## 🎯 Features

✅ Market regime detection (Bull/Bear/Sideways)
✅ Multi-indicator analysis (RSI, MACD, BB, Momentum, Volume)
✅ Smart position sizing (Kelly Criterion)
✅ Trailing stop losses
✅ Active rebalancing
✅ Performance tracking & reporting

## ⚙️ Configuration

Edit `config.py`:
- max_positions: Max concurrent positions
- stop_loss_pct: Stop loss percentage
- take_profit_pct: Take profit percentage
- check_interval_minutes: How often to check positions

## 📊 Trading Strategy

1. Scans top 100 liquid stocks
2. Analyzes 6+ technical indicators
3. Only trades 60%+ confidence
4. Actively manages positions
5. Adapts to market regime

## 🛡️ Risk Protection

- Daily loss limit: 5%
- Max drawdown: 15%
- Trailing stops: Breakeven at 3%, trail by 2% above 6%
- Sector concentration limits

Happy Trading! 🚀📈
