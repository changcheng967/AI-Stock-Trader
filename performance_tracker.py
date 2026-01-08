"""
Performance Analytics and Tracking System
Comprehensive trade history, performance metrics, and analytics.
"""
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger


class PerformanceTracker:
    """Track and analyze trading performance."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.trade_history = []
        self.equity_curve = []
        self.start_time = datetime.now()
        self.starting_equity = None

        self._load_history()

    def _load_history(self):
        """Load historical trade data."""
        history_file = self.data_dir / "trade_history.json"

        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.trade_history = data.get('trades', [])
                    logger.info(f"Loaded {len(self.trade_history)} historical trades")
            except Exception as e:
                logger.warning(f"Could not load trade history: {e}")

    def _save_history(self):
        """Save trade history to disk."""
        history_file = self.data_dir / "trade_history.json"

        try:
            with open(history_file, 'w') as f:
                json.dump({
                    'trades': self.trade_history,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save trade history: {e}")

    def record_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: Optional[float],
        quantity: int,
        entry_time: datetime,
        exit_time: Optional[datetime],
        pnl: Optional[float],
        pnl_pct: Optional[float],
        exit_reason: str,
        confidence: float,
        execution_cost: float = 0.0,
        slippage_pct: float = 0.0
    ):
        """Record a trade in the history.

        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            entry_price: Entry price
            exit_price: Exit price (if position closed)
            quantity: Number of shares
            entry_time: Entry time
            exit_time: Exit time (if position closed)
            pnl: Profit/loss amount
            pnl_pct: Profit/loss percentage
            exit_reason: Reason for exit
            confidence: Signal confidence score
            execution_cost: Execution costs (commissions + slippage)
            slippage_pct: Slippage percentage (0.01 = 1%)
        """

        # Calculate net P&L after execution costs
        net_pnl = pnl - execution_cost if pnl is not None else None

        trade = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'entry_time': entry_time.isoformat() if entry_time else None,
            'exit_time': exit_time.isoformat() if exit_time else None,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'confidence': confidence,
            'hold_minutes': (exit_time - entry_time).total_seconds() / 60 if (exit_time and entry_time) else None,
            'execution_cost': execution_cost,
            'slippage_pct': slippage_pct,
            'net_pnl': net_pnl,
            'net_pnl_pct': ((net_pnl / (entry_price * quantity)) * 100) if (net_pnl and quantity > 0) else pnl_pct
        }

        self.trade_history.append(trade)
        self._save_history()

        pnl_val = pnl if pnl is not None else 0
        net_pnl_val = net_pnl if net_pnl is not None else 0
        pnl_pct_val = pnl_pct if pnl_pct is not None else 0

        logger.info(
            f"Trade recorded: {symbol} {side} {quantity} @ ${entry_price:.2f} | "
            f"Gross P&L: ${pnl_val:,.2f} ({pnl_pct_val:.2f}%) | "
            f"Exec Cost: ${execution_cost:.2f} | "
            f"Net P&L: ${net_pnl_val:,.2f}"
        )

    def record_equity(self, equity: float, cash: float):
        """Record equity snapshot for curve analysis."""
        self.equity_curve.append({
            'timestamp': datetime.now().isoformat(),
            'equity': equity,
            'cash': cash,
            'invested': equity - cash
        })

        # Keep last 1000 records
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]

    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""

        if not self.trade_history:
            return self._empty_metrics()

        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.trade_history)

        # Only analyze closed trades
        closed_trades = df[df['exit_price'].notna()]

        if len(closed_trades) == 0:
            return self._empty_metrics()

        total_trades = len(closed_trades)
        winning_trades = closed_trades[closed_trades['pnl'] > 0]
        losing_trades = closed_trades[closed_trades['pnl'] < 0]

        # Basic metrics
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

        total_pnl = closed_trades['pnl'].sum()
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        # Risk metrics
        profitable_trades = closed_trades[closed_trades['pnl_pct'] > 0]
        avg_win_pct = profitable_trades['pnl_pct'].mean() if len(profitable_trades) > 0 else 0
        avg_loss_pct = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0

        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Time metrics
        avg_hold_time = closed_trades['hold_minutes'].mean() if 'hold_minutes' in closed_trades.columns else 0

        # Calculate drawdown
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            rolling_max = equity_series.expanding().max()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        # Calculate Sharpe ratio (simplified - assumes 5% risk-free rate)
        if len(self.equity_curve) > 10:
            equity_series = pd.Series([e['equity'] for e in self.equity_curve])
            returns = equity_series.pct_change().dropna()

            if len(returns) > 0:
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # Best and worst trades
        best_trade = closed_trades.loc[closed_trades['pnl'].idxmax()] if len(closed_trades) > 0 else None
        worst_trade = closed_trades.loc[closed_trades['pnl'].idxmin()] if len(closed_trades) > 0 else None

        metrics = {
            'total_trades': int(total_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'avg_hold_minutes': avg_hold_time,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'best_trade': {
                'symbol': best_trade['symbol'],
                'pnl': best_trade['pnl'],
                'pnl_pct': best_trade['pnl_pct']
            } if best_trade is not None else None,
            'worst_trade': {
                'symbol': worst_trade['symbol'],
                'pnl': worst_trade['pnl'],
                'pnl_pct': worst_trade['pnl_pct']
            } if worst_trade is not None else None,
            'num_winning': len(winning_trades),
            'num_losing': len(losing_trades),
            'expectancy': (avg_win * len(winning_trades) + avg_loss * len(losing_trades)) / total_trades if total_trades > 0 else 0
        }

        return metrics

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_win_pct': 0,
            'avg_loss_pct': 0,
            'avg_hold_minutes': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'best_trade': None,
            'worst_trade': None,
            'num_winning': 0,
            'num_losing': 0,
            'expectancy': 0
        }

    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get most recent trades."""
        return self.trade_history[-limit:] if self.trade_history else []

    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""

        metrics = self.calculate_metrics()

        # Extract best/worst trade info
        best_sym = metrics['best_trade']['symbol'] if metrics['best_trade'] else 'N/A'
        best_pnl = metrics['best_trade']['pnl'] if metrics['best_trade'] else 0
        best_pnl_pct = metrics['best_trade']['pnl_pct'] if metrics['best_trade'] else 0

        worst_sym = metrics['worst_trade']['symbol'] if metrics['worst_trade'] else 'N/A'
        worst_pnl = metrics['worst_trade']['pnl'] if metrics['worst_trade'] else 0
        worst_pnl_pct = metrics['worst_trade']['pnl_pct'] if metrics['worst_trade'] else 0

        report = f"""
{'='*80}
PERFORMANCE REPORT
{'='*80}

📊 OVERALL PERFORMANCE
{'─'*80}
Total Trades: {metrics['total_trades']}
Win Rate: {metrics['win_rate']:.1%}
Total P&L: ${metrics['total_pnl']:,.2f}
Expectancy: ${metrics['expectancy']:,.2f} per trade

💰 PROFIT/LOSS ANALYSIS
{'─'*80}
Winning Trades: {metrics['num_winning']}
Losing Trades: {metrics['num_losing']}
Average Win: ${metrics['avg_win']:,.2f} ({metrics['avg_win_pct']:.2f}%)
Average Loss: ${metrics['avg_loss']:,.2f} ({metrics['avg_loss_pct']:.2f}%)
Profit Factor: {metrics['profit_factor']:.2f}

📈 RISK METRICS
{'─'*80}
Max Drawdown: {metrics['max_drawdown']:.2%}
Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
Average Hold Time: {metrics['avg_hold_minutes']:.1f} minutes

🏆 BEST TRADE
{'─'*80}
{best_sym}: ${best_pnl:,.2f} ({best_pnl_pct:.2f}%)

📉 WORST TRADE
{'─'*80}
{worst_sym}: ${worst_pnl:,.2f} ({worst_pnl_pct:.2f}%)

{'='*80}
"""

        return report

    def export_to_csv(self, filename: str = None):
        """Export trade history to CSV."""
        if filename is None:
            filename = self.data_dir / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        if self.trade_history:
            df = pd.DataFrame(self.trade_history)
            df.to_csv(filename, index=False)
            logger.info(f"Exported {len(self.trade_history)} trades to {filename}")

        return filename
