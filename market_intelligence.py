"""
Advanced Market Intelligence
Sector analysis, market breadth, correlation analysis, and more.
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from collections import defaultdict

from market_data import get_market_data
from config import settings


# Sector mapping (simplified - can be expanded)
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology", "META": "Technology",
    "NVDA": "Technology", "AMD": "Technology", "INTC": "Technology", "CSCO": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "ORCL": "Technology", "AVGO": "Technology",

    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare", "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "LLY": "Healthcare", "BMY": "Healthcare",

    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "C": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials", "SCHW": "Financials",
    "AXP": "Financials", "CB": "Financials",

    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary", "BKNG": "Consumer Discretionary",

    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "PXD": "Energy", "MPC": "Energy", "PSX": "Energy",

    # Industrials
    "CAT": "Industrials", "HON": "Industrials", "UNP": "Industrials", "UPS": "Industrials",
    "BA": "Industrials", "RTX": "Industrials", "LMT": "Industrials", "GE": "Industrials",
    "MMM": "Industrials", "DE": "Industrials",

    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "D": "Utilities",
    "EXC": "Utilities", "AEP": "Utilities", "NGG": "Utilities", "XEL": "Utilities",

    # Real Estate
    "AMT": "Real Estate", "PLD": "Real Estate", "CCI": "Real Estate", "EQIX": "Real Estate",
    "PRO": "Real Estate", "DLR": "Real Estate", "O": "Real Estate", "VNQ": "Real Estate",

    # Materials
    "SHW": "Materials", "APD": "Materials", "FCX": "Materials", "NEM": "Materials",
    "DOW": "Materials", "DD": "Materials", "BHP": "Materials", "RIO": "Materials",

    # Communication Services
    "GOOG": "Communication Services", "DIS": "Communication Services", "CMCSA": "Communication Services",
    "T": "Communication Services", "VZ": "Communication Services", "NFLX": "Communication Services",
    "TWTR": "Communication Services", "EA": "Communication Services",
}


class MarketIntelligence:
    """Advanced market intelligence analysis."""

    def __init__(self):
        """Initialize intelligence system."""
        self.sector_performance = {}
        self.market_breadth = {}

    def analyze_sector_performance(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Analyze sector performance.

        Returns:
            Dict mapping sector to performance metrics
        """
        logger.info("Analyzing sector performance...")

        # Group symbols by sector
        sector_stocks = defaultdict(list)
        for symbol in symbols:
            sector = SECTOR_MAP.get(symbol, "Other")
            sector_stocks[sector].append(symbol)

        # Analyze each sector
        sector_data = {}

        for sector, sector_symbols in sector_stocks.items():
            if len(sector_symbols) < 3:  # Skip sectors with too few stocks
                continue

            try:
                # Get data for sector stocks
                df = get_market_data(sector_symbols, days=20)

                if df.empty:
                    continue

                # Calculate sector metrics
                daily_returns = []
                volumes = []

                for symbol in sector_symbols:
                    symbol_df = df[df["symbol"] == symbol]
                    if not symbol_df.empty and len(symbol_df) >= 2:
                        # Daily return
                        ret = (symbol_df["close"].iloc[-1] / symbol_df["close"].iloc[-2] - 1) * 100
                        daily_returns.append(ret)
                        volumes.append(symbol_df["volume"].iloc[-1])

                if daily_returns:
                    sector_data[sector] = {
                        "avg_return": np.mean(daily_returns),
                        "stocks_up": sum(1 for r in daily_returns if r > 0),
                        "stocks_down": sum(1 for r in daily_returns if r < 0),
                        "total_stocks": len(daily_returns),
                        "avg_volume": np.mean(volumes) if volumes else 0,
                        "momentum": "bullish" if np.mean(daily_returns) > 0.5 else "bearish" if np.mean(daily_returns) < -0.5 else "neutral"
                    }

            except Exception as e:
                logger.warning(f"Error analyzing sector {sector}: {e}")
                continue

        # Sort by performance
        self.sector_performance = dict(
            sorted(sector_data.items(), key=lambda x: x[1]["avg_return"], reverse=True)
        )

        return self.sector_performance

    def calculate_market_breadth(self, signals: List[Dict]) -> Dict:
        """
        Calculate market breadth indicators.

        Args:
            signals: List of signal dictionaries from scanning

        Returns:
            Market breadth metrics
        """
        logger.info("Calculating market breadth...")

        if not signals:
            return {}

        # Count signals
        buy_signals = sum(1 for s in signals if s["signal"] in ["buy", "strong_buy"])
        sell_signals = sum(1 for s in signals if s["signal"] in ["sell", "strong_sell"])
        hold_signals = sum(1 for s in signals if s["signal"] == "hold")

        total = len(signals)

        # Calculate breadth ratios
        breadth = {
            "total_stocks": total,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "buy_ratio": buy_signals / total if total > 0 else 0,
            "sell_ratio": sell_signals / total if total > 0 else 0,
            "breadth_thrust": (buy_signals - sell_signals) / total if total > 0 else 0,
        }

        # Market sentiment
        if breadth["breadth_thrust"] > 0.3:
            breadth["market_sentiment"] = "strongly_bullish"
        elif breadth["breadth_thrust"] > 0.1:
            breadth["market_sentiment"] = "bullish"
        elif breadth["breadth_thrust"] < -0.3:
            breadth["market_sentiment"] = "strongly_bearish"
        elif breadth["breadth_thrust"] < -0.1:
            breadth["market_sentiment"] = "bearish"
        else:
            breadth["market_sentiment"] = "neutral"

        self.market_breadth = breadth
        return breadth

    def find_correlation_clusters(self, symbols: List[str]) -> Dict:
        """
        Find correlation clusters (stocks that move together).

        Returns:
            Dict with correlation information
        """
        logger.info("Analyzing stock correlations...")

        try:
            # Get data
            df = get_market_data(symbols, days=50)

            if df.empty:
                return {}

            # Create pivot table
            pivot_df = df.pivot_table(
                index="timestamp",
                columns="symbol",
                values="close"
            )

            # Calculate returns
            returns = pivot_df.pct_change().dropna()

            # Calculate correlation matrix
            corr_matrix = returns.corr()

            # Find highly correlated pairs
            correlated_pairs = []

            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]

                    if abs(corr) > 0.7:  # High correlation threshold
                        correlated_pairs.append({
                            "symbol1": corr_matrix.columns[i],
                            "symbol2": corr_matrix.columns[j],
                            "correlation": corr
                        })

            # Sort by correlation
            correlated_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

            return {
                "high_correlation_pairs": correlated_pairs[:20],
                "total_pairs": len(correlated_pairs)
            }

        except Exception as e:
            logger.error(f"Error analyzing correlations: {e}")
            return {}

    def get_top_sectors(self, top_n: int = 3) -> List[str]:
        """Get top performing sectors."""
        if not self.sector_performance:
            return []

        sorted_sectors = sorted(
            self.sector_performance.items(),
            key=lambda x: x[1]["avg_return"],
            reverse=True
        )

        return [sector for sector, _ in sorted_sectors[:top_n]]

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector for a given symbol."""
        return SECTOR_MAP.get(symbol, "Other")

    def should_avoid_correlated_trade(
        self,
        new_symbol: str,
        current_positions: List[str],
        max_correlation: float = 0.7
    ) -> Tuple[bool, str]:
        """
        Check if new symbol is too correlated with existing positions.

        Returns:
            Tuple of (should_avoid, reason)
        """
        if not current_positions:
            return False, ""

        # Check if same sector
        new_sector = self.get_sector_for_symbol(new_symbol)

        for position in current_positions:
            pos_sector = self.get_sector_for_symbol(position)

            if new_sector == pos_sector and new_sector != "Other":
                return True, f"Same sector as {position} ({pos_sector})"

        return False, ""

    def print_intelligence_report(self):
        """Print intelligence summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("MARKET INTELLIGENCE REPORT")
        logger.info("=" * 60)

        # Sector performance
        if self.sector_performance:
            logger.info("\n📊 SECTOR PERFORMANCE:")
            logger.info("-" * 60)

            for sector, data in list(self.sector_performance.items())[:5]:
                momentum_emoji = "📈" if data["momentum"] == "bullish" else "📉" if data["momentum"] == "bearish" else "➡️"
                logger.info(
                    f"  {momentum_emoji} {sector}: {data['avg_return']:+.2f}% "
                    f"({data['stocks_up']}/{data['total_stocks']} up) - {data['momentum']}"
                )

        # Market breadth
        if self.market_breadth:
            logger.info("\n📊 MARKET BREADTH:")
            logger.info("-" * 60)

            breadth = self.market_breadth
            sentiment_emoji = {
                "strongly_bullish": "🚀",
                "bullish": "📈",
                "neutral": "➡️",
                "bearish": "📉",
                "strongly_bearish": "💥"
            }

            logger.info(f"  Sentiment: {sentiment_emoji.get(breadth['market_sentiment'], '')} {breadth['market_sentiment'].upper()}")
            logger.info(f"  Buy Signals: {breadth['buy_signals']} ({breadth['buy_ratio']:.1%})")
            logger.info(f"  Sell Signals: {breadth['sell_signals']} ({breadth['sell_ratio']:.1%})")
            logger.info(f"  Breadth Thrust: {breadth['breadth_thrust']:+.2%}")

        logger.info("")
        logger.info("=" * 60)


# Global instance
market_intel = MarketIntelligence()
