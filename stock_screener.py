"""
TURBO Stock Screener - EXTREME SPEED OPTIMIZED
Fetches ALL tradeable stocks with maximum performance.
"""
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from loguru import logger

from alpaca_client import AlpacaClient
from market_data import get_market_data
from config import settings


class TurboStockScreener:
    """Ultra-fast stock screener with parallel processing."""

    def __init__(self, max_workers: int = 10):
        """Initialize screener with parallel workers."""
        self.client = AlpacaClient()
        self.max_workers = max_workers
        logger.info(f"Turbo Screener initialized with {max_workers} workers")

    def get_all_tradeable_stocks(self) -> List[str]:
        """Fetch all tradeable stocks from Alpaca - OPTIMIZED."""
        logger.info("⚡ Fetching ALL tradeable stocks from Alpaca...")

        try:
            # Get all assets
            assets = self.client.get_all_assets()

            # Fast filtering with list comprehension
            tradeable_stocks = [
                asset["symbol"]
                for asset in assets
                if (
                    asset.get("class") == "us_equity"
                    and asset.get("status") == "active"
                    and asset.get("tradable", False)
                    and asset.get("exchange") in ["NASDAQ", "NYSE", "ARCA", "AMEX"]
                    and not self._should_skip_symbol(asset["symbol"])
                )
            ]

            logger.info(f"✓ Found {len(tradeable_stocks):,} tradeable stocks")
            return tradeable_stocks

        except Exception as e:
            logger.error(f"Error fetching tradeable stocks: {e}")
            # Fallback to major stocks
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]

    def _should_skip_symbol(self, symbol: str) -> bool:
        """Fast filter for problematic symbols."""
        if len(symbol) < 1:
            return True
        if any(char.isdigit() for char in symbol):
            return True
        if symbol.startswith(("U", "W", "Y")):
            return True
        if symbol.endswith("W") or "+" in symbol or "=" in symbol:
            return True
        return False

    def screen_all_stocks(self, max_stocks: int = None) -> List[str]:
        """
        Screen ALL stocks with extreme speed.

        Args:
            max_stocks: Maximum to return (None = ALL stocks)
        """
        logger.info("=" * 80)
        logger.info("⚡ TURBO SCREENING - ALL STOCKS")
        logger.info("=" * 80)
        logger.info("")

        # Get all stocks
        all_stocks = self.get_all_tradeable_stocks()

        if not all_stocks:
            return []

        logger.info(f"⚡ Filtering {len(all_stocks):,} stocks...")
        logger.info("")

        # Apply quick filters WITHOUT fetching data (FAST)
        filtered = self._apply_quick_filters(all_stocks)

        logger.info(f"✓ Quick filter: {len(filtered):,} stocks remain")
        logger.info("")

        # Apply data-based filters if needed
        # For MAXIMUM SPEED, we skip data filtering and just use quick filters
        # This allows scanning THOUSANDS of stocks in seconds

        if max_stocks and len(filtered) > max_stocks:
            logger.info(f"⚡ Limiting to {max_stocks:,} stocks...")
            # Take first N stocks (or could sort by market cap if we had that data)
            filtered = filtered[:max_stocks]

        logger.info("=" * 80)
        logger.info(f"⚡ SCREENING COMPLETE: {len(filtered):,} stocks ready")
        logger.info("=" * 80)

        return filtered

    def _apply_quick_filters(self, symbols: List[str]) -> List[str]:
        """
        Apply quick filters WITHOUT fetching data.
        This makes it EXTREMELY FAST.
        """
        # Quick symbol-based filters
        filtered = [
            s for s in symbols
            if (
                len(s) >= 2  # Minimum 2 chars
                and len(s) <= 5  # Maximum 5 chars
                and s.isalpha()  # Letters only
                and not s.endswith(("W", "U", "R"))  # No warrants/units
                and s[0].isalpha()  # First char is letter
            )
        ]

        return filtered

    def filter_with_data_parallel(
        self,
        symbols: List[str],
        min_volume: int = 500000,
        min_price: float = 5.0,
        max_price: float = 500.0,
        batch_size: int = 100,
        sort_by_liquidity: bool = False
    ) -> List[str]:
        """
        Filter with data fetching in parallel - OPTIMIZED FOR SPEED.

        Args:
            sort_by_liquidity: If True, sort results by dollar volume (liquidity)
        """
        logger.info(f"⚡ Parallel filtering {len(symbols):,} stocks...")

        filtered = []

        # Process in parallel batches
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create batches
            batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]

            # Submit all batches
            future_to_batch = {
                executor.submit(self._filter_batch, batch, min_volume, min_price, max_price): i
                for i, batch in enumerate(batches)
            }

            # Collect results
            completed = 0
            for future in as_completed(future_to_batch):
                completed += 1
                batch_num = future_to_batch[future]

                if completed % 10 == 0:
                    logger.info(f"  Progress: {completed}/{len(batches)} batches...")

                try:
                    result = future.result()
                    filtered.extend(result)
                except Exception as e:
                    logger.warning(f"  Error in batch {batch_num}: {e}")

        logger.info(f"✓ Filtered to {len(filtered):,} stocks")

        # Sort by liquidity if requested (for aggressive early exit)
        if sort_by_liquidity and filtered:
            logger.info("⚡ Sorting stocks by liquidity (dollar volume)...")
            filtered = self._sort_by_liquidity(filtered)
            logger.info(f"✓ Stocks prioritized by liquidity")

        return filtered

    def _filter_batch(
        self,
        symbols: List[str],
        min_volume: int,
        min_price: float,
        max_price: float
    ) -> List[str]:
        """Filter a batch of symbols."""
        try:
            df = get_market_data(symbols, days=5)  # Only need 5 days for filtering

            if df.empty:
                return []

            passed = []

            for symbol in symbols:
                symbol_df = df[df["symbol"] == symbol]

                if symbol_df.empty or len(symbol_df) < 2:
                    continue

                latest = symbol_df.iloc[-1]

                # Price filter
                price = latest["close"]
                if price < min_price or price > max_price:
                    continue

                # Volume filter
                avg_volume = symbol_df["volume"].tail(3).mean()
                if avg_volume < min_volume:
                    continue

                # Dollar volume filter (increased to $10M for better liquidity)
                dollar_volume = price * avg_volume
                if dollar_volume < 10_000_000:  # $10M minimum (enhanced from $5M)
                    continue

                # Additional liquidity filter: minimum 1M shares daily volume
                if avg_volume < 1_000_000:
                    continue

                # Avoid penny stocks and illiquid high-priced stocks
                if price < 1.0:  # Minimum $1 share price
                    continue
                if price > 500:  # Maximum $500 (avoid illiquid high-priced)
                    continue

                # Check volatility (avoid dead stocks and crazy volatile ones)
                returns = symbol_df["close"].pct_change().dropna()
                if len(returns) > 0:
                    volatility = returns.std()
                    if volatility < 0.01:  # Less than 1% daily move = dead stock
                        continue
                    if volatility > 0.10:  # More than 10% daily move = too risky
                        continue

                passed.append(symbol)

            return passed

        except Exception as e:
            logger.warning(f"Error filtering batch: {e}")
            return []

    def _sort_by_liquidity(self, symbols: List[str], top_n: int = None) -> List[str]:
        """
        Sort stocks by liquidity (dollar volume) for priority scanning.

        Args:
            symbols: List of stock symbols to sort
            top_n: Only return top N stocks (None = return all sorted)

        Returns:
            List of symbols sorted by liquidity (highest first)
        """
        try:
            # Fetch recent data for all symbols
            df = get_market_data(symbols, days=5)

            if df.empty:
                return symbols

            # Calculate dollar volume for each symbol
            liquidity_map = {}

            for symbol in symbols:
                symbol_df = df[df["symbol"] == symbol]

                if symbol_df.empty or len(symbol_df) < 2:
                    continue

                latest = symbol_df.iloc[-1]

                # Calculate dollar volume
                price = latest["close"]
                avg_volume = symbol_df["volume"].tail(3).mean()
                dollar_volume = price * avg_volume

                liquidity_map[symbol] = dollar_volume

            # Sort by dollar volume (descending)
            sorted_symbols = sorted(
                liquidity_map.keys(),
                key=lambda s: liquidity_map[s],
                reverse=True
            )

            # Log top 10 for visibility
            logger.info("  Top 10 by liquidity:")
            for i, symbol in enumerate(sorted_symbols[:10], 1):
                logger.info(f"    {i}. {symbol}: ${liquidity_map[symbol]:,.0f}")

            # Return top N if specified
            if top_n and len(sorted_symbols) > top_n:
                sorted_symbols = sorted_symbols[:top_n]

            return sorted_symbols

        except Exception as e:
            logger.warning(f"Error sorting by liquidity: {e}")
            return symbols


# Global instance
turbo_screener = TurboStockScreener(max_workers=10)
