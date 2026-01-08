"""
Market data fetcher using Alpaca Data API (Basic plan included).
"""
import pandas as pd
from typing import List
from datetime import datetime, timedelta
from loguru import logger

try:
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_DATA_AVAILABLE = True
except ImportError:
    ALPACA_DATA_AVAILABLE = False
    logger.warning("alpaca-py not installed")

from config import settings


def get_market_data_alpaca(symbols: List[str], days: int = 100) -> pd.DataFrame:
    """Get market data from Alpaca Data API (Basic plan)."""
    if not ALPACA_DATA_AVAILABLE:
        return pd.DataFrame()

    try:
        # Create data client
        data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key
        )

        # Calculate date range
        end = datetime.now()
        start = end - timedelta(days=days + 30)  # Extra buffer

        logger.info(f"Fetching {len(symbols)} symbols from Alpaca...")

        # Create request - use IEX feed (included in Basic plan, NOT SIP)
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed='iex'  # Use IEX data instead of SIP (included in Basic plan)
        )

        # Fetch bars
        bars = data_client.get_stock_bars(request)

        # Check if we got data
        if bars is None:
            logger.warning("No bars returned from Alpaca Data API")
            return pd.DataFrame()

        # Access the data dictionary from BarSet
        bars_dict = bars.data if hasattr(bars, 'data') else bars

        # Convert to DataFrame
        df_list = []
        for symbol in symbols:
            if symbol not in bars_dict:
                logger.debug(f"No data for {symbol}")
                continue

            symbol_bars = list(bars_dict[symbol])
            if not symbol_bars:
                logger.debug(f"No bars for {symbol}")
                continue

            for bar in symbol_bars:
                df_list.append({
                    "symbol": symbol,
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": getattr(bar, 'vwap', 0),
                })

        if not df_list:
            logger.warning("No data extracted from bars")
            return pd.DataFrame()

        df = pd.DataFrame(df_list)
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)

        # Take last N bars per symbol
        df = df.groupby("symbol").tail(days)

        logger.info(f"✓ Retrieved {len(df)} bars from Alpaca Data API")

        return df

    except Exception as e:
        logger.error(f"Error from Alpaca Data API: {e}")
        return pd.DataFrame()


def get_market_data_yfinance(symbols: List[str], days: int = 100) -> pd.DataFrame:
    """Get market data from yfinance (backup)."""
    try:
        import yfinance as yf
        logger.info("⚠ Using yfinance as backup (Alpaca data unavailable)")

        data_list = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=start_date, end=end_date, interval="1d")

                if not hist.empty:
                    hist_reset = hist.reset_index()
                    for _, row in hist_reset.tail(days).iterrows():
                        data_list.append({
                            "symbol": symbol,
                            "timestamp": row["Date"],
                            "open": row["Open"],
                            "high": row["High"],
                            "low": row["Low"],
                            "close": row["Close"],
                            "volume": row["Volume"],
                        })
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol} from yfinance: {e}")
                continue

        if data_list:
            df = pd.DataFrame(data_list)
            df = df.set_index("timestamp")
            df.index = pd.to_datetime(df.index)
            logger.info(f"Retrieved {len(df)} bars from yfinance")
            return df

    except ImportError:
        logger.error("yfinance not available")

    return pd.DataFrame()


def get_market_data(symbols: List[str], days: int = 100) -> pd.DataFrame:
    """Get market data - uses Alpaca Data API (Basic plan)."""
    logger.info("Fetching market data from Alpaca Data API (Basic plan included)...")

    df = get_market_data_alpaca(symbols, days)

    if not df.empty:
        return df

    # Fallback to yfinance only if Alpaca completely fails
    logger.warning("⚠ Alpaca Data API unavailable - using yfinance backup")
    df = get_market_data_yfinance(symbols, days)

    return df
