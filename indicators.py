"""
Technical indicators for algorithmic trading.
No ML - pure mathematical calculations.
"""
import pandas as pd
import numpy as np
from typing import Dict, List


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to dataframe."""
    if df.empty or len(df) < 50:
        return df

    df = df.copy()

    # Moving Averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["sma_200"] = df["close"].rolling(window=200).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (df["bb_std"] * 2)
    df["bb_lower"] = df["bb_middle"] - (df["bb_std"] * 2)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_percent"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # Stochastic Oscillator
    low_14 = df["low"].rolling(window=14).min()
    high_14 = df["high"].rolling(window=14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14)
    df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

    # ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()

    # Volume indicators
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # Price Rate of Change
    df["roc"] = df["close"].pct_change(periods=10) * 100

    # Williams %R
    high_14 = df["high"].rolling(window=14).max()
    low_14 = df["low"].rolling(window=14).min()
    df["williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14)

    # CCI (Commodity Channel Index)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    tp_sma = tp.rolling(window=20).mean()
    mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
    df["cci"] = (tp - tp_sma) / (0.015 * mad)

    # Momentum
    df["momentum"] = df["close"] - df["close"].shift(10)

    # Returns
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(window=20).std()

    return df


def get_latest_signal(df: pd.DataFrame) -> Dict[str, float]:
    """Get latest indicator values and signals."""
    if df.empty or len(df) < 50:
        return {}

    df = add_indicators(df)
    latest = df.iloc[-1]

    return {
        "close": latest["close"],
        "volume": latest["volume"],
        "returns": latest.get("returns", 0),

        # Moving averages
        "sma_20": latest.get("sma_20", 0),
        "sma_50": latest.get("sma_50", 0),
        "sma_200": latest.get("sma_200", 0),

        # RSI
        "rsi": latest.get("rsi", 50),

        # MACD
        "macd": latest.get("macd", 0),
        "macd_signal": latest.get("macd_signal", 0),
        "macd_hist": latest.get("macd_hist", 0),

        # Bollinger Bands
        "bb_upper": latest.get("bb_upper", 0),
        "bb_lower": latest.get("bb_lower", 0),
        "bb_percent": latest.get("bb_percent", 0.5),

        # Stochastic
        "stoch_k": latest.get("stoch_k", 50),
        "stoch_d": latest.get("stoch_d", 50),

        # ATR
        "atr": latest.get("atr", 0),

        # Volume
        "volume_ratio": latest.get("volume_ratio", 1),

        # ROC
        "roc": latest.get("roc", 0),

        # Williams %R
        "williams_r": latest.get("williams_r", -50),

        # CCI
        "cci": latest.get("cci", 0),
    }


def detect_trend(df: pd.DataFrame) -> str:
    """Detect overall trend: uptrend, downtrend, or sideways."""
    if df.empty or len(df) < 50:
        return "unknown"

    df = add_indicators(df)
    recent = df.tail(20)

    # Check moving average alignment
    sma_align = (
        recent["close"].iloc[-1] > recent["sma_20"].iloc[-1] >
        recent["sma_50"].iloc[-1] > recent["sma_200"].iloc[-1]
    )

    downtrend = (
        recent["close"].iloc[-1] < recent["sma_20"].iloc[-1] <
        recent["sma_50"].iloc[-1] < recent["sma_200"].iloc[-1]
    )

    # Check MACD
    macd_bullish = recent["macd_hist"].iloc[-1] > 0

    # Check price momentum
    price_change = (recent["close"].iloc[-1] - recent["close"].iloc[0]) / recent["close"].iloc[0]

    # Determine trend
    if sma_align and macd_bullish and price_change > 0.02:
        return "uptrend"
    elif downtrend and not macd_bullish and price_change < -0.02:
        return "downtrend"
    else:
        return "sideways"


def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """Calculate support and resistance levels."""
    if df.empty or len(df) < window:
        return {"support": 0, "resistance": 0}

    recent = df.tail(window)

    # Support: recent lows
    support = recent["low"].min()

    # Resistance: recent highs
    resistance = recent["high"].max()

    # Current price
    current = df["close"].iloc[-1]

    return {
        "support": support,
        "resistance": resistance,
        "current": current,
        "support_distance": (current - support) / current,
        "resistance_distance": (resistance - current) / current,
    }
