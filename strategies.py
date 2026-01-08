"""
Advanced Algorithmic Trading Strategies
Multi-strategy ensemble with market adaptation and maximum intelligence.
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

from indicators import add_indicators, get_latest_signal, detect_trend, calculate_support_resistance
from config import settings


class Signal(Enum):
    """Trading signal types."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class MarketRegime(Enum):
    """Market regime types."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"


class AdvancedStrategy:
    """Base advanced trading strategy with intelligence features."""

    def __init__(self, name: str):
        """Initialize strategy."""
        self.name = name
        self.signals_generated = 0

    def analyze(self, symbol: str, df: pd.DataFrame, market_regime: MarketRegime) -> Optional[Dict]:
        """Analyze symbol and generate signal. Override in subclass."""
        raise NotImplementedError

    def calculate_volatility_regime(self, df: pd.DataFrame) -> str:
        """Determine volatility regime."""
        if len(df) < 20:
            return "normal"

        atr = df["atr"].iloc[-1] if "atr" in df else 0
        price = df["close"].iloc[-1]
        atr_pct = (atr / price) if price > 0 else 0

        if atr_pct > 0.03:
            return "high"
        elif atr_pct < 0.01:
            return "low"
        return "normal"

    def detect_volume_surge(self, df: pd.DataFrame) -> bool:
        """Detect abnormal volume surge."""
        if len(df) < 20:
            return False

        recent_vol = df["volume"].iloc[-1]
        avg_vol = df["volume"].iloc[-20:].mean()

        return recent_vol > avg_vol * 2

    def calculate_liquidity_score(self, df: pd.DataFrame) -> float:
        """Calculate liquidity score (0-1)."""
        if len(df) < 20:
            return 0.5

        # Volume consistency
        vol_std = df["volume"].iloc[-20:].std()
        vol_mean = df["volume"].iloc[-20:].mean()
        vol_consistency = 1 - min(vol_std / vol_mean, 1) if vol_mean > 0 else 0

        # Average dollar volume
        avg_dollar_vol = (df["close"] * df["volume"]).iloc[-20:].mean()
        dollar_score = min(avg_dollar_vol / 10_000_000, 1)  # $10M daily = perfect

        return (vol_consistency + dollar_score) / 2


class MomentumStrategy(AdvancedStrategy):
    """Advanced momentum strategy with multi-factor confirmation."""

    def __init__(self):
        super().__init__("Advanced Momentum")

    def analyze(self, symbol: str, df: pd.DataFrame, market_regime: MarketRegime) -> Optional[Dict]:
        """Analyze for momentum signals."""
        if df.empty or len(df) < 50:
            return None

        indicators = get_latest_signal(df)
        trend = detect_trend(df)
        volatility_regime = self.calculate_volatility_regime(df)

        score = 0
        reasons = []

        # 1. Trend confirmation (weighted by regime)
        trend_weight = 1.5 if market_regime == MarketRegime.BULL else 1.0
        if trend == "uptrend":
            score += 2 * trend_weight
            reasons.append("Strong uptrend")
        elif trend == "downtrend":
            score -= 2 * trend_weight
            reasons.append("Strong downtrend")

        # 2. MACD momentum with histogram
        macd_hist = indicators.get("macd_hist", 0)
        if macd_hist > 0:
            # Increasing histogram = strengthening momentum
            if len(df) > 1 and "macd_hist" in df.columns:
                prev_hist = df["macd_hist"].iloc[-2]
                if macd_hist > prev_hist:
                    score += 1.5
                    reasons.append("MACD accelerating")
                else:
                    score += 1
                    reasons.append("MACD bullish")
            else:
                score += 1
                reasons.append("MACD bullish")
        else:
            score -= 1
            reasons.append("MACD bearish")

        # 3. Multi-timeframe moving average alignment
        sma_20 = indicators.get("sma_20", 0)
        sma_50 = indicators.get("sma_50", 0)
        sma_200 = indicators.get("sma_200", 0)
        price = indicators.get("close", 0)

        # Golden cross (50 above 200) in uptrend
        if sma_50 > sma_200 and trend == "uptrend":
            score += 1.5
            reasons.append("Golden cross")

        # Price above MAs
        if price > sma_20 > sma_50:
            score += 1
            reasons.append("Price above MAs")
        elif price < sma_20 < sma_50:
            score -= 1
            reasons.append("Price below MAs")

        # 4. Rate of Change with acceleration
        roc = indicators.get("roc", 0)
        if roc > 3:
            score += 1.5
            reasons.append(f"Strong ROC: {roc:.1f}%")
        elif roc > 1:
            score += 0.5
            reasons.append(f"Positive ROC: {roc:.1f}%")
        elif roc < -3:
            score -= 1.5
            reasons.append(f"Weak ROC: {roc:.1f}%")
        elif roc < -1:
            score -= 0.5
            reasons.append(f"Negative ROC: {roc:.1f}%")

        # 5. Volume confirmation (critical for momentum)
        vol_ratio = indicators.get("volume_ratio", 1)
        if vol_ratio > 2:
            score += 1
            reasons.append("Very high volume")
        elif vol_ratio > 1.5:
            score += 0.5
            reasons.append("High volume")

        # Detect volume surge
        if self.detect_volume_surge(df):
            score += 0.5
            reasons.append("Volume surge detected")

        # 6. RSI confirmation (not overbought)
        rsi = indicators.get("rsi", 50)
        if 50 < rsi < 70:
            score += 0.5
            reasons.append(f"RSI bullish zone: {rsi:.0f}")
        elif rsi > 70:
            score -= 0.5  # Overbought warning
            reasons.append(f"RSI overbought: {rsi:.0f}")

        # 7. Volatility adjustment
        if volatility_regime == "high" and score > 0:
            score *= 0.9  # Reduce signal in high volatility
            reasons.append("High volatility - reduced confidence")

        # 8. Market regime adjustment
        if market_regime == MarketRegime.BEAR and score > 0:
            score *= 0.7  # More cautious in bear market
            reasons.append("Bear market - cautious")
        elif market_regime == MarketRegime.BULL and score > 0:
            score *= 1.2  # More aggressive in bull market
            reasons.append("Bull market - confident")

        # Liquidity check
        liquidity = self.calculate_liquidity_score(df)
        if liquidity < 0.3:
            score *= 0.5  # Reduce signal for illiquid stocks
            reasons.append("Low liquidity")

        signal, confidence = self._score_to_signal(score)

        return {
            "symbol": symbol,
            "signal": signal.value,
            "confidence": confidence,
            "score": score,
            "price": price,
            "reasons": reasons,
            "strategy": self.name,
            "volatility_regime": volatility_regime,
            "liquidity_score": liquidity,
        }

    def _score_to_signal(self, score: float) -> Tuple[Signal, float]:
        """Convert score to signal and confidence."""
        confidence = min(abs(score) / 5, 1.0)

        if score >= 4:
            return Signal.STRONG_BUY, confidence
        elif score >= 2:
            return Signal.BUY, confidence
        elif score <= -4:
            return Signal.STRONG_SELL, confidence
        elif score <= -2:
            return Signal.SELL, confidence
        else:
            return Signal.HOLD, 0.0


class MeanReversionStrategy(AdvancedStrategy):
    """Advanced mean reversion with Bollinger Band and statistical analysis."""

    def __init__(self):
        super().__init__("Advanced Mean Reversion")

    def analyze(self, symbol: str, df: pd.DataFrame, market_regime: MarketRegime) -> Optional[Dict]:
        """Analyze for mean reversion signals."""
        if df.empty or len(df) < 50:
            return None

        indicators = get_latest_signal(df)
        sr = calculate_support_resistance(df)
        volatility_regime = self.calculate_volatility_regime(df)

        score = 0
        reasons = []

        # 1. RSI extreme levels
        rsi = indicators.get("rsi", 50)
        if rsi < 25:
            score += 2.5
            reasons.append(f"RSI extremely oversold: {rsi:.0f}")
        elif rsi < 30:
            score += 2
            reasons.append(f"RSI oversold: {rsi:.0f}")
        elif rsi < 35:
            score += 1
            reasons.append(f"RSI approaching oversold: {rsi:.0f}")
        elif rsi > 75:
            score -= 2.5
            reasons.append(f"RSI extremely overbought: {rsi:.0f}")
        elif rsi > 70:
            score -= 2
            reasons.append(f"RSI overbought: {rsi:.0f}")
        elif rsi > 65:
            score -= 1
            reasons.append(f"RSI approaching overbought: {rsi:.0f}")

        # 2. Bollinger Band extreme
        bb_percent = indicators.get("bb_percent", 0.5)
        bb_lower = indicators.get("bb_lower", 0)
        bb_upper = indicators.get("bb_upper", 0)
        price = indicators.get("close", 0)

        if bb_percent < 0.05:  # Very close to lower band
            score += 2
            reasons.append(f"Near lower BB: {bb_percent:.1%}")
        elif bb_percent < 0.1:
            score += 1.5
            reasons.append(f"Approaching lower BB: {bb_percent:.1%}")
        elif bb_percent > 0.95:  # Very close to upper band
            score -= 2
            reasons.append(f"Near upper BB: {bb_percent:.1%}")
        elif bb_percent > 0.9:
            score -= 1.5
            reasons.append(f"Approaching upper BB: {bb_percent:.1%}")

        # 3. Support/Resistance proximity
        if sr["support_distance"] < 0.01:  # Within 1%
            score += 2
            reasons.append(f"At support: ${sr['support']:.2f}")
        elif sr["support_distance"] < 0.02:
            score += 1
            reasons.append(f"Near support: ${sr['support']:.2f}")

        if sr["resistance_distance"] < 0.01:
            score -= 2
            reasons.append(f"At resistance: ${sr['resistance']:.2f}")
        elif sr["resistance_distance"] < 0.02:
            score -= 1
            reasons.append(f"Near resistance: ${sr['resistance']:.2f}")

        # 4. Stochastic extreme
        stoch_k = indicators.get("stoch_k", 50)
        stoch_d = indicators.get("stoch_d", 50)

        if stoch_k < 15 and stoch_d < 15:
            score += 1.5
            reasons.append("Stochastic extremely oversold")
        elif stoch_k < 20:
            score += 1
            reasons.append("Stochastic oversold")
        elif stoch_k > 85 and stoch_d > 85:
            score -= 1.5
            reasons.append("Stochastic extremely overbought")
        elif stoch_k > 80:
            score -= 1
            reasons.append("Stochastic overbought")

        # 5. Williams %R
        williams_r = indicators.get("williams_r", -50)
        if williams_r < -90:
            score += 1
            reasons.append(f"Williams %R extreme oversold: {williams_r:.0f}")
        elif williams_r < -80:
            score += 0.5
            reasons.append(f"Williams %R oversold: {williams_r:.0f}")
        elif williams_r > -10:
            score -= 1
            reasons.append(f"Williams %R extreme overbought: {williams_r:.0f}")
        elif williams_r > -20:
            score -= 0.5
            reasons.append(f"Williams %R overbought: {williams_r:.0f}")

        # 6. CCI (Commodity Channel Index)
        cci = indicators.get("cci", 0)
        if cci < -200:
            score += 1.5
            reasons.append(f"CCI extremely oversold: {cci:.0f}")
        elif cci < -100:
            score += 0.5
            reasons.append(f"CCI oversold: {cci:.0f}")
        elif cci > 200:
            score -= 1.5
            reasons.append(f"CCI extremely overbought: {cci:.0f}")
        elif cci > 100:
            score -= 0.5
            reasons.append(f"CCI overbought: {cci:.0f}")

        # 7. Market regime adjustment
        # Mean reversion works best in sideways markets
        if market_regime == MarketRegime.SIDEWAYS:
            score *= 1.3  # Boost signals in sideways market
            reasons.append("Sideways market - good for mean reversion")
        elif market_regime in [MarketRegime.BULL, MarketRegime.BEAR]:
            score *= 0.7  # Reduce signals in trending markets
            reasons.append("Trending market - cautious on reversion")

        # 8. Volume confirmation (important for reversals)
        vol_ratio = indicators.get("volume_ratio", 1)
        if vol_ratio > 1.5:
            score *= 1.1  # Boost signal with volume
            reasons.append("Volume supports reversal")

        signal, confidence = self._score_to_signal(score)

        return {
            "symbol": symbol,
            "signal": signal.value,
            "confidence": confidence,
            "score": score,
            "price": price,
            "reasons": reasons,
            "strategy": self.name,
            "volatility_regime": volatility_regime,
        }

    def _score_to_signal(self, score: float) -> Tuple[Signal, float]:
        """Convert score to signal and confidence."""
        confidence = min(abs(score) / 6, 1.0)

        if score >= 4:
            return Signal.STRONG_BUY, confidence
        elif score >= 2:
            return Signal.BUY, confidence
        elif score <= -4:
            return Signal.STRONG_SELL, confidence
        elif score <= -2:
            return Signal.SELL, confidence
        else:
            return Signal.HOLD, 0.0


class BreakoutStrategy(AdvancedStrategy):
    """Advanced breakout strategy with volume and volatility confirmation."""

    def __init__(self):
        super().__init__("Advanced Breakout")

    def analyze(self, symbol: str, df: pd.DataFrame, market_regime: MarketRegime) -> Optional[Dict]:
        """Analyze for breakout signals."""
        if df.empty or len(df) < 50:
            return None

        indicators = get_latest_signal(df)
        volatility_regime = self.calculate_volatility_regime(df)

        # Calculate multiple timeframe ranges
        recent_20 = df.tail(20)
        recent_50 = df.tail(50)

        high_20 = recent_20["high"].max()
        low_20 = recent_20["low"].min()
        high_50 = recent_50["high"].max()
        low_50 = recent_50["low"].min()

        price = indicators.get("close", 0)
        score = 0
        reasons = []

        # 1. 20-day breakout (shorter term)
        if price > high_20 * 1.01:  # 1% above high
            score += 3
            reasons.append(f"Bullish breakout (20-day) above ${high_20:.2f}")
        elif price > high_20:
            score += 1.5
            reasons.append(f"Near 20-day high: ${high_20:.2f}")
        elif price < low_20 * 0.99:  # 1% below low
            score -= 3
            reasons.append(f"Bearish breakdown (20-day) below ${low_20:.2f}")
        elif price < low_20:
            score -= 1.5
            reasons.append(f"Near 20-day low: ${low_20:.2f}")

        # 2. 50-day breakout (longer term - more significant)
        if price > high_50 * 1.02:  # 2% above high
            score += 2
            reasons.append(f"Major breakout (50-day) above ${high_50:.2f}")
        elif price < low_50 * 0.98:  # 2% below low
            score -= 2
            reasons.append(f"Major breakdown (50-day) below ${low_50:.2f}")

        # 3. Volume confirmation (CRITICAL for breakouts)
        vol_ratio = indicators.get("volume_ratio", 1)
        vol_surge = self.detect_volume_surge(df)

        if vol_surge:
            if score > 0:
                score *= 1.4  # Significant boost with volume surge
                reasons.append("MASSIVE volume surge confirms breakout")
            else:
                score *= 1.2
                reasons.append("Volume surge confirms breakdown")
        elif vol_ratio > 1.8:
            if score > 0:
                score *= 1.2
                reasons.append("Very high volume confirms breakout")
            else:
                score *= 1.1
                reasons.append("High volume confirms breakdown")
        elif vol_ratio < 1.2:
            # Low volume = fake breakout warning
            score *= 0.6
            reasons.append("WARNING: Low volume - potential fake breakout")

        # 4. Volatility confirmation
        atr = indicators.get("atr", 0)
        atr_pct = (atr / price) if price > 0 else 0

        if atr_pct > 0.02:
            # High volatility supports breakout
            score *= 1.15
            reasons.append("High volatility supports breakout")
        elif atr_pct < 0.01:
            # Low volatility = weak breakout
            score *= 0.8
            reasons.append("Low volatility - weak breakout")

        # 5. Price consolidation before breakout
        # Check if price was consolidating (low ATR expansion)
        if len(df) > 20:
            atr_recent = df["atr"].iloc[-5:].mean() if "atr" in df else 0
            atr_previous = df["atr"].iloc[-20:-5].mean() if "atr" in df else 0

            if atr_recent > atr_previous * 1.5:
                score *= 1.1
                reasons.append("Volatility expansion confirms breakout")

        # 6. RSI confirmation
        rsi = indicators.get("rsi", 50)
        if score > 0:  # Bullish breakout
            if 50 < rsi < 75:
                score += 0.5
                reasons.append(f"RSI confirms breakout: {rsi:.0f}")
            elif rsi > 80:
                score -= 0.5
                reasons.append(f"RSI overbought - caution: {rsi:.0f}")
        else:  # Bearish breakdown
            if 25 < rsi < 50:
                score -= 0.5
                reasons.append(f"RSI confirms breakdown: {rsi:.0f}")

        # 7. Market regime adjustment
        if market_regime == MarketRegime.BULL and score > 0:
            score *= 1.2  # Bullish breakouts more reliable in bull market
            reasons.append("Bull market supports breakout")
        elif market_regime == MarketRegime.BEAR and score < 0:
            score *= 1.2  # Bearish breakdowns more reliable in bear market
            reasons.append("Bear market supports breakdown")

        # 8. Time of day pattern (if intraday data)
        # For daily data, we'll skip this for now

        signal, confidence = self._score_to_signal(score)

        return {
            "symbol": symbol,
            "signal": signal.value,
            "confidence": confidence,
            "score": score,
            "price": price,
            "reasons": reasons,
            "strategy": self.name,
            "volatility_regime": volatility_regime,
            "breakout_level": high_50 if score > 0 else low_50,
        }

    def _score_to_signal(self, score: float) -> Tuple[Signal, float]:
        """Convert score to signal and confidence."""
        confidence = min(abs(score) / 5, 1.0)

        if score >= 3.5:
            return Signal.STRONG_BUY, confidence
        elif score >= 1.5:
            return Signal.BUY, confidence
        elif score <= -3.5:
            return Signal.STRONG_SELL, confidence
        elif score <= -1.5:
            return Signal.SELL, confidence
        else:
            return Signal.HOLD, 0.0


class MultiStrategyEnsemble(AdvancedStrategy):
    """
    Ensemble strategy that combines multiple strategies with intelligent weighting.
    This is the MAXIMUM INTELLIGENCE approach.
    """

    def __init__(self):
        super().__init__("Multi-Strategy Ensemble")
        self.momentum = MomentumStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.breakout = BreakoutStrategy()

        # Strategy weights based on market regime
        self.regime_weights = {
            MarketRegime.BULL: {"momentum": 0.5, "breakout": 0.35, "mean_reversion": 0.15},
            MarketRegime.BEAR: {"momentum": 0.35, "breakout": 0.40, "mean_reversion": 0.25},
            MarketRegime.SIDEWAYS: {"momentum": 0.20, "breakout": 0.20, "mean_reversion": 0.60},
            MarketRegime.VOLATILE: {"momentum": 0.30, "breakout": 0.50, "mean_reversion": 0.20},
        }

    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime."""
        if len(df) < 50:
            return MarketRegime.SIDEWAYS

        # Use SPY or market data if available, otherwise use symbol's data
        trend = detect_trend(df)

        # Calculate volatility
        returns = df["close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Calculate trend strength
        sma_20 = df["close"].rolling(20).mean().iloc[-1]
        sma_50 = df["close"].rolling(50).mean().iloc[-1]
        sma_200 = df["close"].rolling(200).mean().iloc[-1]

        # Determine regime
        if volatility > 0.35:  # Very high volatility
            return MarketRegime.VOLATILE
        elif trend == "uptrend" and sma_20 > sma_50 > sma_200:
            return MarketRegime.BULL
        elif trend == "downtrend" and sma_20 < sma_50 < sma_200:
            return MarketRegime.BEAR
        else:
            return MarketRegime.SIDEWAYS

    def analyze(self, symbol: str, df: pd.DataFrame, market_regime: MarketRegime = None) -> Optional[Dict]:
        """
        Generate ensemble signal from all strategies.
        This is the MAXIMUM INTELLIGENCE approach.
        """
        if df.empty or len(df) < 50:
            return None

        # Detect market regime if not provided
        if market_regime is None:
            market_regime = self.detect_market_regime(df)

        # Get signals from all strategies
        momentum_signal = self.momentum.analyze(symbol, df, market_regime)
        mean_rev_signal = self.mean_reversion.analyze(symbol, df, market_regime)
        breakout_signal = self.breakout.analyze(symbol, df, market_regime)

        if not all([momentum_signal, mean_rev_signal, breakout_signal]):
            return None

        # Get weights for current regime
        weights = self.regime_weights.get(market_regime, self.regime_weights[MarketRegime.SIDEWAYS])

        # Convert signals to numeric scores
        signal_values = {
            Signal.STRONG_BUY: 2,
            Signal.BUY: 1,
            Signal.HOLD: 0,
            Signal.SELL: -1,
            Signal.STRONG_SELL: -2,
        }

        mom_score = signal_values.get(Signal(momentum_signal["signal"]), 0) * momentum_signal["confidence"]
        mr_score = signal_values.get(Signal(mean_rev_signal["signal"]), 0) * mean_rev_signal["confidence"]
        bo_score = signal_values.get(Signal(breakout_signal["signal"]), 0) * breakout_signal["confidence"]

        # Calculate weighted ensemble score
        ensemble_score = (
            mom_score * weights["momentum"] +
            mr_score * weights["mean_reversion"] +
            bo_score * weights["breakout"]
        )

        # Boost score if multiple strategies agree
        agreement_count = sum([
            mom_score > 0.5,
            mr_score > 0.5,
            bo_score > 0.5
        ])

        if agreement_count >= 2 and ensemble_score > 0:
            ensemble_score *= 1.3  # Boost for bullish agreement
        elif agreement_count >= 2 and ensemble_score < 0:
            ensemble_score *= 1.3  # Boost for bearish agreement

        # Reduce score if strategies disagree (conflicting signals)
        if mom_score > 0.5 and mr_score < -0.5:
            ensemble_score *= 0.7  # Conflict warning
        elif mom_score < -0.5 and mr_score > 0.5:
            ensemble_score *= 0.7  # Conflict warning

        # Gather all reasons
        all_reasons = []
        all_reasons.extend([f"[Momentum] {r}" for r in momentum_signal["reasons"]])
        all_reasons.extend([f"[MeanRev] {r}" for r in mean_rev_signal["reasons"]])
        all_reasons.extend([f"[Breakout] {r}" for r in breakout_signal["reasons"]])

        # Convert ensemble score to signal
        signal, confidence = self._score_to_signal(ensemble_score)

        return {
            "symbol": symbol,
            "signal": signal.value,
            "confidence": confidence,
            "score": ensemble_score,
            "price": momentum_signal["price"],
            "reasons": all_reasons[:10],  # Top 10 reasons
            "strategy": self.name,
            "market_regime": market_regime.value,
            "component_signals": {
                "momentum": {
                    "signal": momentum_signal["signal"],
                    "score": mom_score,
                    "confidence": momentum_signal["confidence"]
                },
                "mean_reversion": {
                    "signal": mean_rev_signal["signal"],
                    "score": mr_score,
                    "confidence": mean_rev_signal["confidence"]
                },
                "breakout": {
                    "signal": breakout_signal["signal"],
                    "score": bo_score,
                    "confidence": breakout_signal["confidence"]
                }
            },
            "strategy_weights": weights,
        }

    def _score_to_signal(self, score: float) -> Tuple[Signal, float]:
        """Convert ensemble score to signal and confidence."""
        confidence = min(abs(score), 1.0)

        if score >= 1.0:
            return Signal.STRONG_BUY, confidence
        elif score >= 0.5:
            return Signal.BUY, confidence
        elif score <= -1.0:
            return Signal.STRONG_SELL, confidence
        elif score <= -0.5:
            return Signal.SELL, confidence
        else:
            return Signal.HOLD, 0.0


def get_strategy(name: str = "ensemble") -> AdvancedStrategy:
    """Factory function to get strategy by name."""
    strategies = {
        "momentum": MomentumStrategy,
        "mean_reversion": MeanReversionStrategy,
        "breakout": BreakoutStrategy,
        "ensemble": MultiStrategyEnsemble,  # DEFAULT - MAX INTELLIGENCE
    }

    strategy_class = strategies.get(name.lower(), MultiStrategyEnsemble)
    return strategy_class()
