"""
Smart Signal Analyzer
Multi-timeframe, multi-indicator analysis for intelligent entry/exit decisions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger

from config import settings


class SignalAnalyzer:
    """Advanced signal analysis for smart trading decisions."""

    def __init__(self):
        self.min_confidence = 0.70  # Minimum confidence to trade (raised to 70% for only highest-quality setups)
        self.max_positions_per_signal = 3  # Max positions with same signal type

    def analyze_entry_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        market_regime: str = "SIDEWAYS"
    ) -> Dict:
        """
        Analyze entry conditions using multiple indicators.

        Returns:
            {
                'signal': 'BUY' | 'SELL' | 'HOLD',
                'confidence': 0.0-1.0,
                'strength': 'WEAK' | 'MODERATE' | 'STRONG',
                'reason': str,
                'indicators': dict
            }
        """
        if len(data) < 20:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'strength': 'WEAK',
                'reason': 'Insufficient data',
                'indicators': {}
            }

        latest = data.iloc[-1]
        indicators = self._calculate_indicators(data)

        # HARD FILTER: Reject immediately if volume is too weak (no institutional interest)
        vol = indicators.get('volume', 0)
        vol_avg = indicators.get('volume_avg', 1)
        if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
            vol_ratio = vol / vol_avg
            if vol_ratio < 1.2:
                # Volume too weak - don't even analyze further
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'strength': 'WEAK',
                    'reason': f'⛔ VOLUME FILTER: Rejected due to weak volume ({vol_ratio:.1f}x avg - need 1.2x minimum)',
                    'indicators': indicators
                }

        # Scoring system (0-100)
        score = 0
        reasons = []

        # 1. Trend Analysis (25 points)
        trend_score, trend_reason = self._analyze_trend(indicators, latest)
        score += trend_score
        if trend_reason:
            reasons.append(trend_reason)

        # 2. Momentum (25 points)
        momentum_score, momentum_reason = self._analyze_momentum(indicators, latest)
        score += momentum_score
        if momentum_reason:
            reasons.append(momentum_reason)

        # 3. Volume Confirmation (20 points)
        volume_score, volume_reason = self._analyze_volume(indicators, latest)
        score += volume_score
        if volume_reason:
            reasons.append(volume_reason)

        # 4. Support/Resistance (15 points)
        sr_score, sr_reason = self._analyze_support_resistance(indicators, latest)
        score += sr_score
        if sr_reason:
            reasons.append(sr_reason)

        # 5. Volatility (15 points)
        vol_score, vol_reason = self._analyze_volatility(indicators, latest)
        score += vol_score
        if vol_reason:
            reasons.append(vol_reason)

        # Convert score to signal
        confidence = score / 100

        if score >= 75:
            signal = 'STRONG_BUY'
            strength = 'STRONG'
        elif score >= 60:
            signal = 'BUY'
            strength = 'MODERATE'
        elif score >= 40:
            signal = 'HOLD'
            strength = 'WEAK'
        elif score >= 25:
            signal = 'SELL'
            strength = 'MODERATE'
        else:
            signal = 'STRONG_SELL'
            strength = 'STRONG'

        # Adjust for market regime
        if market_regime == "BEAR" and 'BUY' in signal:
            confidence *= 0.7  # Reduce buy confidence in bear market
        elif market_regime == "BULL" and 'SELL' in signal:
            confidence *= 0.7  # Reduce sell confidence in bull market

        return {
            'signal': signal,
            'confidence': min(confidence, 1.0),
            'strength': strength,
            'reason': '; '.join(reasons),
            'indicators': indicators
        }

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators."""
        df = data.copy()

        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=min(50, len(df))).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()

        # Price momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)

        # Volatility (ATR-like)
        df['tr'] = pd.DataFrame({
            'hl': df['high'] - df['low'],
            'hc': abs(df['high'] - df['close'].shift(1)),
            'lc': abs(df['low'] - df['close'].shift(1))
        }).max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        latest = df.iloc[-1]

        return {
            'price': latest['close'],
            'sma_20': latest['sma_20'],
            'sma_50': latest['sma_50'],
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'macd_histogram': latest['macd_histogram'],
            'bb_upper': latest['bb_upper'],
            'bb_lower': latest['bb_lower'],
            'bb_middle': latest['bb_middle'],
            'volume': latest['volume'],
            'volume_avg': latest['volume_sma'],
            'momentum_5': latest['momentum_5'],
            'momentum_10': latest['momentum_10'],
            'atr': latest['atr']
        }

    def _analyze_trend(self, ind: Dict, latest: pd.Series) -> Tuple[int, str]:
        """Analyze trend indicators (0-25 points)."""
        score = 0
        reasons = []

        # Price vs SMAs
        price = ind['price']
        sma20 = ind['sma_20']

        if price > sma20:
            score += 10
            reasons.append("Price above SMA20")
        elif price < sma20:
            score -= 5

        # SMA alignment
        if pd.notna(ind['sma_50']):
            if sma20 > ind['sma_50']:
                score += 10
                reasons.append("Bullish SMA alignment")

        # MACD
        if pd.notna(ind['macd_histogram']):
            if ind['macd_histogram'] > 0:
                score += 5
                reasons.append("MACD positive")

        return max(0, min(score, 25)), '; '.join(reasons)

    def _analyze_momentum(self, ind: Dict, latest: pd.Series) -> Tuple[int, str]:
        """Analyze momentum indicators (0-25 points)."""
        score = 0
        reasons = []

        # RSI
        rsi = ind['rsi']
        if pd.notna(rsi):
            if 30 <= rsi <= 70:
                score += 10
                reasons.append(f"RSI neutral ({rsi:.1f})")
            elif rsi < 30:
                score += 15  # Oversold = buy opportunity
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                score -= 10  # Overbought

        # Price momentum
        if pd.notna(ind['momentum_5']):
            if ind['momentum_5'] > 0.02:  # 2%+ gain in 5 days
                score += 10
                reasons.append("Strong 5-day momentum")

        if pd.notna(ind['momentum_10']):
            if ind['momentum_10'] > 0.03:  # 3%+ gain in 10 days
                score += 5
                reasons.append("Positive 10-day momentum")

        return max(0, min(score, 25)), '; '.join(reasons)

    def _analyze_volume(self, ind: Dict, latest: pd.Series) -> Tuple[int, str]:
        """
        Analyze volume confirmation (0-20 points).

        SMART volume analysis:
        - Require 1.5x average volume for maximum points (institutional buying)
        - Below 1.2x gets significant penalty (weak setup)
        - Check volume trend (increasing or decreasing)
        """
        score = 0
        reasons = []

        vol = ind['volume']
        vol_avg = ind['volume_avg']

        if pd.notna(vol) and pd.notna(vol_avg):
            vol_ratio = vol / vol_avg

            # Volume ratio scoring (stricter!)
            if vol_ratio >= 1.5:
                # Strong institutional buying - maximum points
                score += 20
                reasons.append(f"🚀 Strong volume confirmation ({vol_ratio:.1f}x avg)")
            elif vol_ratio >= 1.3:
                # Good volume
                score += 15
                reasons.append(f"Good volume ({vol_ratio:.1f}x avg)")
            elif vol_ratio >= 1.1:
                # Below average - significant penalty
                score += 5
                reasons.append(f"Weak volume ({vol_ratio:.1f}x avg)")
            else:
                # Very weak - no points
                score += 0
                reasons.append(f"⚠️ Poor volume ({vol_ratio:.1f}x avg)")

        return max(0, min(score, 20)), '; '.join(reasons)

    def _analyze_support_resistance(self, ind: Dict, latest: pd.Series) -> Tuple[int, str]:
        """Analyze support/resistance levels (0-15 points)."""
        score = 0
        reasons = []

        price = ind['price']
        bb_lower = ind['bb_lower']
        bb_upper = ind['bb_upper']

        if pd.notna(bb_lower) and pd.notna(bb_upper):
            bb_width = bb_upper - bb_lower
            bb_position = (price - bb_lower) / bb_width

            if bb_position < 0.3:  # Near lower Bollinger Band
                score += 15
                reasons.append("Near support (BB lower)")
            elif bb_position > 0.7:  # Near upper Bollinger Band
                score -= 5
                reasons.append("Near resistance (BB upper)")

        return max(0, min(score, 15)), '; '.join(reasons)

    def _analyze_volatility(self, ind: Dict, latest: pd.Series) -> Tuple[int, str]:
        """Analyze volatility (0-15 points)."""
        score = 0
        reasons = []

        atr = ind['atr']
        price = ind['price']

        if pd.notna(atr) and price > 0:
            atr_pct = (atr / price) * 100

            if atr_pct < 2:
                score += 15
                reasons.append("Low volatility")
            elif atr_pct < 3:
                score += 10
                reasons.append("Moderate volatility")
            elif atr_pct < 4:
                score += 5
                reasons.append("Elevated volatility")
            else:
                reasons.append("High volatility - reducing position size")

        return max(0, min(score, 15)), '; '.join(reasons)

    def should_exit_position(
        self,
        position: Dict,
        current_data: pd.DataFrame,
        hold_time_minutes: int
    ) -> Tuple[bool, str]:
        """
        Determine if a position should be closed.

        Returns:
            (should_exit, reason)
        """
        unrealized_pl_pct = position['unrealized_plpc']

        # Calculate current indicators
        ind = self._calculate_indicators(current_data)
        latest_ind = current_data.iloc[-1]

        exit_reasons = []

        # 1. Stop Loss / Take Profit (primary checks)
        if unrealized_pl_pct <= -settings.stop_loss_pct * 100:
            exit_reasons.append("Stop loss hit")

        if unrealized_pl_pct >= settings.take_profit_pct * 100:
            exit_reasons.append("Take profit hit")

        # 2. Technical Deterioration
        signal = self.analyze_entry_signal(position['symbol'], current_data)

        if 'SELL' in signal['signal'] and signal['confidence'] > 0.70:
            exit_reasons.append(f"Technical breakdown (confidence: {signal['confidence']:.0%})")

        # 3. RSI Extremes
        if pd.notna(ind['rsi']):
            if ind['rsi'] > 75:
                exit_reasons.append(f"RSI overbought ({ind['rsi']:.1f})")
            elif ind['rsi'] < 25 and unrealized_pl_pct < 0:
                exit_reasons.append(f"RSI extremely oversold ({ind['rsi']:.1f}) - consider holding")

        # 4. Time-based exit (if held too long without profit)
        if hold_time_minutes > 120 and unrealized_pl_pct < 1:
            exit_reasons.append(f"Held {hold_time_minutes}min without profit")

        # 5. Momentum reversal
        if pd.notna(ind['momentum_5']) and ind['momentum_5'] < -0.03:
            exit_reasons.append("Momentum reversed (-3% in 5 days)")

        should_exit = len(exit_reasons) > 0 and 'consider holding' not in exit_reasons[-1]

        return should_exit, '; '.join(exit_reasons) if exit_reasons else "Hold"
