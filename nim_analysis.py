"""
NVIDIA NIM LLM integration for market analysis.
Optional - uses LLM to provide market context and sentiment analysis.
"""
import requests
import json
from typing import Dict, Optional
from loguru import logger

from config import settings


class NIMAnalyzer:
    """NVIDIA NIM LLM analyzer for trading insights."""

    def __init__(self):
        """Initialize NIM analyzer."""
        self.api_key = settings.nim_api_key
        self.base_url = settings.nim_base_url
        self.enabled = bool(self.api_key and self.api_key != "")

        if self.enabled:
            logger.info("NVIDIA NIM LLM analyzer enabled")
        else:
            logger.info("NVIDIA NIM LLM analyzer disabled (no API key)")

    def analyze_market_sentiment(self, symbol: str, indicators: Dict) -> Optional[Dict]:
        """Use LLM to analyze market sentiment based on indicators."""
        if not self.enabled:
            return None

        try:
            prompt = self._build_sentiment_prompt(symbol, indicators)

            response = self._call_nim(prompt)

            if response:
                return self._parse_sentiment_response(response, symbol)

        except Exception as e:
            logger.warning(f"NIM sentiment analysis failed: {e}")

        return None

    def analyze_trade_rationale(
        self,
        symbol: str,
        signal: str,
        indicators: Dict,
        reasons: list
    ) -> Optional[str]:
        """Use LLM to explain the trading rationale."""
        if not self.enabled:
            return None

        try:
            prompt = self._build_rationale_prompt(symbol, signal, indicators, reasons)

            response = self._call_nim(prompt)

            if response:
                return response

        except Exception as e:
            logger.warning(f"NIM rationale analysis failed: {e}")

        return None

    def _build_sentiment_prompt(self, symbol: str, indicators: Dict) -> str:
        """Build prompt for sentiment analysis."""
        prompt = f"""Analyze the technical indicators for {symbol} and provide a brief sentiment assessment.

Current Price: ${indicators.get('close', 0):.2f}
RSI: {indicators.get('rsi', 50):.1f}
MACD Histogram: {indicators.get('macd_hist', 0):.2f}
Price vs SMA20: {((indicators.get('close', 0) / indicators.get('sma_20', 1)) - 1) * 100:+.1f}%
Bollinger Band Position: {indicators.get('bb_percent', 0.5) * 100:.1f}%
Rate of Change (10-day): {indicators.get('roc', 0):.1f}%

Provide a JSON response with:
{{
  "sentiment": "bullish|bearish|neutral",
  "strength": "strong|moderate|weak",
  "key_factors": ["factor1", "factor2", "factor3"],
  "overall_assessment": "Brief 1-sentence assessment"
}}

Respond ONLY with valid JSON, no other text."""

        return prompt

    def _build_rationale_prompt(
        self,
        symbol: str,
        signal: str,
        indicators: Dict,
        reasons: list
    ) -> str:
        """Build prompt for trade rationale."""
        reasons_str = ", ".join(reasons)

        prompt = f"""Explain the trading rationale for {symbol}.

Signal: {signal.upper()}
Reasons: {reasons_str}

Key Indicators:
- Price: ${indicators.get('close', 0):.2f}
- RSI: {indicators.get('rsi', 50):.1f}
- MACD: {indicators.get('macd', 0):.2f}
- ROC: {indicators.get('roc', 0):.1f}%

Provide a concise 2-3 sentence explanation of why this signal makes sense based on the technical indicators."""

        return prompt

    def _call_nim(self, prompt: str) -> Optional[str]:
        """Call NVIDIA NIM API."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "meta/llama-3.1-405b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.5,
                "max_tokens": 500
            }

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                logger.warning(f"NIM API error: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"NIM API call failed: {e}")
            return None

    def _parse_sentiment_response(self, response: str, symbol: str) -> Dict:
        """Parse sentiment response from LLM."""
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1

            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)

                return {
                    "symbol": symbol,
                    "sentiment": data.get("sentiment", "neutral"),
                    "strength": data.get("strength", "moderate"),
                    "key_factors": data.get("key_factors", []),
                    "assessment": data.get("overall_assessment", ""),
                }

        except Exception as e:
            logger.warning(f"Failed to parse NIM sentiment response: {e}")

        return None


# Global instance
nim_analyzer = NIMAnalyzer()
