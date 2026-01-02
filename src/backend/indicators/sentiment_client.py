"""
Sentiment Client - Fear & Greed Index from alternative.me
FREE API, no key required
"""

import requests
import time
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SentimentClient:
    """
    Client for fetching Fear & Greed Index from alternative.me API.
    
    The Fear & Greed Index ranges from 0-100:
    - 0-25: Extreme Fear (potential BUY signal - contrarian)
    - 25-45: Fear
    - 45-55: Neutral
    - 55-75: Greed
    - 75-100: Extreme Greed (potential SELL signal - contrarian)
    """
    
    BASE_URL = "https://api.alternative.me/fng/"
    
    def __init__(self, cache_ttl: int = 3600):
        """
        Initialize sentiment client.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_time: Optional[float] = None
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        if self._cache_time is None:
            return False
        return (time.time() - self._cache_time) < self.cache_ttl
    
    def get_fear_greed_index(self, limit: int = 1) -> Optional[Dict[str, Any]]:
        """
        Fetch Fear & Greed Index data.
        
        Args:
            limit: Number of days of data to fetch (1 = today only)
            
        Returns:
            Dict with 'value', 'classification', 'timestamp', 'trend'
            or None on error
        """
        # Check cache first
        if limit == 1 and self._is_cache_valid():
            return self._cache
        
        try:
            params = {"limit": limit, "format": "json"}
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("metadata", {}).get("error"):
                logger.error(f"API error: {data['metadata']['error']}")
                return None
            
            fng_data = data.get("data", [])
            if not fng_data:
                logger.warning("No Fear & Greed data returned")
                return None
            
            # Get latest value
            latest = fng_data[0]
            value = int(latest.get("value", 50))
            classification = latest.get("value_classification", "Neutral")
            timestamp = int(latest.get("timestamp", 0))
            
            # Calculate trend if we have multiple days
            trend = "stable"
            if len(fng_data) >= 2:
                prev_value = int(fng_data[1].get("value", value))
                if value > prev_value + 5:
                    trend = "rising"
                elif value < prev_value - 5:
                    trend = "falling"
            
            result = {
                "value": value,
                "classification": classification,
                "timestamp": datetime.fromtimestamp(timestamp).isoformat() if timestamp else None,
                "trend": trend,
                "signal": self._get_trading_signal(value),
                "historical": [
                    {"value": int(d.get("value", 0)), "date": d.get("timestamp")}
                    for d in fng_data[:7]  # Last 7 days
                ] if limit > 1 else None
            }
            
            # Cache single-day requests
            if limit == 1:
                self._cache = result
                self._cache_time = time.time()
            
            logger.info(f"Fear & Greed Index: {value} ({classification}) - Signal: {result['signal']}")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch Fear & Greed Index: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing Fear & Greed data: {e}")
            return None
    
    def _get_trading_signal(self, value: int) -> str:
        """
        Get trading signal based on Fear & Greed value.
        
        Uses contrarian approach:
        - Extreme Fear = potential buying opportunity
        - Extreme Greed = potential selling opportunity
        
        Args:
            value: Fear & Greed index value (0-100)
            
        Returns:
            Signal string: 'STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'
        """
        if value <= 15:
            return "STRONG_BUY"  # Extreme fear - strong contrarian buy
        elif value <= 25:
            return "BUY"  # Fear - contrarian buy
        elif value <= 45:
            return "LEAN_BUY"  # Moderate fear
        elif value <= 55:
            return "NEUTRAL"
        elif value <= 75:
            return "LEAN_SELL"  # Moderate greed
        elif value <= 85:
            return "SELL"  # Greed - contrarian sell
        else:
            return "STRONG_SELL"  # Extreme greed - strong contrarian sell
    
    def get_sentiment_context(self) -> str:
        """
        Get formatted sentiment context for LLM prompt.
        
        Returns:
            Formatted string describing current sentiment
        """
        data = self.get_fear_greed_index(limit=7)  # Get 7 days for trend
        
        if not data:
            return "Sentiment: Unable to fetch Fear & Greed Index"
        
        value = data["value"]
        classification = data["classification"]
        trend = data["trend"]
        signal = data["signal"]
        
        # Build context string
        context = f"""
## Market Sentiment (Fear & Greed Index)
- Current Value: {value}/100 ({classification})
- Trend: {trend}
- Contrarian Signal: {signal}

Interpretation:
- 0-25: Extreme Fear (historically good buying opportunity)
- 75-100: Extreme Greed (historically good selling opportunity)
- Current market is showing {classification.lower()}, suggesting {self._get_interpretation(value)}
"""
        return context.strip()
    
    def _get_interpretation(self, value: int) -> str:
        """Get human-readable interpretation of the index value."""
        if value <= 25:
            return "potential accumulation opportunity as fear is elevated"
        elif value <= 45:
            return "cautious optimism may be warranted"
        elif value <= 55:
            return "market is balanced, wait for clearer signals"
        elif value <= 75:
            return "some caution advised as greed is building"
        else:
            return "high risk of correction, consider taking profits"


# Singleton instance - shared across all modules for cache efficiency
_shared_client: Optional['SentimentClient'] = None


def get_shared_client(cache_ttl: int = 900) -> 'SentimentClient':
    """Get or create the shared SentimentClient singleton."""
    global _shared_client
    if _shared_client is None:
        _shared_client = SentimentClient(cache_ttl=cache_ttl)
    return _shared_client


# Convenience function for quick access
def get_fear_greed() -> Optional[Dict[str, Any]]:
    """Quick access to Fear & Greed Index."""
    return get_shared_client().get_fear_greed_index()
