"""
Enhanced Indicators Package for crypto-alpha-arena

This package provides advanced market analysis indicators:
- Fear & Greed Index (sentiment analysis)
- Whale Alert monitoring (large transaction tracking)
- Pivot Point calculations (support/resistance levels)
- Candlestick pattern detection
- Advanced volume analysis (OBV, VWAP, divergence)
- Unified enhanced market context

Usage:
    from src.backend.indicators import EnhancedMarketContext
    
    ctx = EnhancedMarketContext()
    analysis = ctx.build_context(
        asset="BTC",
        current_price=100000,
        ohlcv_data=candles,
        daily_ohlc={"high": 101000, "low": 99000, "close": 100500}
    )
    
    # Use in LLM prompt
    prompt_text = analysis["prompt_text"]
    composite_signal = analysis["composite_signal"]
"""

# Core TAAPI client (existing)
try:
    from .taapi_client import TAAPIClient
    from .taapi_cache import TAAPICache, get_cache
except ImportError:
    TAAPIClient = None
    TAAPICache = None
    get_cache = None

# Enhanced indicators (new)
from .sentiment_client import SentimentClient, get_fear_greed
from .whale_alert_client import WhaleAlertClient, get_whale_analysis
from .pivot_calculator import PivotCalculator, PivotLevels, calculate_pivots
from .pattern_detector import PatternDetector, CandlePattern, detect_patterns
from .volume_analyzer import VolumeAnalyzer, VolumeAnalysis, analyze_volume
from .enhanced_context import EnhancedMarketContext, build_enhanced_context

__all__ = [
    # Existing
    "TAAPIClient",
    "TAAPICache",
    "get_cache",
    
    # Sentiment
    "SentimentClient",
    "get_fear_greed",
    
    # Whale Alert
    "WhaleAlertClient",
    "get_whale_analysis",
    
    # Pivot Points
    "PivotCalculator",
    "PivotLevels",
    "calculate_pivots",
    
    # Pattern Detection
    "PatternDetector",
    "CandlePattern",
    "detect_patterns",
    
    # Volume Analysis
    "VolumeAnalyzer",
    "VolumeAnalysis",
    "analyze_volume",
    
    # Main Integration
    "EnhancedMarketContext",
    "build_enhanced_context",
]
