"""
Pattern Detector - Candlestick Pattern Recognition
Pure Python implementation (no TA-Lib dependency)
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CandlePattern:
    """Detected candlestick pattern."""
    name: str
    pattern_type: str  # 'bullish', 'bearish', 'neutral'
    strength: str  # 'strong', 'moderate', 'weak'
    confidence: float  # 0-100
    candle_count: int  # 1, 2, or 3 candle pattern
    description: str


@dataclass
class Candle:
    """Single candlestick data."""
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)
    
    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low
    
    @property
    def total_range(self) -> float:
        return self.high - self.low
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    @property
    def body_pct(self) -> float:
        """Body size as percentage of total range."""
        if self.total_range == 0:
            return 0
        return (self.body_size / self.total_range) * 100


class PatternDetector:
    """
    Detect candlestick patterns for trading signals.
    
    Patterns detected:
    - Single candle: Doji, Hammer, Shooting Star, Marubozu
    - Two candle: Engulfing, Piercing, Dark Cloud, Tweezer
    - Three candle: Morning/Evening Star, Three Soldiers/Crows
    """
    
    def __init__(self, body_threshold: float = 0.1, wick_ratio: float = 2.0):
        """
        Initialize pattern detector.
        
        Args:
            body_threshold: Minimum body size as fraction of range (for doji detection)
            wick_ratio: Minimum wick-to-body ratio for hammer/shooting star
        """
        self.body_threshold = body_threshold
        self.wick_ratio = wick_ratio
    
    def _create_candles(self, ohlcv_data: List[Dict]) -> List[Candle]:
        """Convert OHLCV data to Candle objects."""
        candles = []
        for d in ohlcv_data:
            try:
                candles.append(Candle(
                    open=float(d.get('open', d.get('o', 0))),
                    high=float(d.get('high', d.get('h', 0))),
                    low=float(d.get('low', d.get('l', 0))),
                    close=float(d.get('close', d.get('c', 0))),
                    volume=float(d.get('volume', d.get('v', 0)))
                ))
            except (ValueError, TypeError):
                continue
        return candles
    
    def detect_patterns(self, ohlcv_data: List[Dict], 
                        lookback: int = 5) -> List[CandlePattern]:
        """
        Detect candlestick patterns in OHLCV data.
        
        Args:
            ohlcv_data: List of OHLCV dicts [{open, high, low, close, volume}, ...]
            lookback: Number of recent candles to analyze
            
        Returns:
            List of detected CandlePattern objects
        """
        if len(ohlcv_data) < 3:
            logger.warning("Insufficient data for pattern detection")
            return []
        
        candles = self._create_candles(ohlcv_data)
        if len(candles) < 3:
            return []
        
        # Analyze recent candles only
        recent = candles[-lookback:] if len(candles) > lookback else candles
        patterns = []
        
        # Single candle patterns (check last candle)
        if len(recent) >= 1:
            single_patterns = self._detect_single_patterns(recent[-1], recent)
            patterns.extend(single_patterns)
        
        # Two candle patterns
        if len(recent) >= 2:
            two_patterns = self._detect_two_candle_patterns(recent[-2], recent[-1], recent)
            patterns.extend(two_patterns)
        
        # Three candle patterns
        if len(recent) >= 3:
            three_patterns = self._detect_three_candle_patterns(
                recent[-3], recent[-2], recent[-1], recent
            )
            patterns.extend(three_patterns)
        
        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return patterns
    
    def _detect_single_patterns(self, candle: Candle, 
                                context: List[Candle]) -> List[CandlePattern]:
        """Detect single-candle patterns."""
        patterns = []
        
        # Calculate average range for context
        avg_range = sum(c.total_range for c in context) / len(context) if context else candle.total_range
        
        # Doji (very small body)
        if candle.body_pct < 10:
            patterns.append(CandlePattern(
                name="Doji",
                pattern_type="neutral",
                strength="moderate",
                confidence=70 if candle.body_pct < 5 else 50,
                candle_count=1,
                description="Indecision candle - potential reversal signal"
            ))
        
        # Hammer (bullish reversal)
        if (candle.lower_wick >= self.wick_ratio * candle.body_size and
            candle.upper_wick <= candle.body_size * 0.5 and
            candle.body_size > 0):
            patterns.append(CandlePattern(
                name="Hammer",
                pattern_type="bullish",
                strength="strong" if candle.is_bullish else "moderate",
                confidence=75 if candle.is_bullish else 60,
                candle_count=1,
                description="Bullish reversal pattern - buyers stepping in"
            ))
        
        # Shooting Star (bearish reversal)
        if (candle.upper_wick >= self.wick_ratio * candle.body_size and
            candle.lower_wick <= candle.body_size * 0.5 and
            candle.body_size > 0):
            patterns.append(CandlePattern(
                name="Shooting Star",
                pattern_type="bearish",
                strength="strong" if candle.is_bearish else "moderate",
                confidence=75 if candle.is_bearish else 60,
                candle_count=1,
                description="Bearish reversal pattern - sellers stepping in"
            ))
        
        # Marubozu (strong momentum)
        if candle.body_pct > 90:
            if candle.is_bullish:
                patterns.append(CandlePattern(
                    name="Bullish Marubozu",
                    pattern_type="bullish",
                    strength="strong",
                    confidence=80,
                    candle_count=1,
                    description="Strong bullish momentum - no wicks"
                ))
            else:
                patterns.append(CandlePattern(
                    name="Bearish Marubozu",
                    pattern_type="bearish",
                    strength="strong",
                    confidence=80,
                    candle_count=1,
                    description="Strong bearish momentum - no wicks"
                ))
        
        return patterns
    
    def _detect_two_candle_patterns(self, prev: Candle, curr: Candle,
                                    context: List[Candle]) -> List[CandlePattern]:
        """Detect two-candle patterns."""
        patterns = []
        
        # Bullish Engulfing
        if (prev.is_bearish and curr.is_bullish and
            curr.open < prev.close and curr.close > prev.open):
            patterns.append(CandlePattern(
                name="Bullish Engulfing",
                pattern_type="bullish",
                strength="strong",
                confidence=80,
                candle_count=2,
                description="Strong bullish reversal - buyers overwhelm sellers"
            ))
        
        # Bearish Engulfing
        if (prev.is_bullish and curr.is_bearish and
            curr.open > prev.close and curr.close < prev.open):
            patterns.append(CandlePattern(
                name="Bearish Engulfing",
                pattern_type="bearish",
                strength="strong",
                confidence=80,
                candle_count=2,
                description="Strong bearish reversal - sellers overwhelm buyers"
            ))
        
        # Piercing Line (bullish)
        if (prev.is_bearish and curr.is_bullish and
            curr.open < prev.low and
            curr.close > (prev.open + prev.close) / 2 and
            curr.close < prev.open):
            patterns.append(CandlePattern(
                name="Piercing Line",
                pattern_type="bullish",
                strength="moderate",
                confidence=65,
                candle_count=2,
                description="Bullish reversal - gap down followed by recovery"
            ))
        
        # Dark Cloud Cover (bearish)
        if (prev.is_bullish and curr.is_bearish and
            curr.open > prev.high and
            curr.close < (prev.open + prev.close) / 2 and
            curr.close > prev.open):
            patterns.append(CandlePattern(
                name="Dark Cloud Cover",
                pattern_type="bearish",
                strength="moderate",
                confidence=65,
                candle_count=2,
                description="Bearish reversal - gap up followed by selloff"
            ))
        
        # Tweezer Top (bearish)
        if (abs(prev.high - curr.high) / prev.total_range < 0.05 and
            prev.is_bullish and curr.is_bearish):
            patterns.append(CandlePattern(
                name="Tweezer Top",
                pattern_type="bearish",
                strength="moderate",
                confidence=60,
                candle_count=2,
                description="Bearish reversal - equal highs show resistance"
            ))
        
        # Tweezer Bottom (bullish)
        if (abs(prev.low - curr.low) / prev.total_range < 0.05 and
            prev.is_bearish and curr.is_bullish):
            patterns.append(CandlePattern(
                name="Tweezer Bottom",
                pattern_type="bullish",
                strength="moderate",
                confidence=60,
                candle_count=2,
                description="Bullish reversal - equal lows show support"
            ))
        
        return patterns
    
    def _detect_three_candle_patterns(self, first: Candle, second: Candle,
                                      third: Candle,
                                      context: List[Candle]) -> List[CandlePattern]:
        """Detect three-candle patterns."""
        patterns = []
        
        # Morning Star (bullish reversal)
        if (first.is_bearish and
            second.body_pct < 30 and  # Small body (star)
            third.is_bullish and
            third.close > (first.open + first.close) / 2):
            patterns.append(CandlePattern(
                name="Morning Star",
                pattern_type="bullish",
                strength="strong",
                confidence=85,
                candle_count=3,
                description="Strong bullish reversal - indecision followed by buying"
            ))
        
        # Evening Star (bearish reversal)
        if (first.is_bullish and
            second.body_pct < 30 and  # Small body (star)
            third.is_bearish and
            third.close < (first.open + first.close) / 2):
            patterns.append(CandlePattern(
                name="Evening Star",
                pattern_type="bearish",
                strength="strong",
                confidence=85,
                candle_count=3,
                description="Strong bearish reversal - indecision followed by selling"
            ))
        
        # Three White Soldiers (bullish continuation)
        if (first.is_bullish and second.is_bullish and third.is_bullish and
            second.close > first.close and third.close > second.close and
            first.body_pct > 50 and second.body_pct > 50 and third.body_pct > 50):
            patterns.append(CandlePattern(
                name="Three White Soldiers",
                pattern_type="bullish",
                strength="strong",
                confidence=80,
                candle_count=3,
                description="Strong bullish continuation - sustained buying pressure"
            ))
        
        # Three Black Crows (bearish continuation)
        if (first.is_bearish and second.is_bearish and third.is_bearish and
            second.close < first.close and third.close < second.close and
            first.body_pct > 50 and second.body_pct > 50 and third.body_pct > 50):
            patterns.append(CandlePattern(
                name="Three Black Crows",
                pattern_type="bearish",
                strength="strong",
                confidence=80,
                candle_count=3,
                description="Strong bearish continuation - sustained selling pressure"
            ))
        
        return patterns
    
    def get_pattern_summary(self, ohlcv_data: List[Dict]) -> Dict[str, Any]:
        """
        Get summary of detected patterns with overall bias.
        
        Args:
            ohlcv_data: OHLCV data
            
        Returns:
            Dict with patterns and bias analysis
        """
        patterns = self.detect_patterns(ohlcv_data)
        
        if not patterns:
            return {
                "patterns": [],
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "overall_bias": "NEUTRAL",
                "confidence": 50,
                "description": "No significant patterns detected"
            }
        
        bullish = [p for p in patterns if p.pattern_type == "bullish"]
        bearish = [p for p in patterns if p.pattern_type == "bearish"]
        neutral = [p for p in patterns if p.pattern_type == "neutral"]
        
        # Calculate weighted score
        bullish_score = sum(p.confidence * (1.5 if p.strength == "strong" else 1.0) for p in bullish)
        bearish_score = sum(p.confidence * (1.5 if p.strength == "strong" else 1.0) for p in bearish)
        
        # Determine bias
        if bullish_score > bearish_score * 1.3:
            bias = "BULLISH"
            confidence = min(90, 50 + (bullish_score - bearish_score) / 5)
        elif bearish_score > bullish_score * 1.3:
            bias = "BEARISH"
            confidence = min(90, 50 + (bearish_score - bullish_score) / 5)
        elif bullish_score > bearish_score:
            bias = "LEAN_BULLISH"
            confidence = 55
        elif bearish_score > bullish_score:
            bias = "LEAN_BEARISH"
            confidence = 55
        else:
            bias = "NEUTRAL"
            confidence = 50
        
        # Build description
        top_pattern = patterns[0] if patterns else None
        description = top_pattern.description if top_pattern else "Mixed signals"
        
        return {
            "patterns": [
                {
                    "name": p.name,
                    "type": p.pattern_type,
                    "strength": p.strength,
                    "confidence": p.confidence
                }
                for p in patterns[:5]  # Top 5 patterns
            ],
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "overall_bias": bias,
            "confidence": round(confidence),
            "description": description
        }
    
    def get_pattern_context(self, ohlcv_data: List[Dict]) -> str:
        """
        Get formatted pattern context for LLM prompt.
        
        Args:
            ohlcv_data: OHLCV data
            
        Returns:
            Formatted string describing detected patterns
        """
        summary = self.get_pattern_summary(ohlcv_data)
        
        patterns_text = ""
        for p in summary["patterns"][:3]:
            patterns_text += f"\n  - {p['name']} ({p['type']}, {p['strength']}, {p['confidence']}%)"
        
        if not patterns_text:
            patterns_text = "\n  - No significant patterns"
        
        context = f"""
## Candlestick Patterns
Detected Patterns:{patterns_text}

Pattern Analysis:
- Bullish patterns: {summary['bullish_count']}
- Bearish patterns: {summary['bearish_count']}
- Neutral patterns: {summary['neutral_count']}
- Overall Bias: {summary['overall_bias']} (confidence: {summary['confidence']}%)
- Key Signal: {summary['description']}
"""
        return context.strip()


# Convenience function
def detect_patterns(ohlcv_data: List[Dict]) -> Dict[str, Any]:
    """Quick access to pattern detection."""
    detector = PatternDetector()
    return detector.get_pattern_summary(ohlcv_data)
