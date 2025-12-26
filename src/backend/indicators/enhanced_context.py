"""
Enhanced Market Context - Combines all advanced indicators into unified context
Main integration module for LLM decision making
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .sentiment_client import get_shared_client
from .whale_alert_client import WhaleAlertClient
from .pivot_calculator import PivotCalculator
from .pattern_detector import PatternDetector
from .volume_analyzer import VolumeAnalyzer

logger = logging.getLogger(__name__)


class EnhancedMarketContext:
    """
    Aggregates all advanced indicators into unified market context.
    
    Provides:
    - Fear & Greed sentiment
    - Whale activity analysis
    - Pivot point levels
    - Candlestick patterns
    - Volume analysis
    - Composite signal calculation
    """
    
    # Signal weights for composite calculation
    WEIGHTS = {
        "sentiment": 1.0,
        "whale_alert": 1.5,  # High impact
        "stablecoin_flow": 1.0,
        "pivot_position": 0.75,
        "patterns": 1.25,
        "volume": 1.0,
        "volume_divergence": 1.5,  # Strong reversal signal
    }
    
    def __init__(self, 
                 sentiment_cache_ttl: int = 3600,
                 whale_cache_ttl: int = 300):
        """
        Initialize enhanced context with all indicator clients.
        
        Args:
            sentiment_cache_ttl: Cache TTL for sentiment data (default: 1 hour)
            whale_cache_ttl: Cache TTL for whale data (default: 5 minutes)
        """
        self.sentiment = get_shared_client(cache_ttl=sentiment_cache_ttl)
        self.whale = WhaleAlertClient(cache_ttl=whale_cache_ttl)
        self.pivot = PivotCalculator()
        self.pattern = PatternDetector()
        self.volume = VolumeAnalyzer()
        
        logger.info("EnhancedMarketContext initialized with all indicator modules")
    
    def build_context(self,
                      asset: str,
                      current_price: float,
                      ohlcv_data: Optional[List[Dict]] = None,
                      daily_ohlc: Optional[Dict] = None,
                      timeframe: str = "4h") -> Dict[str, Any]:
        """
        Build complete enhanced market context.
        
        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            current_price: Current market price
            ohlcv_data: OHLCV candle data for patterns/volume
            daily_ohlc: Daily high/low/close for pivot calculation
            timeframe: Timeframe for context description
            
        Returns:
            Dict with all indicator data and composite signal
        """
        context = {
            "asset": asset,
            "current_price": current_price,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
            "indicators": {},
            "signals": [],
            "composite_signal": "NEUTRAL",
            "composite_score": 0,
            "confidence": 50,
            "prompt_text": ""
        }
        
        # Track signal scores for composite
        signal_scores = []
        
        # 1. Fear & Greed Index
        try:
            fng = self.sentiment.get_fear_greed_index()
            if fng:
                context["indicators"]["sentiment"] = {
                    "value": fng["value"],
                    "classification": fng["classification"],
                    "trend": fng["trend"],
                    "signal": fng["signal"]
                }
                
                # Convert signal to score (-1 to +1)
                signal_map = {
                    "STRONG_BUY": 1.0, "BUY": 0.6, "LEAN_BUY": 0.3,
                    "NEUTRAL": 0, 
                    "LEAN_SELL": -0.3, "SELL": -0.6, "STRONG_SELL": -1.0
                }
                score = signal_map.get(fng["signal"], 0)
                signal_scores.append(("sentiment", score, self.WEIGHTS["sentiment"]))
                
                if abs(score) >= 0.6:
                    context["signals"].append(f"FNG_{fng['signal']}")
        except Exception as e:
            logger.warning(f"Failed to get Fear & Greed: {e}")
        
        # 2. Whale Activity
        try:
            whale_data = self.whale.analyze_whale_activity(asset=asset, hours=24)
            if not whale_data.get("error"):
                context["indicators"]["whale_activity"] = {
                    "exchange_inflows_usd": whale_data["exchange_inflows_usd"],
                    "exchange_outflows_usd": whale_data["exchange_outflows_usd"],
                    "net_flow_usd": whale_data["net_flow_usd"],
                    "stablecoin_to_exchange_usd": whale_data["stablecoin_to_exchange_usd"],
                    "signal": whale_data["signal"],
                    "confidence": whale_data["confidence"]
                }
                
                # Whale signal score
                whale_signal_map = {
                    "BULLISH": 0.8, "LEAN_BULLISH": 0.4,
                    "NEUTRAL": 0,
                    "LEAN_BEARISH": -0.4, "BEARISH": -0.8
                }
                whale_score = whale_signal_map.get(whale_data["signal"], 0)
                signal_scores.append(("whale_alert", whale_score, self.WEIGHTS["whale_alert"]))
                
                # Stablecoin flow as separate signal
                if whale_data["stablecoin_to_exchange_usd"] > 50_000_000:
                    signal_scores.append(("stablecoin_flow", 0.5, self.WEIGHTS["stablecoin_flow"]))
                    context["signals"].append("STABLECOIN_INFLOW_BULLISH")
                
                if whale_data["signal"] in ["BULLISH", "BEARISH"]:
                    context["signals"].append(f"WHALE_{whale_data['signal']}")
        except Exception as e:
            logger.warning(f"Failed to get whale data: {e}")
        
        # 3. Pivot Points
        if daily_ohlc and all(k in daily_ohlc for k in ['high', 'low', 'close']):
            try:
                pivots = self.pivot.calculate_fibonacci(
                    high=daily_ohlc['high'],
                    low=daily_ohlc['low'],
                    close=daily_ohlc['close']
                )
                analysis = self.pivot.analyze_price_position(current_price, pivots)
                
                context["indicators"]["pivot_points"] = {
                    "pivot": pivots.pivot,
                    "r1": pivots.r1, "r2": pivots.r2, "r3": pivots.r3,
                    "s1": pivots.s1, "s2": pivots.s2, "s3": pivots.s3,
                    "zone": analysis["zone"],
                    "bias": analysis["bias"],
                    "distance_to_pivot_pct": analysis["distance_to_pivot_pct"]
                }
                
                # Pivot position score
                pivot_bias_map = {
                    "OVERBOUGHT": -0.5, "STRONG_RESISTANCE": -0.4, "RESISTANCE": -0.2,
                    "LEAN_BULLISH": 0.2, "LEAN_BEARISH": -0.2,
                    "SUPPORT": 0.2, "STRONG_SUPPORT": 0.4, "OVERSOLD": 0.5
                }
                pivot_score = pivot_bias_map.get(analysis["bias"], 0)
                signal_scores.append(("pivot_position", pivot_score, self.WEIGHTS["pivot_position"]))
                
                if analysis["bias"] in ["OVERBOUGHT", "OVERSOLD", "STRONG_SUPPORT", "STRONG_RESISTANCE"]:
                    context["signals"].append(f"PIVOT_{analysis['bias']}")
            except Exception as e:
                logger.warning(f"Failed to calculate pivots: {e}")
        
        # 4. Candlestick Patterns
        if ohlcv_data and len(ohlcv_data) >= 3:
            try:
                pattern_summary = self.pattern.get_pattern_summary(ohlcv_data)
                context["indicators"]["patterns"] = {
                    "detected": pattern_summary["patterns"][:3],
                    "bullish_count": pattern_summary["bullish_count"],
                    "bearish_count": pattern_summary["bearish_count"],
                    "bias": pattern_summary["overall_bias"],
                    "confidence": pattern_summary["confidence"]
                }
                
                # Pattern score
                pattern_bias_map = {
                    "BULLISH": 0.7, "LEAN_BULLISH": 0.35,
                    "NEUTRAL": 0,
                    "LEAN_BEARISH": -0.35, "BEARISH": -0.7
                }
                pattern_score = pattern_bias_map.get(pattern_summary["overall_bias"], 0)
                signal_scores.append(("patterns", pattern_score, self.WEIGHTS["patterns"]))
                
                if pattern_summary["overall_bias"] in ["BULLISH", "BEARISH"]:
                    top_pattern = pattern_summary["patterns"][0]["name"] if pattern_summary["patterns"] else "Unknown"
                    context["signals"].append(f"PATTERN_{pattern_summary['overall_bias']}_{top_pattern}")
            except Exception as e:
                logger.warning(f"Failed to detect patterns: {e}")
        
        # 5. Volume Analysis
        if ohlcv_data and len(ohlcv_data) >= 5:
            try:
                volume_bias = self.volume.get_volume_bias(ohlcv_data, current_price)
                context["indicators"]["volume"] = {
                    "obv_trend": volume_bias["obv_trend"],
                    "vwap": volume_bias["vwap"],
                    "volume_ratio": volume_bias["volume_ratio"],
                    "volume_signal": volume_bias["volume_signal"],
                    "divergence": volume_bias["divergence"],
                    "bias": volume_bias["bias"]
                }
                
                # Volume score
                volume_bias_map = {
                    "BULLISH": 0.5, "LEAN_BULLISH": 0.25,
                    "NEUTRAL": 0,
                    "LEAN_BEARISH": -0.25, "BEARISH": -0.5
                }
                volume_score = volume_bias_map.get(volume_bias["bias"], 0)
                signal_scores.append(("volume", volume_score, self.WEIGHTS["volume"]))
                
                # Divergence is a strong signal
                if volume_bias["divergence"] == "bullish":
                    signal_scores.append(("volume_divergence", 0.8, self.WEIGHTS["volume_divergence"]))
                    context["signals"].append("VOLUME_BULLISH_DIVERGENCE")
                elif volume_bias["divergence"] == "bearish":
                    signal_scores.append(("volume_divergence", -0.8, self.WEIGHTS["volume_divergence"]))
                    context["signals"].append("VOLUME_BEARISH_DIVERGENCE")
                
                if volume_bias["volume_signal"] in ["HIGH_VOLUME", "VERY_HIGH_VOLUME"]:
                    context["signals"].append(f"VOLUME_{volume_bias['volume_signal']}")
            except Exception as e:
                logger.warning(f"Failed to analyze volume: {e}")
        
        # 6. Calculate Composite Signal
        composite = self._calculate_composite_signal(signal_scores)
        context["composite_signal"] = composite["signal"]
        context["composite_score"] = composite["normalized_score"]
        context["confidence"] = composite["confidence"]
        
        # 7. Build LLM Prompt Text
        context["prompt_text"] = self._build_prompt_text(context)
        
        return context
    
    def _calculate_composite_signal(self, 
                                    signal_scores: List[tuple]) -> Dict[str, Any]:
        """
        Calculate weighted composite signal from all indicators.
        
        Args:
            signal_scores: List of (name, score, weight) tuples
            
        Returns:
            Dict with composite signal, normalized score, and confidence
        """
        if not signal_scores:
            return {
                "signal": "NEUTRAL",
                "normalized_score": 0,
                "raw_score": 0,
                "confidence": 50
            }
        
        # Calculate weighted average
        total_weight = sum(w for _, _, w in signal_scores)
        weighted_sum = sum(score * weight for _, score, weight in signal_scores)
        
        raw_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Normalize to -1 to +1
        normalized_score = max(-1, min(1, raw_score))
        
        # Determine signal
        if normalized_score >= 0.5:
            signal = "BULLISH"
        elif normalized_score >= 0.2:
            signal = "LEAN_BULLISH"
        elif normalized_score <= -0.5:
            signal = "BEARISH"
        elif normalized_score <= -0.2:
            signal = "LEAN_BEARISH"
        else:
            signal = "NEUTRAL"
        
        # Confidence based on signal agreement
        scores_only = [s for _, s, _ in signal_scores]
        if all(s > 0 for s in scores_only) or all(s < 0 for s in scores_only):
            confidence = min(90, 60 + len(scores_only) * 5)  # Strong agreement
        elif abs(sum(scores_only)) / len(scores_only) > 0.3:
            confidence = 65  # Moderate agreement
        else:
            confidence = 50  # Mixed signals
        
        return {
            "signal": signal,
            "normalized_score": round(normalized_score, 3),
            "raw_score": round(raw_score, 3),
            "confidence": confidence
        }
    
    def _build_prompt_text(self, context: Dict[str, Any]) -> str:
        """
        Build formatted text for LLM prompt.
        
        Args:
            context: Complete context dict
            
        Returns:
            Formatted string for prompt insertion
        """
        sections = []
        
        # Header
        sections.append(f"## Enhanced Market Analysis ({context['asset']})")
        sections.append(f"Time: {context['timestamp']}")
        sections.append(f"Current Price: ${context['current_price']:,.2f}")
        sections.append("")
        
        # Sentiment
        if "sentiment" in context["indicators"]:
            s = context["indicators"]["sentiment"]
            sections.append(f"### Market Sentiment (Fear & Greed)")
            sections.append(f"- Value: {s['value']}/100 ({s['classification']})")
            sections.append(f"- Trend: {s['trend']}")
            sections.append(f"- Contrarian Signal: {s['signal']}")
            sections.append("")
        
        # Whale Activity
        if "whale_activity" in context["indicators"]:
            w = context["indicators"]["whale_activity"]
            sections.append(f"### Whale Activity (24h)")
            sections.append(f"- Exchange Inflows: ${w['exchange_inflows_usd']/1e6:.1f}M")
            sections.append(f"- Exchange Outflows: ${w['exchange_outflows_usd']/1e6:.1f}M")
            sections.append(f"- Net Flow: ${w['net_flow_usd']/1e6:+.1f}M")
            sections.append(f"- Stablecoin to Exchanges: ${w['stablecoin_to_exchange_usd']/1e6:.1f}M")
            sections.append(f"- Signal: {w['signal']}")
            sections.append("")
        
        # Pivot Points
        if "pivot_points" in context["indicators"]:
            p = context["indicators"]["pivot_points"]
            sections.append(f"### Pivot Points (Fibonacci)")
            sections.append(f"- R3: ${p['r3']:,.2f} | R2: ${p['r2']:,.2f} | R1: ${p['r1']:,.2f}")
            sections.append(f"- Pivot: ${p['pivot']:,.2f}")
            sections.append(f"- S1: ${p['s1']:,.2f} | S2: ${p['s2']:,.2f} | S3: ${p['s3']:,.2f}")
            sections.append(f"- Zone: {p['zone']} | Bias: {p['bias']}")
            sections.append("")
        
        # Patterns
        if "patterns" in context["indicators"]:
            pt = context["indicators"]["patterns"]
            sections.append(f"### Candlestick Patterns")
            if pt["detected"]:
                for p in pt["detected"][:2]:
                    sections.append(f"- {p['name']} ({p['type']}, {p['confidence']}%)")
            sections.append(f"- Overall Bias: {pt['bias']} (confidence: {pt['confidence']}%)")
            sections.append("")
        
        # Volume
        if "volume" in context["indicators"]:
            v = context["indicators"]["volume"]
            sections.append(f"### Volume Analysis")
            sections.append(f"- OBV Trend: {v['obv_trend'].upper()}")
            sections.append(f"- VWAP: ${v['vwap']:,.2f}")
            sections.append(f"- Volume Ratio: {v['volume_ratio']:.2f}x ({v['volume_signal']})")
            if v["divergence"]:
                sections.append(f"- ⚠️ DIVERGENCE: {v['divergence'].upper()}")
            sections.append(f"- Bias: {v['bias']}")
            sections.append("")
        
        # Composite Signal
        sections.append(f"### COMPOSITE SIGNAL")
        sections.append(f"- Signal: **{context['composite_signal']}**")
        sections.append(f"- Normalized Score: {context['composite_score']:+.3f} (-1 to +1)")
        sections.append(f"- Confidence: {context['confidence']}%")
        
        if context["signals"]:
            sections.append(f"- Active Signals: {', '.join(context['signals'][:5])}")
        
        return "\n".join(sections)


# Convenience function for quick context building
def build_enhanced_context(asset: str, 
                           current_price: float,
                           ohlcv_data: Optional[List[Dict]] = None,
                           daily_ohlc: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Quick access to enhanced market context.
    
    Args:
        asset: Asset symbol
        current_price: Current price
        ohlcv_data: OHLCV data
        daily_ohlc: Daily OHLC for pivots
        
    Returns:
        Complete context dict
    """
    ctx = EnhancedMarketContext()
    return ctx.build_context(
        asset=asset,
        current_price=current_price,
        ohlcv_data=ohlcv_data,
        daily_ohlc=daily_ohlc
    )
