"""
Volume Analyzer - Advanced Volume Analysis for Trading
OBV, VWAP, Volume Profile, and Divergence Detection
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VolumeAnalysis:
    """Container for volume analysis results."""
    obv: float
    obv_trend: str  # 'rising', 'falling', 'flat'
    vwap: float
    volume_ratio: float  # Current vs average
    volume_signal: str  # 'HIGH_VOLUME', 'NORMAL', 'LOW_VOLUME'
    divergence: Optional[str]  # 'bullish', 'bearish', None
    confidence: float


class VolumeAnalyzer:
    """
    Advanced volume analysis for trading signals.
    
    Metrics calculated:
    - OBV (On-Balance Volume) with trend detection
    - VWAP (Volume-Weighted Average Price)
    - Volume ratio vs moving average
    - Price-Volume divergence detection
    """
    
    def __init__(self, ma_period: int = 20):
        """
        Initialize volume analyzer.
        
        Args:
            ma_period: Period for volume moving average (default: 20)
        """
        self.ma_period = ma_period
    
    def calculate_obv(self, ohlcv_data: List[Dict]) -> Tuple[List[float], float, str]:
        """
        Calculate On-Balance Volume (OBV).
        
        OBV adds volume on up days and subtracts on down days.
        Rising OBV with rising price = bullish confirmation
        Rising OBV with falling price = bullish divergence (potential reversal)
        
        Args:
            ohlcv_data: List of OHLCV dicts
            
        Returns:
            (obv_series, current_obv, trend)
        """
        if len(ohlcv_data) < 2:
            return [0], 0, "flat"
        
        obv_series = [0]  # Start at 0
        
        for i in range(1, len(ohlcv_data)):
            prev_close = float(ohlcv_data[i-1].get('close', ohlcv_data[i-1].get('c', 0)))
            curr_close = float(ohlcv_data[i].get('close', ohlcv_data[i].get('c', 0)))
            volume = float(ohlcv_data[i].get('volume', ohlcv_data[i].get('v', 0)))
            
            if curr_close > prev_close:
                obv_series.append(obv_series[-1] + volume)
            elif curr_close < prev_close:
                obv_series.append(obv_series[-1] - volume)
            else:
                obv_series.append(obv_series[-1])
        
        current_obv = obv_series[-1]
        
        # Determine trend using last 5 values
        if len(obv_series) >= 5:
            recent = obv_series[-5:]
            if recent[-1] > recent[0] * 1.05:
                trend = "rising"
            elif recent[-1] < recent[0] * 0.95:
                trend = "falling"
            else:
                trend = "flat"
        else:
            trend = "flat"
        
        return obv_series, current_obv, trend
    
    def calculate_vwap(self, ohlcv_data: List[Dict]) -> float:
        """
        Calculate Volume-Weighted Average Price (VWAP).
        
        VWAP = Sum(Price * Volume) / Sum(Volume)
        Using typical price: (High + Low + Close) / 3
        
        Args:
            ohlcv_data: List of OHLCV dicts
            
        Returns:
            VWAP value
        """
        if not ohlcv_data:
            return 0
        
        total_pv = 0  # Price * Volume
        total_volume = 0
        
        for d in ohlcv_data:
            high = float(d.get('high', d.get('h', 0)))
            low = float(d.get('low', d.get('l', 0)))
            close = float(d.get('close', d.get('c', 0)))
            volume = float(d.get('volume', d.get('v', 0)))
            
            typical_price = (high + low + close) / 3
            total_pv += typical_price * volume
            total_volume += volume
        
        if total_volume == 0:
            return 0
        
        return total_pv / total_volume
    
    def calculate_volume_ratio(self, ohlcv_data: List[Dict]) -> Tuple[float, str]:
        """
        Calculate current volume vs moving average ratio.
        
        Args:
            ohlcv_data: List of OHLCV dicts
            
        Returns:
            (ratio, signal) where ratio is current/average
        """
        if len(ohlcv_data) < self.ma_period:
            return 1.0, "NORMAL"
        
        volumes = []
        for d in ohlcv_data:
            volumes.append(float(d.get('volume', d.get('v', 0))))
        
        # Calculate MA
        ma_volumes = volumes[-self.ma_period:]
        avg_volume = sum(ma_volumes) / len(ma_volumes) if ma_volumes else 0
        
        # Current volume (last candle)
        current_volume = volumes[-1] if volumes else 0
        
        if avg_volume == 0:
            return 1.0, "NORMAL"
        
        ratio = current_volume / avg_volume
        
        # Determine signal
        if ratio >= 2.0:
            signal = "VERY_HIGH_VOLUME"
        elif ratio >= 1.5:
            signal = "HIGH_VOLUME"
        elif ratio >= 0.7:
            signal = "NORMAL"
        elif ratio >= 0.3:
            signal = "LOW_VOLUME"
        else:
            signal = "VERY_LOW_VOLUME"
        
        return round(ratio, 2), signal
    
    def detect_divergence(self, ohlcv_data: List[Dict], 
                          lookback: int = 10) -> Optional[str]:
        """
        Detect price-volume divergence.
        
        Bullish divergence: Price falling, OBV rising
        Bearish divergence: Price rising, OBV falling
        
        Args:
            ohlcv_data: List of OHLCV dicts
            lookback: Number of candles to analyze
            
        Returns:
            'bullish', 'bearish', or None
        """
        if len(ohlcv_data) < lookback:
            return None
        
        recent = ohlcv_data[-lookback:]
        
        # Calculate OBV for recent period
        obv_series, _, obv_trend = self.calculate_obv(recent)
        
        # Calculate price trend
        first_close = float(recent[0].get('close', recent[0].get('c', 0)))
        last_close = float(recent[-1].get('close', recent[-1].get('c', 0)))
        
        price_change_pct = ((last_close - first_close) / first_close) * 100 if first_close else 0
        
        # Divergence detection
        # Price down but OBV up = bullish divergence
        if price_change_pct < -2 and obv_trend == "rising":
            return "bullish"
        
        # Price up but OBV down = bearish divergence
        if price_change_pct > 2 and obv_trend == "falling":
            return "bearish"
        
        return None
    
    def analyze(self, ohlcv_data: List[Dict], 
                current_price: Optional[float] = None) -> VolumeAnalysis:
        """
        Perform complete volume analysis.
        
        Args:
            ohlcv_data: List of OHLCV dicts
            current_price: Current market price (optional)
            
        Returns:
            VolumeAnalysis object with all metrics
        """
        if not ohlcv_data or len(ohlcv_data) < 2:
            return VolumeAnalysis(
                obv=0,
                obv_trend="flat",
                vwap=0,
                volume_ratio=1.0,
                volume_signal="NORMAL",
                divergence=None,
                confidence=50
            )
        
        # Calculate all metrics
        obv_series, current_obv, obv_trend = self.calculate_obv(ohlcv_data)
        vwap = self.calculate_vwap(ohlcv_data)
        volume_ratio, volume_signal = self.calculate_volume_ratio(ohlcv_data)
        divergence = self.detect_divergence(ohlcv_data)
        
        # Calculate confidence based on signal strength
        confidence = 50
        if volume_signal in ["HIGH_VOLUME", "VERY_HIGH_VOLUME"]:
            confidence += 15
        if divergence:
            confidence += 20
        if obv_trend in ["rising", "falling"]:
            confidence += 10
        
        return VolumeAnalysis(
            obv=current_obv,
            obv_trend=obv_trend,
            vwap=round(vwap, 2),
            volume_ratio=volume_ratio,
            volume_signal=volume_signal,
            divergence=divergence,
            confidence=min(95, confidence)
        )
    
    def get_volume_bias(self, ohlcv_data: List[Dict],
                        current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Get trading bias from volume analysis.
        
        Args:
            ohlcv_data: OHLCV data
            current_price: Current price
            
        Returns:
            Dict with bias and signals
        """
        analysis = self.analyze(ohlcv_data, current_price)
        
        # Determine overall bias
        signals = []
        bias_score = 0  # -100 to +100
        
        # OBV trend
        if analysis.obv_trend == "rising":
            signals.append("OBV_RISING")
            bias_score += 20
        elif analysis.obv_trend == "falling":
            signals.append("OBV_FALLING")
            bias_score -= 20
        
        # Volume signal
        if analysis.volume_signal in ["HIGH_VOLUME", "VERY_HIGH_VOLUME"]:
            signals.append(analysis.volume_signal)
            # High volume confirms trend
            bias_score += 10 if bias_score > 0 else -10
        elif analysis.volume_signal in ["LOW_VOLUME", "VERY_LOW_VOLUME"]:
            signals.append("LOW_VOLUME_WARNING")
        
        # Divergence (strong signal)
        if analysis.divergence == "bullish":
            signals.append("BULLISH_DIVERGENCE")
            bias_score += 35
        elif analysis.divergence == "bearish":
            signals.append("BEARISH_DIVERGENCE")
            bias_score -= 35
        
        # VWAP position
        if current_price and analysis.vwap > 0:
            if current_price > analysis.vwap * 1.01:
                signals.append("ABOVE_VWAP")
                bias_score += 10
            elif current_price < analysis.vwap * 0.99:
                signals.append("BELOW_VWAP")
                bias_score -= 10
        
        # Determine bias string
        if bias_score >= 40:
            bias = "BULLISH"
        elif bias_score >= 15:
            bias = "LEAN_BULLISH"
        elif bias_score <= -40:
            bias = "BEARISH"
        elif bias_score <= -15:
            bias = "LEAN_BEARISH"
        else:
            bias = "NEUTRAL"
        
        return {
            "obv": analysis.obv,
            "obv_trend": analysis.obv_trend,
            "vwap": analysis.vwap,
            "volume_ratio": analysis.volume_ratio,
            "volume_signal": analysis.volume_signal,
            "divergence": analysis.divergence,
            "signals": signals,
            "bias": bias,
            "bias_score": bias_score,
            "confidence": analysis.confidence
        }
    
    def get_volume_context(self, ohlcv_data: List[Dict],
                           current_price: Optional[float] = None) -> str:
        """
        Get formatted volume context for LLM prompt.
        
        Args:
            ohlcv_data: OHLCV data
            current_price: Current price
            
        Returns:
            Formatted string describing volume analysis
        """
        bias = self.get_volume_bias(ohlcv_data, current_price)
        
        signals_text = ", ".join(bias["signals"]) if bias["signals"] else "No significant signals"
        
        divergence_text = ""
        if bias["divergence"]:
            if bias["divergence"] == "bullish":
                divergence_text = "\n⚠️ BULLISH DIVERGENCE: Price falling but buying volume increasing - potential reversal"
            else:
                divergence_text = "\n⚠️ BEARISH DIVERGENCE: Price rising but buying volume decreasing - potential reversal"
        
        context = f"""
## Volume Analysis
- OBV Trend: {bias['obv_trend'].upper()}
- VWAP: ${bias['vwap']:,.2f}
- Volume Ratio: {bias['volume_ratio']:.2f}x average ({bias['volume_signal']})
- Signals: {signals_text}
- Overall Bias: {bias['bias']} (confidence: {bias['confidence']}%){divergence_text}

Interpretation:
- Volume ratio >1.5x suggests strong conviction
- OBV rising with price = trend confirmation
- Divergence between price and OBV = potential reversal warning
"""
        return context.strip()


# Convenience function
def analyze_volume(ohlcv_data: List[Dict], 
                   current_price: Optional[float] = None) -> Dict[str, Any]:
    """Quick access to volume analysis."""
    analyzer = VolumeAnalyzer()
    return analyzer.get_volume_bias(ohlcv_data, current_price)
