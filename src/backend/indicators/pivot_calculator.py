"""
Pivot Point Calculator - Support and Resistance Levels
Calculates pivot points using multiple methods for technical analysis
"""

import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PivotLevels:
    """Container for pivot point levels."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float
    method: str


class PivotCalculator:
    """
    Calculate pivot points and support/resistance levels.
    
    Supports multiple calculation methods:
    - Standard (Floor): Classic pivot points
    - Fibonacci: Fibonacci retracement-based pivots
    - Woodie: Weighted close calculation
    - Camarilla: Tighter range pivots
    - DeMark: Conditional pivot calculation
    """
    
    def __init__(self):
        pass
    
    def calculate_standard(self, high: float, low: float, close: float) -> PivotLevels:
        """
        Calculate Standard (Floor) Pivot Points.
        
        Formula:
        - Pivot = (High + Low + Close) / 3
        - R1 = (2 * Pivot) - Low
        - R2 = Pivot + (High - Low)
        - R3 = High + 2 * (Pivot - Low)
        - S1 = (2 * Pivot) - High
        - S2 = Pivot - (High - Low)
        - S3 = Low - 2 * (High - Pivot)
        """
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return PivotLevels(
            pivot=round(pivot, 2),
            r1=round(r1, 2),
            r2=round(r2, 2),
            r3=round(r3, 2),
            s1=round(s1, 2),
            s2=round(s2, 2),
            s3=round(s3, 2),
            method="standard"
        )
    
    def calculate_fibonacci(self, high: float, low: float, close: float) -> PivotLevels:
        """
        Calculate Fibonacci Pivot Points.
        
        Uses Fibonacci retracement levels (0.382, 0.618, 1.0)
        """
        pivot = (high + low + close) / 3
        range_hl = high - low
        
        r1 = pivot + (0.382 * range_hl)
        r2 = pivot + (0.618 * range_hl)
        r3 = pivot + (1.0 * range_hl)
        s1 = pivot - (0.382 * range_hl)
        s2 = pivot - (0.618 * range_hl)
        s3 = pivot - (1.0 * range_hl)
        
        return PivotLevels(
            pivot=round(pivot, 2),
            r1=round(r1, 2),
            r2=round(r2, 2),
            r3=round(r3, 2),
            s1=round(s1, 2),
            s2=round(s2, 2),
            s3=round(s3, 2),
            method="fibonacci"
        )
    
    def calculate_woodie(self, high: float, low: float, close: float) -> PivotLevels:
        """
        Calculate Woodie's Pivot Points.
        
        Gives more weight to the closing price.
        Pivot = (High + Low + 2*Close) / 4
        """
        pivot = (high + low + 2 * close) / 4
        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = r1 + (high - low)
        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = s1 - (high - low)
        
        return PivotLevels(
            pivot=round(pivot, 2),
            r1=round(r1, 2),
            r2=round(r2, 2),
            r3=round(r3, 2),
            s1=round(s1, 2),
            s2=round(s2, 2),
            s3=round(s3, 2),
            method="woodie"
        )
    
    def calculate_camarilla(self, high: float, low: float, close: float) -> PivotLevels:
        """
        Calculate Camarilla Pivot Points.
        
        Produces tighter support/resistance levels,
        good for intraday trading.
        """
        pivot = (high + low + close) / 3
        range_hl = high - low
        
        r1 = close + (range_hl * 1.1 / 12)
        r2 = close + (range_hl * 1.1 / 6)
        r3 = close + (range_hl * 1.1 / 4)
        s1 = close - (range_hl * 1.1 / 12)
        s2 = close - (range_hl * 1.1 / 6)
        s3 = close - (range_hl * 1.1 / 4)
        
        return PivotLevels(
            pivot=round(pivot, 2),
            r1=round(r1, 2),
            r2=round(r2, 2),
            r3=round(r3, 2),
            s1=round(s1, 2),
            s2=round(s2, 2),
            s3=round(s3, 2),
            method="camarilla"
        )
    
    def calculate_demark(self, high: float, low: float, close: float, 
                         open_price: Optional[float] = None) -> PivotLevels:
        """
        Calculate DeMark Pivot Points.
        
        Conditional calculation based on open vs close relationship.
        """
        if open_price is None:
            open_price = close  # Fallback
        
        # DeMark X calculation depends on close vs open
        if close < open_price:
            x = high + (2 * low) + close
        elif close > open_price:
            x = (2 * high) + low + close
        else:
            x = high + low + (2 * close)
        
        pivot = x / 4
        r1 = x / 2 - low
        s1 = x / 2 - high
        
        # DeMark typically only calculates R1/S1
        # We'll extrapolate R2/R3/S2/S3 using standard method
        range_hl = high - low
        r2 = pivot + range_hl
        r3 = r1 + range_hl
        s2 = pivot - range_hl
        s3 = s1 - range_hl
        
        return PivotLevels(
            pivot=round(pivot, 2),
            r1=round(r1, 2),
            r2=round(r2, 2),
            r3=round(r3, 2),
            s1=round(s1, 2),
            s2=round(s2, 2),
            s3=round(s3, 2),
            method="demark"
        )
    
    def calculate_all_methods(self, high: float, low: float, close: float,
                              open_price: Optional[float] = None) -> Dict[str, PivotLevels]:
        """
        Calculate pivot points using all available methods.
        
        Returns:
            Dict mapping method name to PivotLevels
        """
        return {
            "standard": self.calculate_standard(high, low, close),
            "fibonacci": self.calculate_fibonacci(high, low, close),
            "woodie": self.calculate_woodie(high, low, close),
            "camarilla": self.calculate_camarilla(high, low, close),
            "demark": self.calculate_demark(high, low, close, open_price),
        }
    
    def analyze_price_position(self, current_price: float, 
                               pivots: PivotLevels) -> Dict[str, Any]:
        """
        Analyze current price position relative to pivot levels.
        
        Args:
            current_price: Current market price
            pivots: PivotLevels object
            
        Returns:
            Dict with position analysis and trading bias
        """
        # Determine zone
        if current_price >= pivots.r3:
            zone = "ABOVE_R3"
            nearest_support = pivots.r3
            nearest_resistance = None
            bias = "OVERBOUGHT"
        elif current_price >= pivots.r2:
            zone = "R2_TO_R3"
            nearest_support = pivots.r2
            nearest_resistance = pivots.r3
            bias = "STRONG_RESISTANCE"
        elif current_price >= pivots.r1:
            zone = "R1_TO_R2"
            nearest_support = pivots.r1
            nearest_resistance = pivots.r2
            bias = "RESISTANCE"
        elif current_price >= pivots.pivot:
            zone = "ABOVE_PIVOT"
            nearest_support = pivots.pivot
            nearest_resistance = pivots.r1
            bias = "LEAN_BULLISH"
        elif current_price >= pivots.s1:
            zone = "BELOW_PIVOT"
            nearest_support = pivots.s1
            nearest_resistance = pivots.pivot
            bias = "LEAN_BEARISH"
        elif current_price >= pivots.s2:
            zone = "S1_TO_S2"
            nearest_support = pivots.s2
            nearest_resistance = pivots.s1
            bias = "SUPPORT"
        elif current_price >= pivots.s3:
            zone = "S2_TO_S3"
            nearest_support = pivots.s3
            nearest_resistance = pivots.s2
            bias = "STRONG_SUPPORT"
        else:
            zone = "BELOW_S3"
            nearest_support = None
            nearest_resistance = pivots.s3
            bias = "OVERSOLD"
        
        # Calculate distance to key levels
        distance_to_pivot = current_price - pivots.pivot
        distance_to_pivot_pct = (distance_to_pivot / pivots.pivot) * 100 if pivots.pivot else 0
        
        return {
            "current_price": current_price,
            "pivot": pivots.pivot,
            "zone": zone,
            "bias": bias,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "distance_to_pivot": round(distance_to_pivot, 2),
            "distance_to_pivot_pct": round(distance_to_pivot_pct, 2),
            "levels": {
                "r3": pivots.r3,
                "r2": pivots.r2,
                "r1": pivots.r1,
                "pivot": pivots.pivot,
                "s1": pivots.s1,
                "s2": pivots.s2,
                "s3": pivots.s3,
            }
        }
    
    def get_pivot_context(self, current_price: float, high: float, 
                          low: float, close: float,
                          method: str = "fibonacci") -> str:
        """
        Get formatted pivot point context for LLM prompt.
        
        Args:
            current_price: Current market price
            high: Previous period high
            low: Previous period low
            close: Previous period close
            method: Calculation method to use
            
        Returns:
            Formatted string describing pivot levels and position
        """
        # Calculate pivots
        if method == "standard":
            pivots = self.calculate_standard(high, low, close)
        elif method == "fibonacci":
            pivots = self.calculate_fibonacci(high, low, close)
        elif method == "woodie":
            pivots = self.calculate_woodie(high, low, close)
        elif method == "camarilla":
            pivots = self.calculate_camarilla(high, low, close)
        else:
            pivots = self.calculate_fibonacci(high, low, close)
        
        # Analyze position
        analysis = self.analyze_price_position(current_price, pivots)
        
        context = f"""
## Pivot Points ({method.capitalize()} Method)
- R3: ${pivots.r3:,.2f} (Strong Resistance)
- R2: ${pivots.r2:,.2f}
- R1: ${pivots.r1:,.2f}
- Pivot: ${pivots.pivot:,.2f}
- S1: ${pivots.s1:,.2f}
- S2: ${pivots.s2:,.2f}
- S3: ${pivots.s3:,.2f} (Strong Support)

Current Position:
- Price: ${current_price:,.2f}
- Zone: {analysis['zone']}
- Bias: {analysis['bias']}
- Distance from Pivot: {analysis['distance_to_pivot_pct']:+.2f}%
- Nearest Support: ${analysis['nearest_support']:,.2f} if analysis['nearest_support'] else 'None'
- Nearest Resistance: ${analysis['nearest_resistance']:,.2f} if analysis['nearest_resistance'] else 'None'
"""
        return context.strip()


# Convenience function
def calculate_pivots(high: float, low: float, close: float, 
                     method: str = "fibonacci") -> PivotLevels:
    """Quick access to pivot calculation."""
    calc = PivotCalculator()
    methods = {
        "standard": calc.calculate_standard,
        "fibonacci": calc.calculate_fibonacci,
        "woodie": calc.calculate_woodie,
        "camarilla": calc.calculate_camarilla,
    }
    return methods.get(method, calc.calculate_fibonacci)(high, low, close)
