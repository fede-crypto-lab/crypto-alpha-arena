"""
Weighted Scoring System for Trading Decisions
Framework v3 - Forward Testing Version

Calculates a weighted score based on multiple indicators,
adjusts for volatility, and suggests position sizing.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Result of the scoring calculation."""
    asset: str
    timestamp: str

    # Raw signals (-1 to +1)
    signals: Dict[str, float]

    # Weights used (may be adjusted for volatility)
    weights: Dict[str, float]

    # Calculated values
    raw_score: float          # -1 to +1
    final_score: float        # after FNG filter
    base_confidence: float    # 0.5 to 1.0
    final_confidence: float   # after volatility adjustment

    # Suggested action
    suggested_action: str     # BUY, SELL, HOLD
    suggested_size_pct: float # % of max position

    # TP/SL suggestions
    suggested_tp: Optional[float]
    suggested_sl: Optional[float]

    # Context
    atr_ratio: float
    fng_value: int
    btc_trend: Optional[float]  # for altcoins

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "timestamp": self.timestamp,
            "signals": self.signals,
            "weights": self.weights,
            "raw_score": round(self.raw_score, 4),
            "final_score": round(self.final_score, 4),
            "base_confidence": round(self.base_confidence, 4),
            "final_confidence": round(self.final_confidence, 4),
            "suggested_action": self.suggested_action,
            "suggested_size_pct": round(self.suggested_size_pct, 4),
            "suggested_tp": self.suggested_tp,
            "suggested_sl": self.suggested_sl,
            "atr_ratio": round(self.atr_ratio, 4) if self.atr_ratio else None,
            "fng_value": self.fng_value,
            "btc_trend": self.btc_trend
        }


class WeightedScorer:
    """
    Weighted scoring system for trading decisions.

    Calculates scores based on:
    - Funding rate (cost of holding)
    - Supertrend (macro trend)
    - RSI (momentum)
    - BTC correlation (for altcoins)
    - OBV divergence (volume confirmation)
    - Pivot zones (support/resistance)
    - Candlestick patterns (filtered)
    """

    # Base weights (will be adjusted dynamically)
    BASE_WEIGHTS = {
        "funding": 0.20,
        "supertrend": 0.20,
        "rsi": 0.15,
        "btc_correlation": 0.15,
        "obv": 0.15,
        "pivot": 0.10,
        "pattern": 0.05
    }

    # Position limits per asset
    POSITION_LIMITS = {
        "BTC": 0.40,
        "ETH": 0.30,
        "SOL": 0.15,
        "default": 0.10
    }

    # Global limits
    MAX_TOTAL_EXPOSURE = 1.5      # 150% with leverage
    MAX_CORRELATED = 0.50         # 50% on correlated assets
    MAX_SINGLE_TRADE = 0.15       # 15% per single trade

    def __init__(self, btc_supertrend: Optional[float] = None):
        """
        Initialize scorer.

        Args:
            btc_supertrend: BTC supertrend signal for altcoin correlation (-1, 0, or 1)
        """
        self.btc_supertrend = btc_supertrend
        self._atr_sma20_cache: Dict[str, float] = {}

    def calculate_score(
        self,
        asset: str,
        market_data: Dict[str, Any],
        fng_value: int = 50,
        current_price: Optional[float] = None,
        atr_sma20: Optional[float] = None
    ) -> ScoringResult:
        """
        Calculate weighted score for an asset.

        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            market_data: Market data dict from bot_engine
            fng_value: Fear & Greed index (0-100)
            current_price: Current price (for TP/SL calculation)
            atr_sma20: 20-period SMA of ATR (for volatility ratio)

        Returns:
            ScoringResult with all calculated values
        """
        timestamp = datetime.utcnow().isoformat()

        # Extract signals from market data
        signals = self._extract_signals(asset, market_data, fng_value)

        # Calculate ATR ratio for dynamic weight adjustment
        atr_current = market_data.get("long_term", {}).get("atr14")
        atr_ratio = self._calculate_atr_ratio(asset, atr_current, atr_sma20)

        # Adjust weights based on volatility
        weights = self._adjust_weights_for_volatility(atr_ratio, asset)

        # Calculate raw weighted score
        raw_score = self._calculate_weighted_score(signals, weights)

        # Apply FNG filter
        final_score = self._apply_fng_filter(raw_score, fng_value)

        # Calculate confidence
        base_confidence = self._calculate_base_confidence(signals, final_score)
        confidence_multiplier = self._get_volatility_confidence_multiplier(atr_ratio)
        final_confidence = base_confidence * confidence_multiplier

        # Determine action
        suggested_action = self._score_to_action(final_score)

        # Calculate position size
        suggested_size_pct = self._calculate_position_size(
            asset, final_score, final_confidence
        )

        # Calculate TP/SL
        suggested_tp, suggested_sl = None, None
        if current_price and atr_current and suggested_action != "HOLD":
            suggested_tp, suggested_sl = self._calculate_dynamic_tpsl(
                current_price, atr_current, final_score,
                is_long=(suggested_action == "BUY")
            )

        return ScoringResult(
            asset=asset,
            timestamp=timestamp,
            signals=signals,
            weights=weights,
            raw_score=raw_score,
            final_score=final_score,
            base_confidence=base_confidence,
            final_confidence=final_confidence,
            suggested_action=suggested_action,
            suggested_size_pct=suggested_size_pct,
            suggested_tp=suggested_tp,
            suggested_sl=suggested_sl,
            atr_ratio=atr_ratio if atr_ratio else 1.0,
            fng_value=fng_value,
            btc_trend=self.btc_supertrend if asset != "BTC" else None
        )

    def _extract_signals(
        self,
        asset: str,
        market_data: Dict[str, Any],
        fng_value: int
    ) -> Dict[str, float]:
        """Extract normalized signals (-1 to +1) from market data."""
        signals = {}

        # 1. Funding signal
        funding_annual = market_data.get("funding_annualized_pct", 0)
        signals["funding"] = self._normalize_funding(funding_annual)

        # 2. Supertrend signal
        supertrend = market_data.get("long_term", {}).get("supertrend", {})
        advice = supertrend.get("advice")
        if advice == "long":
            signals["supertrend"] = 1.0
        elif advice == "short":
            signals["supertrend"] = -1.0
        else:
            signals["supertrend"] = 0.0

        # 3. RSI signal
        rsi = market_data.get("intraday", {}).get("rsi14")
        signals["rsi"] = self._normalize_rsi(rsi)

        # 4. BTC correlation (for altcoins only)
        if asset != "BTC" and self.btc_supertrend is not None:
            signals["btc_correlation"] = self.btc_supertrend
        else:
            signals["btc_correlation"] = 0.0

        # 5. OBV signal
        obv_trend = self._extract_obv_trend(market_data)
        signals["obv"] = obv_trend

        # 6. Pivot zone signal
        enhanced = market_data.get("enhanced_analysis", "")
        signals["pivot"] = self._extract_pivot_signal(enhanced)

        # 7. Pattern signal (only high-reliability patterns)
        signals["pattern"] = self._extract_pattern_signal(enhanced)

        return signals

    def _normalize_funding(self, funding_annual_pct: float) -> float:
        """
        Normalize funding rate to signal.
        High positive funding = expensive to long = bearish signal
        High negative funding = expensive to short = bullish signal
        """
        if funding_annual_pct is None:
            return 0.0

        # Threshold: >100% annual = strong signal
        if funding_annual_pct > 100:
            return -1.0  # very expensive to long
        elif funding_annual_pct > 50:
            return -0.5
        elif funding_annual_pct < -100:
            return 1.0   # very expensive to short (bullish for longs)
        elif funding_annual_pct < -50:
            return 0.5
        else:
            return 0.0

    def _normalize_rsi(self, rsi: Optional[float]) -> float:
        """Normalize RSI to signal."""
        if rsi is None:
            return 0.0

        if rsi > 70:
            return -1.0  # overbought
        elif rsi > 60:
            return -0.5
        elif rsi < 30:
            return 1.0   # oversold
        elif rsi < 40:
            return 0.5
        else:
            return 0.0

    def _extract_obv_trend(self, market_data: Dict[str, Any]) -> float:
        """Extract OBV trend signal from enhanced analysis."""
        enhanced = market_data.get("enhanced_analysis", "")

        if "BULLISH_DIVERGENCE" in enhanced:
            return 1.0
        elif "BEARISH_DIVERGENCE" in enhanced:
            return -1.0
        elif "OBV Trend: RISING" in enhanced:
            return 0.5
        elif "OBV Trend: FALLING" in enhanced:
            return -0.5
        else:
            return 0.0

    def _extract_pivot_signal(self, enhanced: str) -> float:
        """Extract pivot zone signal."""
        if "ABOVE_R3" in enhanced or "OVERBOUGHT" in enhanced:
            return -1.0
        elif "ABOVE_R2" in enhanced:
            return -0.5
        elif "BELOW_S3" in enhanced or "OVERSOLD" in enhanced:
            return 1.0
        elif "BELOW_S2" in enhanced:
            return 0.5
        elif "ABOVE_PIVOT" in enhanced:
            return 0.25
        elif "BELOW_PIVOT" in enhanced:
            return -0.25
        else:
            return 0.0

    def _extract_pattern_signal(self, enhanced: str) -> float:
        """
        Extract pattern signal.
        Only high-reliability patterns at key levels.
        """
        # Bullish patterns
        if "Bullish Engulfing" in enhanced:
            return 1.0
        elif "Hammer" in enhanced and ("S1" in enhanced or "S2" in enhanced or "OVERSOLD" in enhanced):
            return 0.8
        elif "Bullish Marubozu" in enhanced:
            return 0.6

        # Bearish patterns
        elif "Bearish Engulfing" in enhanced:
            return -1.0
        elif "Shooting Star" in enhanced and ("R1" in enhanced or "R2" in enhanced or "OVERBOUGHT" in enhanced):
            return -0.8
        elif "Bearish Marubozu" in enhanced:
            return -0.6

        # Neutral patterns
        elif "Doji" in enhanced:
            return 0.0

        return 0.0

    def _calculate_atr_ratio(
        self,
        asset: str,
        atr_current: Optional[float],
        atr_sma20: Optional[float]
    ) -> Optional[float]:
        """Calculate ATR ratio for volatility assessment."""
        if atr_current is None:
            return None

        if atr_sma20 is not None:
            return atr_current / atr_sma20 if atr_sma20 > 0 else 1.0

        # Use cached SMA if available
        if asset in self._atr_sma20_cache:
            cached = self._atr_sma20_cache[asset]
            return atr_current / cached if cached > 0 else 1.0

        # First time: assume ratio = 1
        self._atr_sma20_cache[asset] = atr_current
        return 1.0

    def _adjust_weights_for_volatility(
        self,
        atr_ratio: Optional[float],
        asset: str
    ) -> Dict[str, float]:
        """Adjust weights based on volatility regime."""
        weights = self.BASE_WEIGHTS.copy()

        # For BTC, remove btc_correlation
        if asset == "BTC":
            weights["btc_correlation"] = 0.0

        if atr_ratio is None:
            # Normalize and return
            total = sum(weights.values())
            return {k: v / total for k, v in weights.items()}

        # High volatility: pivots less reliable, trend more reliable
        if atr_ratio > 1.5:
            weights["pivot"] *= 0.5
            weights["supertrend"] *= 1.2

        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _calculate_weighted_score(
        self,
        signals: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """Calculate weighted sum of signals."""
        score = 0.0
        for key, weight in weights.items():
            signal = signals.get(key, 0.0)
            score += signal * weight
        return max(-1.0, min(1.0, score))

    def _apply_fng_filter(self, raw_score: float, fng_value: int) -> float:
        """
        Apply Fear & Greed filter.
        Reduces (not blocks) contrarian positions.
        """
        if fng_value < 20 and raw_score < 0:
            # Extreme fear: reduce short bias by 50%
            return raw_score * 0.5
        elif fng_value > 80 and raw_score > 0:
            # Extreme greed: reduce long bias by 50%
            return raw_score * 0.5
        return raw_score

    def _calculate_base_confidence(
        self,
        signals: Dict[str, float],
        final_score: float
    ) -> float:
        """
        Calculate base confidence based on signal concordance.
        Range: 0.5 to 1.0
        """
        if final_score == 0:
            return 0.5

        score_direction = 1 if final_score > 0 else -1

        concordant = 0
        total = 0

        for key, value in signals.items():
            if value == 0:
                continue
            total += 1
            if (value > 0 and score_direction > 0) or (value < 0 and score_direction < 0):
                concordant += 1

        if total == 0:
            return 0.5

        return 0.5 + 0.5 * (concordant / total)

    def _get_volatility_confidence_multiplier(self, atr_ratio: Optional[float]) -> float:
        """Get confidence multiplier based on volatility."""
        if atr_ratio is None:
            return 1.0

        if atr_ratio < 0.7:
            return 0.7  # Low volatility = signals less reliable
        else:
            return 1.0

    def _score_to_action(self, final_score: float) -> str:
        """Convert score to action."""
        if final_score > 0.2:
            return "BUY"
        elif final_score < -0.2:
            return "SELL"
        else:
            return "HOLD"

    def _calculate_position_size(
        self,
        asset: str,
        final_score: float,
        confidence: float
    ) -> float:
        """
        Calculate suggested position size as % of max.

        size = base_limit * |score| * confidence
        """
        base_limit = self.POSITION_LIMITS.get(asset, self.POSITION_LIMITS["default"])
        max_trade = min(base_limit, self.MAX_SINGLE_TRADE)

        size_pct = max_trade * abs(final_score) * confidence
        return min(size_pct, max_trade)

    def _calculate_dynamic_tpsl(
        self,
        entry_price: float,
        atr: float,
        final_score: float,
        is_long: bool
    ) -> Tuple[float, float]:
        """
        Calculate dynamic TP/SL based on ATR and score magnitude.

        Score forte → TP più aggressivo, SL più stretto
        Score debole → TP conservativo, SL più largo
        """
        score_magnitude = abs(final_score)

        # SL: 2.0 ATR at low score → 1.2 ATR at high score
        sl_multiplier = 2.0 - (score_magnitude * 0.8)

        # TP: 1.5 ATR at low score → 2.5 ATR at high score
        tp_multiplier = 1.5 + (score_magnitude * 1.0)

        sl_distance = atr * sl_multiplier
        tp_distance = atr * tp_multiplier

        if is_long:
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance

        return round(tp_price, 6), round(sl_price, 6)


def score_all_assets(
    market_sections: List[Dict[str, Any]],
    fng_value: int = 50
) -> Dict[str, ScoringResult]:
    """
    Score all assets in a market data list.

    Args:
        market_sections: List of market data dicts from bot_engine
        fng_value: Fear & Greed index

    Returns:
        Dict mapping asset to ScoringResult
    """
    results = {}

    # First pass: get BTC supertrend for correlation
    btc_supertrend = None
    for section in market_sections:
        if section.get("asset") == "BTC":
            st = section.get("long_term", {}).get("supertrend", {})
            advice = st.get("advice")
            if advice == "long":
                btc_supertrend = 1.0
            elif advice == "short":
                btc_supertrend = -1.0
            break

    scorer = WeightedScorer(btc_supertrend=btc_supertrend)

    # Score each asset
    for section in market_sections:
        asset = section.get("asset")
        if not asset:
            continue

        current_price = section.get("current_price")

        result = scorer.calculate_score(
            asset=asset,
            market_data=section,
            fng_value=fng_value,
            current_price=current_price
        )

        results[asset] = result

    return results
