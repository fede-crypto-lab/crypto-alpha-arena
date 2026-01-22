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
    final_confidence: float   # after volatility and RSI timing adjustment

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

    # RSI timing info
    rsi_raw: Optional[float] = None         # Raw RSI value
    rsi_timing_boost: Optional[float] = None  # Timing multiplier applied (1.3 or 0.7)

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
            "btc_trend": self.btc_trend,
            "rsi_raw": round(self.rsi_raw, 2) if self.rsi_raw else None,
            "rsi_timing_boost": round(self.rsi_timing_boost, 2) if self.rsi_timing_boost else None
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
    # NOTE: funding ridotto da 0.20 a 0.12 - può persistere settimane senza squeeze
    # supertrend aumentato da 0.20 a 0.25 - trend è più affidabile
    BASE_WEIGHTS = {
        "funding": 0.12,
        "supertrend": 0.25,
        "rsi": 0.15,
        "btc_correlation": 0.15,
        "obv": 0.15,
        "pivot": 0.10,
        "pattern": 0.08
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

        # ===== SUPERTREND PENALTY: Penalize counter-trend trades =====
        # Se supertrend è SHORT e score è positivo (BUY), riduci lo score del 50%
        # Se supertrend è LONG e score è negativo (SELL), riduci lo score del 50%
        supertrend_signal = signals.get("supertrend", 0)
        if supertrend_signal == -1.0 and raw_score > 0:
            logger.info(f"⚠️ {asset}: Score {raw_score:.3f} ridotto 50% per LONG contro supertrend SHORT")
            raw_score = raw_score * 0.5
        elif supertrend_signal == 1.0 and raw_score < 0:
            logger.info(f"⚠️ {asset}: Score {raw_score:.3f} ridotto 50% per SHORT contro supertrend LONG")
            raw_score = raw_score * 0.5

        # Apply FNG filter
        final_score = self._apply_fng_filter(raw_score, fng_value)

        # Calculate confidence
        base_confidence = self._calculate_base_confidence(signals, final_score)
        confidence_multiplier = self._get_volatility_confidence_multiplier(atr_ratio)
        confidence_after_volatility = base_confidence * confidence_multiplier

        # Apply RSI timing adjustment for entry confirmation
        # Supertrend 4h gives direction, RSI confirms entry timing
        rsi_raw = market_data.get("intraday", {}).get("rsi14")
        rsi_timing_boost = self._calculate_rsi_timing_boost(rsi_raw, final_score)
        final_confidence = min(1.0, confidence_after_volatility * rsi_timing_boost)  # Cap at 100%

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
            btc_trend=self.btc_supertrend if asset != "BTC" else None,
            rsi_raw=rsi_raw,
            rsi_timing_boost=rsi_timing_boost
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

        NOTE: Threshold AUMENTATI e segnali RIDOTTI.
        Il funding negativo può persistere settimane senza squeeze.
        Non deve mai essere il driver principale di un trade.
        """
        if funding_annual_pct is None:
            return 0.0

        # Threshold più alti - solo valori estremi contano
        if funding_annual_pct > 150:
            return -0.7  # Ridotto da -1.0
        elif funding_annual_pct > 80:
            return -0.3  # Ridotto da -0.5
        elif funding_annual_pct < -150:
            return 0.7   # Ridotto da 1.0
        elif funding_annual_pct < -80:
            return 0.3   # Ridotto da 0.5
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

    def _calculate_rsi_timing_boost(
        self,
        rsi: Optional[float],
        final_score: float
    ) -> float:
        """
        Calculate RSI timing boost for entry confirmation.

        Supertrend 4h gives direction, RSI confirms entry timing:
        - RSI < 35 (oversold) + positive score → good long entry → boost 1.3x
        - RSI > 65 (overbought) + negative score → good short entry → boost 1.3x
        - Otherwise → suboptimal timing → reduce to 0.7x

        This helps avoid entering longs at overbought levels and
        shorts at oversold levels, improving entry timing.
        """
        if rsi is None:
            return 1.0  # No RSI data, no adjustment

        # RSI thresholds for timing
        RSI_OVERSOLD = 35
        RSI_OVERBOUGHT = 65
        BOOST_MULTIPLIER = 1.3
        REDUCE_MULTIPLIER = 0.7

        # Good long entry: oversold + bullish score
        if rsi < RSI_OVERSOLD and final_score > 0:
            logger.debug(f"RSI timing BOOST for LONG: RSI={rsi:.1f} < {RSI_OVERSOLD}, score={final_score:.3f} > 0")
            return BOOST_MULTIPLIER

        # Good short entry: overbought + bearish score
        if rsi > RSI_OVERBOUGHT and final_score < 0:
            logger.debug(f"RSI timing BOOST for SHORT: RSI={rsi:.1f} > {RSI_OVERBOUGHT}, score={final_score:.3f} < 0")
            return BOOST_MULTIPLIER

        # Suboptimal timing: wait for better entry
        logger.debug(f"RSI timing REDUCE: RSI={rsi:.1f}, score={final_score:.3f} - waiting for better timing")
        return REDUCE_MULTIPLIER

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

    def _get_volatility_tpsl_multiplier(self, atr_pct: float) -> float:
        """
        Get TP/SL multiplier based on asset volatility.

        Higher volatility assets (like ETH) get wider TP/SL to avoid
        being stopped out by normal price noise.

        Args:
            atr_pct: ATR as percentage of price (atr/price * 100)

        Returns:
            Multiplier for TP/SL distances (1.0 = no change)
        """
        if atr_pct > 3.0:      # Very high volatility (ETH-like)
            return 1.5         # 50% wider TP/SL
        elif atr_pct > 2.0:    # High volatility
            return 1.25        # 25% wider
        elif atr_pct > 1.0:    # Medium volatility
            return 1.0         # Default
        else:                  # Low volatility (BTC-like)
            return 0.85        # 15% tighter

    def _calculate_dynamic_tpsl(
        self,
        entry_price: float,
        atr: float,
        final_score: float,
        is_long: bool
    ) -> Tuple[float, float]:
        """
        Calculate dynamic TP/SL based on ATR, score magnitude, and volatility.

        Score forte → TP più aggressivo, SL più stretto
        Score debole → TP conservativo, SL più largo

        High volatility assets get wider TP/SL to handle price noise.
        """
        score_magnitude = abs(final_score)

        # Calculate ATR as percentage of price
        atr_pct = (atr / entry_price) * 100 if entry_price > 0 else 2.0

        # Get volatility-based multiplier
        vol_multiplier = self._get_volatility_tpsl_multiplier(atr_pct)

        logger.debug(f"TPSL calc: ATR%={atr_pct:.2f}%, vol_mult={vol_multiplier:.2f}")

        # Base multipliers (adjusted by score)
        # SL: 2.0 ATR at low score → 1.2 ATR at high score
        base_sl_multiplier = 2.0 - (score_magnitude * 0.8)

        # TP: 1.5 ATR at low score → 2.5 ATR at high score
        base_tp_multiplier = 1.5 + (score_magnitude * 1.0)

        # Apply volatility adjustment
        sl_multiplier = base_sl_multiplier * vol_multiplier
        tp_multiplier = base_tp_multiplier * vol_multiplier

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
