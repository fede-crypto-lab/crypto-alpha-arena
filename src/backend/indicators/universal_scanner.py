"""
Universal Market Scanner - Market-wide opportunity scanning with multi-source data.

Combines:
- CoinGecko: Market-wide data (price, volume, market cap, % changes)
- TAAPI: Technical indicators (RSI, MACD, EMA)
- Exchange: Availability check and funding data
"""

import asyncio
import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field

from .coingecko_client import get_coingecko_client, CoinMarketData

logger = logging.getLogger(__name__)


@dataclass
class UniversalOpportunity:
    """Market opportunity with multi-source data."""
    # Basic info
    symbol: str
    name: str
    price: float
    market_cap: float
    market_cap_rank: int

    # Price changes (from CoinGecko)
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    volume_24h: float
    ath_change_pct: float

    # Exchange data (if available)
    exchange_available: bool = False
    exchange_symbol: str = ""  # Symbol on exchange (e.g., "BTC" -> "BTC")
    funding_rate: float = 0.0
    open_interest: float = 0.0

    # Technical indicators (from TAAPI if available)
    rsi: Optional[float] = None
    macd_histogram: Optional[float] = None
    ema_trend: Optional[str] = None  # "BULLISH", "BEARISH", "NEUTRAL"
    supertrend: Optional[str] = None  # "LONG", "SHORT" from TAAPI 4h supertrend

    # Scoring
    score: float = 0.0
    signal: str = "NEUTRAL"  # "LONG", "SHORT", "NEUTRAL"
    confidence: float = 0.5  # 0.5 to 1.0 based on signal concordance
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price": self.price,
            "market_cap": self.market_cap,
            "market_cap_rank": self.market_cap_rank,
            "price_change_1h": self.price_change_1h,
            "price_change_24h": self.price_change_24h,
            "price_change_7d": self.price_change_7d,
            "volume_24h": self.volume_24h,
            "ath_change_pct": self.ath_change_pct,
            "exchange_available": self.exchange_available,
            "exchange_symbol": self.exchange_symbol,
            "funding_rate": self.funding_rate,
            "funding_annualized": self.funding_rate * 24 * 365 * 100 if self.funding_rate else 0,
            "open_interest": self.open_interest,
            "rsi": self.rsi,
            "macd_histogram": self.macd_histogram,
            "ema_trend": self.ema_trend,
            "supertrend": self.supertrend,
            "score": round(self.score, 2),
            "signal": self.signal,
            "confidence": round(self.confidence, 2),
            "reasons": self.reasons,
        }


class UniversalScanner:
    """
    Universal market scanner using multiple data sources.

    Data flow:
    1. CoinGecko -> Top coins by market cap, price changes
    2. Exchange (Hyperliquid) -> Check availability, get funding
    3. TAAPI -> Technical indicators (optional, rate limited)
    4. Score and rank opportunities using weighted scoring (like WeightedScorer)
    """

    # Core coins excluded from dynamic opportunities
    DEFAULT_CORE_COINS = ["BTC", "ETH", "SOL", "DOGE", "AVAX"]

    # Minimum thresholds for scanning
    MIN_MARKET_CAP = 10_000_000  # $10M min market cap
    MIN_VOLUME_24H = 1_000_000   # $1M min daily volume

    # Weighted scoring - ALIGNED with WeightedScorer (bot_engine uses this)
    # NOTE: funding ridotto da 0.25 a 0.12 - può persistere settimane senza squeeze
    # supertrend aggiunto a 0.25 - trend è il filtro principale
    BASE_WEIGHTS = {
        "funding": 0.12,       # REDUCED from 0.25 - aligned with WeightedScorer
        "supertrend": 0.25,    # NEW - macro trend filter (critical!)
        "momentum_1h": 0.13,   # Reduced - short-term trend
        "momentum_24h": 0.10,  # Reduced - medium-term trend
        "volume": 0.08,        # From CoinGecko - liquidity
        "rsi": 0.15,           # From TAAPI (when available)
        "btc_correlation": 0.10,  # BTC trend alignment for altcoins
        "ath_discount": 0.07,  # Value opportunity
    }

    def __init__(
        self,
        exchange_api=None,
        taapi_client=None,
        core_coins: List[str] = None,
        use_taapi: bool = True
    ):
        """
        Initialize universal scanner.

        Args:
            exchange_api: Exchange API (e.g., HyperliquidAPI) for availability check
            taapi_client: TAAPI client for technical indicators
            core_coins: List of core coins to track separately
            use_taapi: Whether to fetch TAAPI indicators (rate limited)
        """
        self.coingecko = get_coingecko_client()
        self.exchange_api = exchange_api
        self.taapi_client = taapi_client
        self.core_coins = set(core_coins or self.DEFAULT_CORE_COINS)
        self.use_taapi = use_taapi

        # Cache for exchange symbols
        self._exchange_symbols: Set[str] = set()
        self._exchange_data: Dict[str, Dict] = {}
        self._last_scan_results: List[UniversalOpportunity] = []

        # BTC trend for correlation (updated during scan)
        self._btc_momentum_signal: float = 0.0

    def set_core_coins(self, coins: List[str]):
        """Update core coins list."""
        self.core_coins = set(coins)
        logger.info(f"Core coins updated: {self.core_coins}")

    # ===== SIGNAL NORMALIZATION (like WeightedScorer) =====

    def _normalize_funding(self, funding_rate: float) -> float:
        """
        Normalize funding rate to signal (-1 to +1).
        Negative funding = expensive to short = bullish for longs
        Positive funding = expensive to long = bearish
        """
        if not funding_rate:
            return 0.0

        # Convert to annualized percentage
        funding_annual = funding_rate * 24 * 365 * 100

        if funding_annual < -100:
            return 1.0   # Very negative = strong long signal
        elif funding_annual < -50:
            return 0.5
        elif funding_annual > 100:
            return -1.0  # Very positive = strong short signal
        elif funding_annual > 50:
            return -0.5
        else:
            return 0.0

    def _normalize_momentum(self, pct_change: float, threshold: float = 5.0) -> float:
        """
        Normalize price change to signal (-1 to +1).
        Uses threshold for scaling.
        """
        if pct_change is None:
            return 0.0

        # Scale: ±threshold% = ±0.5, ±2*threshold% = ±1.0
        normalized = pct_change / (threshold * 2)
        return max(-1.0, min(1.0, normalized))

    def _normalize_volume(self, volume_24h: float) -> float:
        """
        Normalize volume to signal (0 to 1).
        High volume = more reliable signals.
        """
        if not volume_24h:
            return 0.0

        # $100M+ = 1.0, $10M = 0.5, $1M = 0.0
        if volume_24h >= 100_000_000:
            return 1.0
        elif volume_24h >= 10_000_000:
            return 0.5 + 0.5 * ((volume_24h - 10_000_000) / 90_000_000)
        else:
            return max(0.0, volume_24h / 20_000_000)

    def _normalize_rsi(self, rsi: Optional[float]) -> float:
        """
        Normalize RSI to signal (-1 to +1).
        Oversold (<30) = bullish, Overbought (>70) = bearish.
        """
        if rsi is None:
            return 0.0

        if rsi < 30:
            return 1.0   # Oversold = long signal
        elif rsi < 40:
            return 0.5
        elif rsi > 70:
            return -1.0  # Overbought = short signal
        elif rsi > 60:
            return -0.5
        else:
            return 0.0

    def _normalize_ath_discount(self, ath_change_pct: float) -> float:
        """
        Normalize ATH discount to signal (0 to 1).
        Bigger discount = potential value opportunity.
        """
        if ath_change_pct is None or ath_change_pct >= 0:
            return 0.0

        # -50% = 0.3, -70% = 0.6, -90% = 1.0
        discount = abs(ath_change_pct)
        if discount >= 90:
            return 1.0
        elif discount >= 70:
            return 0.6 + 0.4 * ((discount - 70) / 20)
        elif discount >= 50:
            return 0.3 + 0.3 * ((discount - 50) / 20)
        else:
            return max(0.0, discount / 166)

    def _normalize_supertrend(self, supertrend_data: Optional[Dict]) -> float:
        """
        Normalize supertrend to signal (-1 to +1).
        LONG advice = bullish = +1
        SHORT advice = bearish = -1
        """
        if not supertrend_data:
            return 0.0

        advice = supertrend_data.get("advice", "").lower()
        if advice == "long":
            return 1.0
        elif advice == "short":
            return -1.0
        else:
            return 0.0

    async def _fetch_exchange_symbols(self) -> Set[str]:
        """Fetch available symbols from connected exchange."""
        if not self.exchange_api:
            return set()

        try:
            # For Hyperliquid
            if hasattr(self.exchange_api, 'get_meta_and_ctxs'):
                response = await self.exchange_api.get_meta_and_ctxs()
                if isinstance(response, list) and len(response) >= 2:
                    meta = response[0]
                    universe = meta.get("universe", [])
                    symbols = {a.get("name", "").upper() for a in universe}
                    symbols.discard("")
                    return symbols

            # Generic fallback
            if hasattr(self.exchange_api, 'get_all_symbols'):
                return set(await self.exchange_api.get_all_symbols())

        except Exception as e:
            logger.error(f"Error fetching exchange symbols: {e}")

        return set()

    async def _fetch_exchange_data(self) -> Dict[str, Dict]:
        """Fetch market data from exchange (funding, OI, etc.)."""
        if not self.exchange_api:
            return {}

        try:
            if hasattr(self.exchange_api, 'get_meta_and_ctxs'):
                response = await self.exchange_api.get_meta_and_ctxs()
                if isinstance(response, list) and len(response) >= 2:
                    meta, asset_ctxs = response[0], response[1]
                    universe = meta.get("universe", [])

                    data = {}
                    for i, asset_info in enumerate(universe):
                        symbol = asset_info.get("name", "").upper()
                        if not symbol:
                            continue
                        ctx = asset_ctxs[i] if i < len(asset_ctxs) else {}
                        data[symbol] = {
                            "funding_rate": float(ctx.get("funding", 0) or 0),
                            "open_interest": float(ctx.get("openInterest", 0) or 0),
                            "volume_24h": float(ctx.get("dayNtlVlm", 0) or 0),
                        }
                    return data
        except Exception as e:
            logger.error(f"Error fetching exchange data: {e}")

        return {}

    async def _fetch_taapi_indicators(self, symbol: str, include_supertrend: bool = False) -> Dict:
        """
        Fetch technical indicators from TAAPI.

        Args:
            symbol: Coin symbol (e.g., "BTC")
            include_supertrend: If True, fetch supertrend 4h (for trend filter)
        """
        if not self.taapi_client or not self.use_taapi:
            logger.debug(f"TAAPI fetch skipped: taapi_client={self.taapi_client is not None}, use_taapi={self.use_taapi}")
            return {}

        try:
            indicators = {}
            taapi_symbol = f"{symbol}/USDT"

            # Fetch supertrend 4h (critical for trend filter)
            if include_supertrend:
                try:
                    # Use bulk endpoint to fetch supertrend
                    supertrend_config = [
                        {"id": "supertrend", "indicator": "supertrend", "period": 10, "multiplier": 3}
                    ]
                    # TAAPIClient methods are sync, run in thread pool
                    bulk_result = await asyncio.to_thread(
                        self.taapi_client.fetch_bulk_indicators,
                        taapi_symbol,
                        "4h",
                        supertrend_config
                    )
                    if bulk_result and "supertrend" in bulk_result:
                        st_data = bulk_result["supertrend"]
                        if isinstance(st_data, dict):
                            # Extract advice field (valueAdvice contains "long" or "short")
                            advice = st_data.get("valueAdvice", "")
                            indicators["supertrend"] = {
                                "value": st_data.get("valueSupertrend"),
                                "advice": advice  # "long" or "short"
                            }
                            logger.info(f"📊 {symbol} supertrend 4h: {advice.upper()}")
                except Exception as e:
                    logger.debug(f"Supertrend fetch failed for {symbol}: {e}")

            # Optionally fetch RSI/MACD for high-momentum coins
            # (commented out to save rate limits - supertrend is the priority)
            # rsi_config = [{"id": "rsi", "indicator": "rsi", "period": 14}]
            # bulk_1h = await asyncio.to_thread(
            #     self.taapi_client.fetch_bulk_indicators, taapi_symbol, "1h", rsi_config
            # )
            # if bulk_1h and "rsi" in bulk_1h:
            #     indicators["rsi"] = bulk_1h["rsi"].get("value")

            return indicators

        except Exception as e:
            logger.debug(f"TAAPI error for {symbol}: {e}")
            return {}

    def _calculate_score(
        self,
        coin: CoinMarketData,
        exchange_data: Dict,
        taapi_data: Dict,
        is_exchange_available: bool
    ) -> tuple[float, str, List[str], float]:
        """
        Calculate opportunity score using weighted signals (ALIGNED with WeightedScorer).

        Returns:
            Tuple of (score_points, signal, reasons, confidence)
        """
        signals = {}
        reasons = []

        # === EXTRACT NORMALIZED SIGNALS ===

        # 1. Funding signal (from exchange) - REDUCED weight, thresholds aligned with scorer
        funding_rate = exchange_data.get("funding_rate", 0)
        signals["funding"] = self._normalize_funding(funding_rate)
        if signals["funding"] != 0:
            funding_apr = funding_rate * 24 * 365 * 100
            reasons.append(f"Funding {funding_apr:+.0f}% APR")

        # 2. Supertrend signal (from TAAPI) - NEW: critical trend filter
        supertrend_data = taapi_data.get("supertrend")
        signals["supertrend"] = self._normalize_supertrend(supertrend_data)
        if signals["supertrend"] != 0:
            advice = supertrend_data.get("advice", "").upper() if supertrend_data else "N/A"
            reasons.append(f"ST:{advice}")

        # 3. 1h momentum (from CoinGecko)
        signals["momentum_1h"] = self._normalize_momentum(coin.price_change_1h, threshold=3.0)
        if abs(coin.price_change_1h) > 2:
            reasons.append(f"1h {coin.price_change_1h:+.1f}%")

        # 4. 24h momentum (from CoinGecko)
        signals["momentum_24h"] = self._normalize_momentum(coin.price_change_24h, threshold=8.0)
        if abs(coin.price_change_24h) > 5:
            reasons.append(f"24h {coin.price_change_24h:+.1f}%")

        # 5. Volume signal
        signals["volume"] = self._normalize_volume(coin.volume_24h)
        if coin.volume_24h > 50_000_000:
            reasons.append(f"Vol ${coin.volume_24h/1e6:.0f}M")

        # 6. RSI signal (from TAAPI if available)
        rsi = taapi_data.get("rsi")
        signals["rsi"] = self._normalize_rsi(rsi)
        if rsi is not None and (rsi < 35 or rsi > 65):
            reasons.append(f"RSI {rsi:.0f}")

        # 7. BTC correlation (for altcoins only)
        if coin.symbol != "BTC":
            signals["btc_correlation"] = self._btc_momentum_signal
        else:
            signals["btc_correlation"] = 0.0

        # 8. ATH discount
        signals["ath_discount"] = self._normalize_ath_discount(coin.ath_change_pct)
        if coin.ath_change_pct < -60:
            reasons.append(f"{abs(coin.ath_change_pct):.0f}% < ATH")

        # === CALCULATE WEIGHTED SCORE ===
        weights = self.BASE_WEIGHTS.copy()

        # For BTC, remove btc_correlation weight and redistribute
        if coin.symbol == "BTC":
            weights["btc_correlation"] = 0.0

        # If no supertrend data, remove its weight (don't penalize for missing data)
        if signals["supertrend"] == 0 and not supertrend_data:
            weights["supertrend"] = 0.0

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate weighted sum
        raw_score = 0.0
        for key, weight in weights.items():
            signal_value = signals.get(key, 0.0)
            raw_score += signal_value * weight

        # Clamp to [-1, 1]
        raw_score = max(-1.0, min(1.0, raw_score))

        # === SUPERTREND PENALTY: Penalize counter-trend trades (like WeightedScorer) ===
        supertrend_signal = signals.get("supertrend", 0)
        if supertrend_signal == -1.0 and raw_score > 0:
            logger.info(f"⚠️ SCANNER {coin.symbol}: Score {raw_score:.3f} ridotto 50% per LONG contro supertrend SHORT")
            raw_score = raw_score * 0.5
            reasons.append("⚠️ Counter-trend")
        elif supertrend_signal == 1.0 and raw_score < 0:
            logger.info(f"⚠️ SCANNER {coin.symbol}: Score {raw_score:.3f} ridotto 50% per SHORT contro supertrend LONG")
            raw_score = raw_score * 0.5
            reasons.append("⚠️ Counter-trend")

        # === DETERMINE SIGNAL DIRECTION ===
        if raw_score > 0.15:
            signal = "LONG"
        elif raw_score < -0.15:
            signal = "SHORT"
        else:
            signal = "NEUTRAL"

        # === CALCULATE CONFIDENCE (signal concordance) ===
        confidence = self._calculate_confidence(signals, raw_score)

        # === CONVERT TO POINTS (for backward compatibility) ===
        # Map [-1, 1] score to [0, 100] points range
        # Strong signals (>0.5 or <-0.5) should get 70+ points
        score_points = 50 + (raw_score * 50)  # Base 50, ±50 based on score

        # Boost for high confidence
        if confidence > 0.7:
            score_points *= 1.2

        # Boost for exchange availability
        if is_exchange_available:
            score_points += 10
            if "Tradeable" not in str(reasons):
                reasons.append("Tradeable")

        # Core coin bonus
        if coin.symbol in self.core_coins:
            score_points += 10
            reasons.append("Core")

        # Market cap rank bonus
        if coin.market_cap_rank <= 20:
            score_points += 5

        return score_points, signal, reasons, confidence

    def _calculate_confidence(self, signals: Dict[str, float], raw_score: float) -> float:
        """
        Calculate confidence based on signal concordance.
        Returns 0.5 to 1.0 based on how many signals agree.
        """
        if raw_score == 0:
            return 0.5

        score_direction = 1 if raw_score > 0 else -1
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

    async def scan_market(
        self,
        top_n: int = 100,
        max_results: int = 20,
        include_non_tradeable: bool = True
    ) -> List[UniversalOpportunity]:
        """
        Scan the market for trading opportunities.

        Args:
            top_n: Number of top coins to scan from CoinGecko
            max_results: Maximum opportunities to return
            include_non_tradeable: Include coins not available on exchange

        Returns:
            List of UniversalOpportunity sorted by score
        """
        logger.info(f"Starting universal market scan (top {top_n} coins, use_taapi={self.use_taapi}, taapi_client={self.taapi_client is not None})...")

        # Fetch data from all sources in parallel
        coingecko_task = self.coingecko.get_top_coins(limit=top_n)
        exchange_symbols_task = self._fetch_exchange_symbols()
        exchange_data_task = self._fetch_exchange_data()

        coins, exchange_symbols, exchange_data = await asyncio.gather(
            coingecko_task,
            exchange_symbols_task,
            exchange_data_task
        )

        self._exchange_symbols = exchange_symbols
        self._exchange_data = exchange_data

        logger.info(f"Fetched {len(coins)} coins from CoinGecko, {len(exchange_symbols)} on exchange")

        # Extract BTC momentum for altcoin correlation
        self._btc_momentum_signal = 0.0
        for coin in coins:
            if coin.symbol == "BTC":
                # Use 1h and 24h momentum to determine BTC trend
                btc_1h = self._normalize_momentum(coin.price_change_1h, threshold=3.0)
                btc_24h = self._normalize_momentum(coin.price_change_24h, threshold=8.0)
                # Weight: 60% 1h, 40% 24h
                self._btc_momentum_signal = btc_1h * 0.6 + btc_24h * 0.4
                logger.info(f"BTC correlation signal: {self._btc_momentum_signal:+.2f} (1h={coin.price_change_1h:+.1f}%, 24h={coin.price_change_24h:+.1f}%)")
                break

        # === PASS 1: Calculate preliminary scores WITHOUT TAAPI (fast) ===
        preliminary_results = []

        for coin in coins:
            # Skip low market cap
            if coin.market_cap < self.MIN_MARKET_CAP:
                continue

            # Skip low volume
            if coin.volume_24h < self.MIN_VOLUME_24H:
                continue

            # Check exchange availability
            is_available = coin.symbol in exchange_symbols
            exch_data = exchange_data.get(coin.symbol, {})

            if not include_non_tradeable and not is_available:
                continue

            # Calculate preliminary score WITHOUT TAAPI data
            prelim_score, signal, reasons, confidence = self._calculate_score(
                coin, exch_data, {}, is_available
            )

            preliminary_results.append({
                'coin': coin,
                'is_available': is_available,
                'exch_data': exch_data,
                'prelim_score': prelim_score,
                'signal': signal,
                'reasons': reasons,
                'confidence': confidence,
            })

        # Sort by preliminary score (absolute value for both long/short)
        preliminary_results.sort(key=lambda x: abs(x['prelim_score']), reverse=True)

        # === PASS 2: Fetch TAAPI supertrend ONLY for top N tradeable candidates ===
        MAX_TAAPI_CALLS = 15  # Limit TAAPI calls to save rate limits
        taapi_fetch_count = 0

        opportunities = []

        for item in preliminary_results:
            coin = item['coin']
            is_available = item['is_available']
            exch_data = item['exch_data']

            # Fetch TAAPI only for top tradeable coins
            taapi_data = {}
            should_fetch_taapi = (
                is_available and
                self.use_taapi and
                self.taapi_client and
                taapi_fetch_count < MAX_TAAPI_CALLS
            )

            if should_fetch_taapi:
                logger.debug(f"Fetching TAAPI supertrend for {coin.symbol} (#{taapi_fetch_count + 1})")
                taapi_data = await self._fetch_taapi_indicators(
                    coin.symbol,
                    include_supertrend=True
                )
                taapi_fetch_count += 1

            # Recalculate final score WITH TAAPI data (if available)
            if taapi_data:
                score, signal, reasons, confidence = self._calculate_score(
                    coin, exch_data, taapi_data, is_available
                )
            else:
                # Use preliminary results
                score = item['prelim_score']
                signal = item['signal']
                reasons = item['reasons']
                confidence = item['confidence']

            # Extract supertrend advice for display
            supertrend_data = taapi_data.get("supertrend")
            supertrend_advice = supertrend_data.get("advice", "").upper() if supertrend_data else None

            opp = UniversalOpportunity(
                symbol=coin.symbol,
                name=coin.name,
                price=coin.price,
                market_cap=coin.market_cap,
                market_cap_rank=coin.market_cap_rank,
                price_change_1h=coin.price_change_1h,
                price_change_24h=coin.price_change_24h,
                price_change_7d=coin.price_change_7d,
                volume_24h=coin.volume_24h,
                ath_change_pct=coin.ath_change_pct,
                exchange_available=is_available,
                exchange_symbol=coin.symbol if is_available else "",
                funding_rate=exch_data.get("funding_rate", 0),
                open_interest=exch_data.get("open_interest", 0),
                rsi=taapi_data.get("rsi"),
                macd_histogram=taapi_data.get("macd_histogram"),
                supertrend=supertrend_advice,
                score=score,
                signal=signal,
                confidence=confidence,
                reasons=reasons,
            )
            opportunities.append(opp)

        logger.info(f"TAAPI calls: {taapi_fetch_count} (max {MAX_TAAPI_CALLS})")

        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)

        # Limit results
        opportunities = opportunities[:max_results]

        self._last_scan_results = opportunities

        # Log top results
        logger.info(f"Scan complete: {len(opportunities)} opportunities (weighted scoring)")
        tradeable = len([o for o in opportunities if o.exchange_available])
        logger.info(f"  Tradeable on exchange: {tradeable}")
        for opp in opportunities[:5]:
            avail = "✓" if opp.exchange_available else "✗"
            conf_pct = int(opp.confidence * 100)
            logger.info(f"  {avail} {opp.symbol}: score={opp.score:.0f} signal={opp.signal} conf={conf_pct}%")

        return opportunities

    def get_last_scan(self) -> List[Dict]:
        """Get last scan results as list of dicts."""
        return [o.to_dict() for o in self._last_scan_results]

    def get_trading_opportunities(
        self,
        exclude_symbols: List[str] = None,
        min_score: float = 25,
        only_tradeable: bool = True
    ) -> List[Dict]:
        """
        Get actionable trading opportunities from last scan.

        Args:
            exclude_symbols: Symbols to exclude (core coins, open positions)
            min_score: Minimum score threshold
            only_tradeable: Only return coins available on exchange

        Returns:
            List of tradeable opportunities
        """
        exclude = set(exclude_symbols or [])
        opportunities = []

        for opp in self._last_scan_results:
            # Skip excluded
            if opp.symbol in exclude:
                continue

            # Skip non-tradeable
            if only_tradeable and not opp.exchange_available:
                continue

            # Skip low score
            if opp.score < min_score:
                continue

            # Skip neutral signals
            if opp.signal == "NEUTRAL":
                continue

            opportunities.append(opp.to_dict())

        return opportunities

    def is_available_on_exchange(self, symbol: str) -> bool:
        """Check if a symbol is available on the connected exchange."""
        return symbol.upper() in self._exchange_symbols


# Singleton instance
_scanner: Optional[UniversalScanner] = None


def get_universal_scanner(
    exchange_api=None,
    taapi_client=None,
    core_coins: List[str] = None
) -> UniversalScanner:
    """Get or create universal scanner instance."""
    global _scanner
    if _scanner is None:
        _scanner = UniversalScanner(
            exchange_api=exchange_api,
            taapi_client=taapi_client,
            core_coins=core_coins
        )
    else:
        # Update components if provided and not already set
        if exchange_api and not _scanner.exchange_api:
            _scanner.exchange_api = exchange_api
        if taapi_client and not _scanner.taapi_client:
            _scanner.taapi_client = taapi_client
            _scanner.use_taapi = True  # Enable TAAPI usage when client is set
            logger.info("Scanner: TAAPI client injected, supertrend fetching enabled")

    return _scanner
