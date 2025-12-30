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

    # Scoring
    score: float = 0.0
    signal: str = "NEUTRAL"  # "LONG", "SHORT", "NEUTRAL"
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
            "score": round(self.score, 2),
            "signal": self.signal,
            "reasons": self.reasons,
        }


class UniversalScanner:
    """
    Universal market scanner using multiple data sources.

    Data flow:
    1. CoinGecko -> Top coins by market cap, price changes
    2. Exchange (Hyperliquid) -> Check availability, get funding
    3. TAAPI -> Technical indicators (optional, rate limited)
    4. Score and rank opportunities
    """

    # Core coins excluded from dynamic opportunities
    DEFAULT_CORE_COINS = ["BTC", "ETH", "SOL", "DOGE", "AVAX"]

    # Minimum thresholds for scanning
    MIN_MARKET_CAP = 10_000_000  # $10M min market cap
    MIN_VOLUME_24H = 1_000_000   # $1M min daily volume

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

    def set_core_coins(self, coins: List[str]):
        """Update core coins list."""
        self.core_coins = set(coins)
        logger.info(f"Core coins updated: {self.core_coins}")

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

    async def _fetch_taapi_indicators(self, symbol: str) -> Dict:
        """Fetch technical indicators from TAAPI."""
        if not self.taapi_client or not self.use_taapi:
            return {}

        try:
            # Use bulk endpoint if available
            if hasattr(self.taapi_client, 'get_bulk_indicators'):
                return await self.taapi_client.get_bulk_indicators(
                    symbol=f"{symbol}/USDT",
                    exchange="binance",
                    interval="1h",
                    indicators=["rsi", "macd", "ema"]
                )

            # Fallback to individual calls (rate limited!)
            indicators = {}
            if hasattr(self.taapi_client, 'get_indicator'):
                rsi = await self.taapi_client.get_indicator("rsi", f"{symbol}/USDT", "1h")
                if rsi:
                    indicators["rsi"] = rsi.get("value")

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
    ) -> tuple[float, str, List[str]]:
        """
        Calculate opportunity score based on multiple factors.

        Returns:
            Tuple of (score, signal, reasons)
        """
        score = 0.0
        reasons = []
        signal = "NEUTRAL"

        # === MOMENTUM SIGNALS (from CoinGecko) ===

        # 1h momentum
        if abs(coin.price_change_1h) > 3:
            score += 20
            direction = "up" if coin.price_change_1h > 0 else "down"
            reasons.append(f"1h momentum {coin.price_change_1h:+.1f}%")
            if signal == "NEUTRAL":
                signal = "LONG" if coin.price_change_1h > 0 else "SHORT"

        # 24h momentum
        if abs(coin.price_change_24h) > 8:
            score += 15
            reasons.append(f"24h move {coin.price_change_24h:+.1f}%")
            if signal == "NEUTRAL":
                signal = "LONG" if coin.price_change_24h > 0 else "SHORT"

        # 7d trend
        if abs(coin.price_change_7d) > 15:
            score += 10
            reasons.append(f"7d trend {coin.price_change_7d:+.1f}%")

        # === VOLUME SIGNALS ===
        if coin.volume_24h > 100_000_000:
            score += 20
            reasons.append(f"High volume ${coin.volume_24h/1e6:.0f}M")
        elif coin.volume_24h > 10_000_000:
            score += 10
            reasons.append(f"Good volume ${coin.volume_24h/1e6:.0f}M")

        # === ATH DISCOUNT ===
        if coin.ath_change_pct < -70:
            score += 15
            reasons.append(f"{abs(coin.ath_change_pct):.0f}% below ATH")
        elif coin.ath_change_pct < -50:
            score += 8

        # === FUNDING RATE (from exchange) ===
        funding_rate = exchange_data.get("funding_rate", 0)
        if funding_rate:
            funding_apr = funding_rate * 24 * 365 * 100

            if funding_rate < -0.0001:  # Negative funding = long opportunity
                score += 25
                reasons.append(f"Negative funding {funding_apr:.1f}% APR")
                signal = "LONG"
            elif funding_rate > 0.0003:  # High positive = short opportunity
                score += 20
                reasons.append(f"High funding {funding_apr:.1f}% APR")
                signal = "SHORT"

        # === OPEN INTEREST ===
        oi = exchange_data.get("open_interest", 0)
        if oi > 50_000_000:
            score += 10
            reasons.append(f"High OI ${oi/1e6:.0f}M")

        # === TECHNICAL INDICATORS (from TAAPI) ===
        rsi = taapi_data.get("rsi")
        if rsi is not None:
            if rsi < 30:
                score += 20
                reasons.append(f"RSI oversold ({rsi:.0f})")
                if signal == "NEUTRAL":
                    signal = "LONG"
            elif rsi > 70:
                score += 15
                reasons.append(f"RSI overbought ({rsi:.0f})")
                if signal == "NEUTRAL":
                    signal = "SHORT"

        macd = taapi_data.get("macd_histogram") or taapi_data.get("valueHist")
        if macd is not None:
            if macd > 0:
                score += 5
                reasons.append("MACD bullish")
            else:
                score += 5
                reasons.append("MACD bearish")

        # === MARKET CAP RANK BONUS ===
        if coin.market_cap_rank <= 20:
            score += 10
            reasons.append(f"Top {coin.market_cap_rank} by mcap")
        elif coin.market_cap_rank <= 50:
            score += 5

        # === EXCHANGE AVAILABILITY BONUS ===
        if is_exchange_available:
            score += 15
            reasons.append("Tradeable on exchange")

        # === CORE COIN BONUS ===
        if coin.symbol in self.core_coins:
            score += 10
            reasons.append("Core asset")

        return score, signal, reasons

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
        logger.info(f"Starting universal market scan (top {top_n} coins)...")

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

        opportunities = []

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

            # Get TAAPI indicators (only for available coins to save rate limits)
            taapi_data = {}
            if is_available and self.use_taapi:
                # Only fetch for high potential coins to save rate limits
                if coin.price_change_1h > 2 or coin.price_change_1h < -2:
                    taapi_data = await self._fetch_taapi_indicators(coin.symbol)

            # Calculate score
            score, signal, reasons = self._calculate_score(
                coin, exch_data, taapi_data, is_available
            )

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
                score=score,
                signal=signal,
                reasons=reasons,
            )
            opportunities.append(opp)

        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)

        # Limit results
        opportunities = opportunities[:max_results]

        self._last_scan_results = opportunities

        # Log top results
        logger.info(f"Scan complete: {len(opportunities)} opportunities")
        tradeable = len([o for o in opportunities if o.exchange_available])
        logger.info(f"  Tradeable on exchange: {tradeable}")
        for opp in opportunities[:5]:
            avail = "✓" if opp.exchange_available else "✗"
            logger.info(f"  {avail} {opp.symbol}: score={opp.score:.0f} signal={opp.signal}")

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
    elif exchange_api and not _scanner.exchange_api:
        _scanner.exchange_api = exchange_api
    elif taapi_client and not _scanner.taapi_client:
        _scanner.taapi_client = taapi_client

    return _scanner
