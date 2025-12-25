"""
Market Scanner - Dynamic coin selection based on momentum and opportunity signals.

Exchange-agnostic scanner that ranks coins by trading opportunity score.
Supports any exchange via the ExchangeDataProvider interface.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ===== EXCHANGE-AGNOSTIC INTERFACE =====

class ExchangeDataProvider(Protocol):
    """
    Protocol for exchange data providers.
    Implement this interface to add support for new exchanges.
    """

    async def get_all_symbols(self) -> List[str]:
        """Return all tradeable symbol names."""
        ...

    async def get_market_data(self, symbol: str) -> Dict:
        """
        Get market data for a symbol.

        Expected keys:
            - price: float
            - volume_24h: float
            - open_interest: float (optional)
            - funding_rate: float (optional, for perps)
            - price_change_1h: float (percentage)
            - price_change_24h: float (percentage)
        """
        ...

    async def get_all_market_data(self) -> List[Dict]:
        """
        Get market data for all symbols in one call (more efficient).

        Each dict should have 'symbol' key plus same keys as get_market_data.
        """
        ...


@dataclass
class CoinOpportunity:
    """Represents a trading opportunity for a coin."""
    symbol: str
    price: float
    volume_24h: float
    open_interest: float
    funding_rate: float
    price_change_1h: float
    price_change_24h: float
    score: float
    signal: str  # "LONG", "SHORT", "NEUTRAL"
    reasons: List[str]

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "volume_24h": self.volume_24h,
            "open_interest": self.open_interest,
            "funding_rate": self.funding_rate,
            "funding_annualized": self.funding_rate * 24 * 365 * 100 if self.funding_rate else 0,
            "price_change_1h": self.price_change_1h,
            "price_change_24h": self.price_change_24h,
            "score": round(self.score, 2),
            "signal": self.signal,
            "reasons": self.reasons
        }


# ===== HYPERLIQUID PROVIDER =====

class HyperliquidDataProvider:
    """Exchange data provider for Hyperliquid."""

    EXCLUDED = {"USDC", "USDT", "DAI", "USDP", "TUSD", "FRAX"}

    def __init__(self, api):
        """
        Args:
            api: HyperliquidAPI instance
        """
        self.api = api
        self._cache = {}

    async def get_all_symbols(self) -> List[str]:
        data = await self._get_meta()
        return [a["symbol"] for a in data if a["symbol"] not in self.EXCLUDED]

    async def get_market_data(self, symbol: str) -> Dict:
        all_data = await self.get_all_market_data()
        for d in all_data:
            if d["symbol"] == symbol:
                return d
        return {}

    async def get_all_market_data(self) -> List[Dict]:
        """Get all market data from Hyperliquid in one call."""
        try:
            response = await self.api.get_meta_and_ctxs()
            if not isinstance(response, list) or len(response) < 2:
                return []

            meta, asset_ctxs = response[0], response[1]
            universe = meta.get("universe", [])

            # Get all mid prices
            mids = await self.api.info.all_mids()

            results = []
            for i, asset_info in enumerate(universe):
                symbol = asset_info.get("name")
                if not symbol or symbol in self.EXCLUDED:
                    continue

                ctx = asset_ctxs[i] if i < len(asset_ctxs) else {}

                # Get price
                price = float(mids.get(symbol, 0) or ctx.get("markPx", 0) or 0)

                results.append({
                    "symbol": symbol,
                    "price": price,
                    "volume_24h": float(ctx.get("dayNtlVlm", 0) or 0),
                    "open_interest": float(ctx.get("openInterest", 0) or 0),
                    "funding_rate": float(ctx.get("funding", 0) or 0),
                    "price_change_1h": 0,  # Would need candles for this
                    "price_change_24h": 0,  # Would need candles for this
                    "premium": float(ctx.get("premium", 0) or 0),
                })

            return results

        except Exception as e:
            logger.error(f"Error fetching Hyperliquid market data: {e}")
            return []

    async def _get_meta(self) -> List[Dict]:
        return await self.get_all_market_data()


# ===== MAIN SCANNER (EXCHANGE-AGNOSTIC) =====

class MarketScanner:
    """
    Exchange-agnostic market scanner for finding trading opportunities.

    Scoring criteria:
    - Negative funding = long opportunity (shorts paying longs)
    - Positive funding = short opportunity (longs paying shorts)
    - High volume = better liquidity
    - Price momentum (1h/24h change)
    - Open interest
    """

    # Default core coins (can be overridden)
    DEFAULT_CORE_COINS = ["BTC", "ETH", "SOL", "DOGE", "XRP"]

    # Minimum thresholds
    MIN_VOLUME_24H = 100_000  # $100k min daily volume
    MIN_OPEN_INTEREST = 50_000  # $50k min OI

    def __init__(
        self,
        data_provider: ExchangeDataProvider,
        core_coins: List[str] = None
    ):
        """
        Initialize scanner.

        Args:
            data_provider: Exchange data provider implementing ExchangeDataProvider
            core_coins: List of core coins to always include (overrides default)
        """
        self.provider = data_provider
        self.core_coins = set(core_coins or self.DEFAULT_CORE_COINS)
        self._last_scan_results: List[CoinOpportunity] = []

    def set_core_coins(self, coins: List[str]):
        """Update the core coins list."""
        self.core_coins = set(coins)
        logger.info(f"Core coins updated: {self.core_coins}")

    def calculate_opportunity_score(
        self,
        symbol: str,
        funding_rate: float,
        volume_24h: float,
        open_interest: float,
        price_change_1h: float,
        price_change_24h: float
    ) -> tuple[float, str, List[str]]:
        """
        Calculate opportunity score for a coin.

        Returns:
            Tuple of (score, signal, reasons)
        """
        score = 0.0
        reasons = []
        signal = "NEUTRAL"

        # === FUNDING RATE SIGNALS ===
        funding_annualized = funding_rate * 24 * 365 * 100

        if funding_rate < -0.0001:  # < -0.01% per 8h = negative funding
            score += 30
            reasons.append(f"Negative funding ({funding_annualized:.1f}% APR)")
            signal = "LONG"
        elif funding_rate > 0.0003:  # > 0.03% per 8h = high positive
            score += 20
            reasons.append(f"High funding ({funding_annualized:.1f}% APR)")
            signal = "SHORT"

        # === VOLUME SCORE ===
        if volume_24h > 10_000_000:
            score += 25
            reasons.append(f"High volume (${volume_24h/1e6:.1f}M)")
        elif volume_24h > 1_000_000:
            score += 15
            reasons.append(f"Good volume (${volume_24h/1e6:.1f}M)")
        elif volume_24h > 100_000:
            score += 5

        # === MOMENTUM SCORE ===
        if abs(price_change_1h) > 3:
            score += 20
            direction = "up" if price_change_1h > 0 else "down"
            reasons.append(f"1h momentum ({price_change_1h:+.1f}%)")
            if signal == "NEUTRAL":
                signal = "LONG" if price_change_1h > 0 else "SHORT"

        if abs(price_change_24h) > 10:
            score += 15
            reasons.append(f"24h move ({price_change_24h:+.1f}%)")

        # === OPEN INTEREST ===
        if open_interest > 10_000_000:
            score += 10
            reasons.append(f"High OI (${open_interest/1e6:.1f}M)")

        # === CORE COIN BONUS ===
        if symbol in self.core_coins:
            score += 15
            reasons.append("Core asset")

        return score, signal, reasons

    async def scan_market(
        self,
        max_dynamic: int = 5,
        include_core: bool = True
    ) -> List[CoinOpportunity]:
        """
        Scan the market and return top opportunities.

        Args:
            max_dynamic: Max number of dynamic (non-core) opportunities
            include_core: Whether to always include core coins

        Returns:
            List of CoinOpportunity sorted by score
        """
        logger.info("Starting market scan...")

        all_data = await self.provider.get_all_market_data()
        if not all_data:
            logger.warning("No market data from provider")
            return []

        opportunities = []

        for data in all_data:
            symbol = data.get("symbol")
            if not symbol:
                continue

            volume_24h = data.get("volume_24h", 0)
            open_interest = data.get("open_interest", 0)
            funding_rate = data.get("funding_rate", 0)

            # Skip low liquidity (unless core)
            is_core = symbol in self.core_coins
            if not is_core:
                if volume_24h < self.MIN_VOLUME_24H:
                    continue
                if open_interest < self.MIN_OPEN_INTEREST:
                    continue

            score, signal, reasons = self.calculate_opportunity_score(
                symbol=symbol,
                funding_rate=funding_rate,
                volume_24h=volume_24h,
                open_interest=open_interest,
                price_change_1h=data.get("price_change_1h", 0),
                price_change_24h=data.get("price_change_24h", 0)
            )

            opp = CoinOpportunity(
                symbol=symbol,
                price=data.get("price", 0),
                volume_24h=volume_24h,
                open_interest=open_interest,
                funding_rate=funding_rate,
                price_change_1h=data.get("price_change_1h", 0),
                price_change_24h=data.get("price_change_24h", 0),
                score=score,
                signal=signal,
                reasons=reasons
            )
            opportunities.append(opp)

        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)

        # Separate core and dynamic
        core_opps = [o for o in opportunities if o.symbol in self.core_coins]
        dynamic_opps = [o for o in opportunities if o.symbol not in self.core_coins]

        # Build final list
        result = []
        if include_core:
            result.extend(core_opps)
        result.extend(dynamic_opps[:max_dynamic])
        result.sort(key=lambda x: x.score, reverse=True)

        self._last_scan_results = result

        logger.info(f"Scan complete: {len(core_opps)} core + {min(len(dynamic_opps), max_dynamic)} dynamic")
        for opp in result[:10]:
            logger.info(f"  {opp.symbol}: score={opp.score:.0f} signal={opp.signal}")

        return result

    def get_tradeable_symbols(self) -> List[str]:
        """Get symbols from last scan."""
        return [o.symbol for o in self._last_scan_results]

    def get_last_scan(self) -> List[Dict]:
        """Get last scan as list of dicts."""
        return [o.to_dict() for o in self._last_scan_results]


# ===== FACTORY FUNCTION =====

def create_scanner(exchange: str, api) -> MarketScanner:
    """
    Factory to create a scanner for a specific exchange.

    Args:
        exchange: Exchange name ("hyperliquid", "binance", etc.)
        api: Exchange API client

    Returns:
        Configured MarketScanner instance
    """
    exchange = exchange.lower()

    if exchange == "hyperliquid":
        provider = HyperliquidDataProvider(api)
    else:
        raise ValueError(f"Unsupported exchange: {exchange}. Supported: hyperliquid")

    return MarketScanner(data_provider=provider)
