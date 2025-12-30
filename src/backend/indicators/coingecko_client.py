"""
CoinGecko Client - Market-wide crypto data for scanner

Free API: 50 requests/minute, no API key required
Docs: https://www.coingecko.com/en/api/documentation
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CoinMarketData:
    """Market data for a single coin"""
    id: str  # CoinGecko ID (e.g., "bitcoin")
    symbol: str  # Trading symbol (e.g., "BTC")
    name: str  # Full name (e.g., "Bitcoin")
    price: float
    market_cap: float
    market_cap_rank: int
    volume_24h: float
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    ath: float  # All-time high
    ath_change_pct: float  # % from ATH

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "symbol": self.symbol.upper(),
            "name": self.name,
            "price": self.price,
            "market_cap": self.market_cap,
            "market_cap_rank": self.market_cap_rank,
            "volume_24h": self.volume_24h,
            "price_change_1h": self.price_change_1h,
            "price_change_24h": self.price_change_24h,
            "price_change_7d": self.price_change_7d,
            "ath": self.ath,
            "ath_change_pct": self.ath_change_pct,
        }


class CoinGeckoClient:
    """
    Client for CoinGecko API - free market data for all crypto.

    Features:
    - Top coins by market cap
    - Price, volume, market cap data
    - Price changes (1h, 24h, 7d)
    - No API key required (free tier)
    """

    BASE_URL = "https://api.coingecko.com/api/v3"

    # Stablecoins and wrapped tokens to exclude
    EXCLUDED_CATEGORIES = {
        "stablecoins", "wrapped-tokens", "bridged-tokens"
    }
    EXCLUDED_SYMBOLS = {
        "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "FRAX", "USDD",
        "WBTC", "WETH", "STETH", "WSTETH", "RETH", "CBETH",
        "LEO", "OKB", "CRO", "KCS", "HT", "FTT",  # Exchange tokens
    }

    def __init__(self, cache_ttl: int = 60):
        """
        Initialize CoinGecko client.

        Args:
            cache_ttl: Cache time-to-live in seconds (default 60s)
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}  # {key: (data, timestamp)}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_cache(self, key: str) -> Optional[any]:
        """Get cached data if not expired"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return data
        return None

    def _set_cache(self, key: str, data: any):
        """Set cache with current timestamp"""
        self._cache[key] = (data, datetime.now())

    async def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}{endpoint}"

            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("CoinGecko rate limit hit, waiting 60s...")
                    await asyncio.sleep(60)
                    return await self._request(endpoint, params)
                else:
                    logger.error(f"CoinGecko API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"CoinGecko request failed: {e}")
            return None

    async def get_top_coins(
        self,
        limit: int = 100,
        exclude_stables: bool = True
    ) -> List[CoinMarketData]:
        """
        Get top coins by market cap with full market data.

        Args:
            limit: Number of coins to fetch (max 250)
            exclude_stables: Exclude stablecoins and wrapped tokens

        Returns:
            List of CoinMarketData objects
        """
        cache_key = f"top_coins_{limit}_{exclude_stables}"
        cached = self._get_cache(cache_key)
        if cached:
            logger.debug("Using cached top coins data")
            return cached

        # Fetch more than needed to account for exclusions
        fetch_limit = min(limit * 2, 250) if exclude_stables else limit

        data = await self._request("/coins/markets", {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": fetch_limit,
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d"
        })

        if not data:
            logger.error("Failed to fetch top coins from CoinGecko")
            return []

        coins = []
        for item in data:
            symbol = item.get("symbol", "").upper()

            # Skip excluded symbols
            if exclude_stables and symbol in self.EXCLUDED_SYMBOLS:
                continue

            try:
                coin = CoinMarketData(
                    id=item.get("id", ""),
                    symbol=symbol,
                    name=item.get("name", ""),
                    price=float(item.get("current_price", 0) or 0),
                    market_cap=float(item.get("market_cap", 0) or 0),
                    market_cap_rank=int(item.get("market_cap_rank", 0) or 0),
                    volume_24h=float(item.get("total_volume", 0) or 0),
                    price_change_1h=float(item.get("price_change_percentage_1h_in_currency", 0) or 0),
                    price_change_24h=float(item.get("price_change_percentage_24h", 0) or 0),
                    price_change_7d=float(item.get("price_change_percentage_7d_in_currency", 0) or 0),
                    ath=float(item.get("ath", 0) or 0),
                    ath_change_pct=float(item.get("ath_change_percentage", 0) or 0),
                )
                coins.append(coin)

                if len(coins) >= limit:
                    break

            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse coin data for {symbol}: {e}")
                continue

        logger.info(f"Fetched {len(coins)} coins from CoinGecko")
        self._set_cache(cache_key, coins)
        return coins

    async def get_coin_by_symbol(self, symbol: str) -> Optional[CoinMarketData]:
        """
        Get market data for a specific coin by symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")

        Returns:
            CoinMarketData or None if not found
        """
        # First try cache
        coins = self._get_cache("top_coins_100_True")
        if coins:
            for coin in coins:
                if coin.symbol == symbol.upper():
                    return coin

        # Otherwise fetch fresh
        coins = await self.get_top_coins(limit=250)
        for coin in coins:
            if coin.symbol == symbol.upper():
                return coin

        return None

    async def get_trending_coins(self) -> List[Dict]:
        """
        Get trending coins on CoinGecko (search trends).

        Returns:
            List of trending coin info
        """
        cache_key = "trending"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        data = await self._request("/search/trending")
        if not data:
            return []

        trending = []
        for item in data.get("coins", []):
            coin = item.get("item", {})
            trending.append({
                "id": coin.get("id"),
                "symbol": coin.get("symbol", "").upper(),
                "name": coin.get("name"),
                "market_cap_rank": coin.get("market_cap_rank"),
                "score": coin.get("score"),  # Trending score
            })

        self._set_cache(cache_key, trending)
        return trending

    async def get_global_data(self) -> Dict:
        """
        Get global crypto market data.

        Returns:
            Dict with total market cap, volume, BTC dominance, etc.
        """
        cache_key = "global"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        data = await self._request("/global")
        if not data:
            return {}

        global_data = data.get("data", {})
        result = {
            "total_market_cap": global_data.get("total_market_cap", {}).get("usd", 0),
            "total_volume_24h": global_data.get("total_volume", {}).get("usd", 0),
            "btc_dominance": global_data.get("market_cap_percentage", {}).get("btc", 0),
            "eth_dominance": global_data.get("market_cap_percentage", {}).get("eth", 0),
            "active_cryptocurrencies": global_data.get("active_cryptocurrencies", 0),
            "market_cap_change_24h": global_data.get("market_cap_change_percentage_24h_usd", 0),
        }

        self._set_cache(cache_key, result)
        return result


# Singleton instance
_client: Optional[CoinGeckoClient] = None

def get_coingecko_client() -> CoinGeckoClient:
    """Get singleton CoinGecko client instance"""
    global _client
    if _client is None:
        _client = CoinGeckoClient()
    return _client
