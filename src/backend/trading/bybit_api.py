"""Bybit exchange client for USDT perpetual futures.

This module provides a Bybit implementation of the BaseExchange interface,
enabling trading on Bybit USDT perpetual futures markets.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple

from src.backend.config_loader import CONFIG
from src.backend.trading.base_exchange import BaseExchange

logger = logging.getLogger(__name__)


class BybitAPI(BaseExchange):
    """Bybit USDT Perpetuals exchange client.

    Implements BaseExchange interface for Bybit's unified trading API.
    Supports both testnet and mainnet via BYBIT_NETWORK env var.
    """

    def __init__(self):
        """Initialize Bybit client with credentials from config.

        Raises:
            ValueError: If BYBIT_API_KEY or BYBIT_API_SECRET is missing
            ImportError: If pybit library is not installed
        """
        try:
            from pybit.unified_trading import HTTP
        except ImportError:
            raise ImportError(
                "pybit library not installed. "
                "Run: pip install pybit>=5.6.0"
            )

        api_key = CONFIG.get("bybit_api_key")
        api_secret = CONFIG.get("bybit_api_secret")
        network = (CONFIG.get("bybit_network") or "testnet").lower()

        if not api_key or api_key == "your_api_key_here":
            raise ValueError("BYBIT_API_KEY is required")
        if not api_secret or api_secret == "your_api_secret_here":
            raise ValueError("BYBIT_API_SECRET is required")

        self.testnet = network == "testnet"
        self.client = HTTP(
            testnet=self.testnet,
            api_key=api_key,
            api_secret=api_secret,
        )

        self._meta_cache: Optional[Dict] = None
        self._instruments_cache: Dict[str, Dict] = {}
        self._last_cache_time: float = 0
        self._cache_ttl: float = 300  # 5 minutes

        logger.info(f"Bybit client initialized ({'TESTNET' if self.testnet else 'MAINNET'})")

    def _symbol(self, asset: str) -> str:
        """Convert asset name to Bybit symbol format.

        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")

        Returns:
            Bybit symbol (e.g., "BTCUSDT")
        """
        asset = asset.upper().strip()
        if not asset.endswith("USDT"):
            return f"{asset}USDT"
        return asset

    def _asset_from_symbol(self, symbol: str) -> str:
        """Convert Bybit symbol back to asset name.

        Args:
            symbol: Bybit symbol (e.g., "BTCUSDT")

        Returns:
            Asset name (e.g., "BTC")
        """
        if symbol.endswith("USDT"):
            return symbol[:-4]
        return symbol

    async def _fetch_instruments(self) -> None:
        """Fetch and cache instrument info for size/price rounding."""
        now = time.time()
        if self._instruments_cache and (now - self._last_cache_time) < self._cache_ttl:
            return

        try:
            result = await asyncio.to_thread(
                self.client.get_instruments_info,
                category="linear"
            )

            if result.get("retCode") == 0:
                for item in result.get("result", {}).get("list", []):
                    symbol = item.get("symbol", "")
                    self._instruments_cache[symbol] = {
                        "tickSize": float(item.get("priceFilter", {}).get("tickSize", 0.01)),
                        "qtyStep": float(item.get("lotSizeFilter", {}).get("qtyStep", 0.001)),
                        "minOrderQty": float(item.get("lotSizeFilter", {}).get("minOrderQty", 0.001)),
                    }
                self._last_cache_time = now
                logger.debug(f"Cached {len(self._instruments_cache)} Bybit instruments")
        except Exception as e:
            logger.error(f"Failed to fetch Bybit instruments: {e}")

    def _get_instrument(self, asset: str) -> Dict:
        """Get instrument info for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Dict with tickSize, qtyStep, minOrderQty
        """
        symbol = self._symbol(asset)
        return self._instruments_cache.get(symbol, {
            "tickSize": 0.01,
            "qtyStep": 0.001,
            "minOrderQty": 0.001,
        })

    def round_size(self, asset: str, amount: float) -> float:
        """Round order size to Bybit precision.

        Args:
            asset: Asset symbol
            amount: Raw amount

        Returns:
            Rounded amount
        """
        inst = self._get_instrument(asset)
        qty_step = inst["qtyStep"]
        rounded = round(amount / qty_step) * qty_step

        # Determine decimal places
        if qty_step >= 1:
            return round(rounded)
        else:
            decimals = len(str(qty_step).split('.')[-1].rstrip('0'))
            return round(rounded, decimals)

    def round_price(self, asset: str, price: float) -> float:
        """Round price to Bybit tick size.

        Args:
            asset: Asset symbol
            price: Raw price

        Returns:
            Rounded price
        """
        if price is None or price <= 0:
            return price

        inst = self._get_instrument(asset)
        tick_size = inst["tickSize"]
        rounded = round(price / tick_size) * tick_size

        # Determine decimal places
        if tick_size >= 1:
            return round(rounded)
        else:
            decimals = len(str(tick_size).split('.')[-1].rstrip('0'))
            return round(rounded, decimals)

    def get_tick_size(self, asset: str) -> float:
        """Get tick size for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Tick size
        """
        inst = self._get_instrument(asset)
        return inst["tickSize"]

    def extract_oids(self, order_result: Dict[str, Any]) -> List[str]:
        """Extract order IDs from Bybit response.

        Args:
            order_result: Raw order response

        Returns:
            List of order IDs
        """
        oids = []
        try:
            if order_result.get("retCode") == 0:
                result = order_result.get("result", {})
                if "orderId" in result:
                    oids.append(result["orderId"])
                if "orderLinkId" in result:
                    oids.append(result["orderLinkId"])
        except (KeyError, TypeError):
            pass
        return oids

    async def place_buy_order(self, asset: str, amount: float, slippage: float = 0.01) -> Dict[str, Any]:
        """Open a long position with market order.

        Args:
            asset: Asset symbol
            amount: Contract size
            slippage: Not used for Bybit market orders

        Returns:
            Bybit API response
        """
        await self._fetch_instruments()
        symbol = self._symbol(asset)
        qty = str(self.round_size(asset, amount))

        logger.info(f"📤 Bybit BUY: {symbol} qty={qty}")

        result = await asyncio.to_thread(
            self.client.place_order,
            category="linear",
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty=qty,
        )

        if result.get("retCode") != 0:
            logger.error(f"❌ Bybit BUY failed: {result.get('retMsg')}")
        else:
            logger.info(f"✅ Bybit BUY success: {result.get('result', {}).get('orderId')}")

        return result

    async def place_sell_order(self, asset: str, amount: float, slippage: float = 0.01) -> Dict[str, Any]:
        """Open a short position with market order.

        Args:
            asset: Asset symbol
            amount: Contract size
            slippage: Not used for Bybit market orders

        Returns:
            Bybit API response
        """
        await self._fetch_instruments()
        symbol = self._symbol(asset)
        qty = str(self.round_size(asset, amount))

        logger.info(f"📤 Bybit SELL: {symbol} qty={qty}")

        result = await asyncio.to_thread(
            self.client.place_order,
            category="linear",
            symbol=symbol,
            side="Sell",
            orderType="Market",
            qty=qty,
        )

        if result.get("retCode") != 0:
            logger.error(f"❌ Bybit SELL failed: {result.get('retMsg')}")
        else:
            logger.info(f"✅ Bybit SELL success: {result.get('result', {}).get('orderId')}")

        return result

    async def place_take_profit(self, asset: str, is_buy: bool, amount: float, tp_price: float) -> Dict[str, Any]:
        """Place a take-profit order.

        Args:
            asset: Asset symbol
            is_buy: True if the original position is long
            amount: Contract size
            tp_price: Take-profit price

        Returns:
            Dict with success, oid, error, raw_response
        """
        await self._fetch_instruments()
        symbol = self._symbol(asset)
        qty = str(self.round_size(asset, amount))
        tp_price = str(self.round_price(asset, tp_price))

        # For long position, TP is a sell; for short, TP is a buy
        side = "Sell" if is_buy else "Buy"

        logger.info(f"📤 Bybit TP: {symbol} {side} qty={qty} @ {tp_price}")

        try:
            result = await asyncio.to_thread(
                self.client.place_order,
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=qty,
                triggerPrice=tp_price,
                triggerBy="MarkPrice",
                reduceOnly=True,
            )

            success = result.get("retCode") == 0
            oid = result.get("result", {}).get("orderId") if success else None
            error = result.get("retMsg") if not success else None

            if success:
                logger.info(f"✅ Bybit TP placed: {oid}")
            else:
                logger.error(f"❌ Bybit TP failed: {error}")

            return {
                "success": success,
                "oid": oid,
                "error": error,
                "raw_response": result,
            }
        except Exception as e:
            logger.error(f"❌ Bybit TP exception: {e}")
            return {
                "success": False,
                "oid": None,
                "error": str(e),
                "raw_response": None,
            }

    async def place_stop_loss(self, asset: str, is_buy: bool, amount: float, sl_price: float) -> Dict[str, Any]:
        """Place a stop-loss order.

        Args:
            asset: Asset symbol
            is_buy: True if the original position is long
            amount: Contract size
            sl_price: Stop-loss price

        Returns:
            Dict with success, oid, error, raw_response
        """
        await self._fetch_instruments()
        symbol = self._symbol(asset)
        qty = str(self.round_size(asset, amount))
        sl_price = str(self.round_price(asset, sl_price))

        # For long position, SL is a sell; for short, SL is a buy
        side = "Sell" if is_buy else "Buy"

        logger.info(f"📤 Bybit SL: {symbol} {side} qty={qty} @ {sl_price}")

        try:
            result = await asyncio.to_thread(
                self.client.place_order,
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=qty,
                triggerPrice=sl_price,
                triggerBy="MarkPrice",
                reduceOnly=True,
            )

            success = result.get("retCode") == 0
            oid = result.get("result", {}).get("orderId") if success else None
            error = result.get("retMsg") if not success else None

            if success:
                logger.info(f"✅ Bybit SL placed: {oid}")
            else:
                logger.error(f"❌ Bybit SL failed: {error}")

            return {
                "success": success,
                "oid": oid,
                "error": error,
                "raw_response": result,
            }
        except Exception as e:
            logger.error(f"❌ Bybit SL exception: {e}")
            return {
                "success": False,
                "oid": None,
                "error": str(e),
                "raw_response": None,
            }

    async def cancel_order(self, asset: str, oid: Any) -> Dict[str, Any]:
        """Cancel a specific order.

        Args:
            asset: Asset symbol
            oid: Order ID

        Returns:
            Bybit API response
        """
        symbol = self._symbol(asset)

        result = await asyncio.to_thread(
            self.client.cancel_order,
            category="linear",
            symbol=symbol,
            orderId=str(oid),
        )

        if result.get("retCode") != 0:
            logger.warning(f"Cancel order failed: {result.get('retMsg')}")

        return result

    async def cancel_all_orders(self, asset: str) -> Dict[str, Any]:
        """Cancel all orders for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Dict with status and cancelled_count
        """
        symbol = self._symbol(asset)

        try:
            result = await asyncio.to_thread(
                self.client.cancel_all_orders,
                category="linear",
                symbol=symbol,
            )

            if result.get("retCode") == 0:
                cancelled = result.get("result", {}).get("list", [])
                return {"status": "ok", "cancelled_count": len(cancelled)}
            else:
                return {"status": "error", "message": result.get("retMsg")}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders.

        Returns:
            List of order dicts
        """
        try:
            result = await asyncio.to_thread(
                self.client.get_open_orders,
                category="linear",
            )

            orders = []
            if result.get("retCode") == 0:
                for order in result.get("result", {}).get("list", []):
                    # Normalize to our format
                    orders.append({
                        "oid": order.get("orderId"),
                        "coin": self._asset_from_symbol(order.get("symbol", "")),
                        "symbol": order.get("symbol"),
                        "side": order.get("side"),
                        "sz": float(order.get("qty", 0)),
                        "limitPx": float(order.get("price", 0)) if order.get("price") else None,
                        "triggerPx": float(order.get("triggerPrice", 0)) if order.get("triggerPrice") else None,
                        "orderType": order.get("orderType"),
                        "reduceOnly": order.get("reduceOnly", False),
                        "createdTime": order.get("createdTime"),
                    })
            return orders
        except Exception as e:
            logger.error(f"Get open orders error: {e}")
            return []

    async def get_user_state(self) -> Dict[str, Any]:
        """Get account balance and positions.

        Returns:
            Dict with balance, total_value, positions
        """
        try:
            # Get wallet balance
            wallet_result = await asyncio.to_thread(
                self.client.get_wallet_balance,
                accountType="UNIFIED",
            )

            balance = 0.0
            total_value = 0.0

            if wallet_result.get("retCode") == 0:
                accounts = wallet_result.get("result", {}).get("list", [])
                for acc in accounts:
                    total_value = float(acc.get("totalEquity", 0))
                    balance = float(acc.get("totalAvailableBalance", 0))
                    break

            # Get positions
            pos_result = await asyncio.to_thread(
                self.client.get_positions,
                category="linear",
                settleCoin="USDT",
            )

            positions = []
            if pos_result.get("retCode") == 0:
                for pos in pos_result.get("result", {}).get("list", []):
                    size = float(pos.get("size", 0))
                    if size == 0:
                        continue

                    side = pos.get("side", "")
                    entry_px = float(pos.get("avgPrice", 0))
                    unrealized_pnl = float(pos.get("unrealisedPnl", 0))

                    positions.append({
                        "coin": self._asset_from_symbol(pos.get("symbol", "")),
                        "symbol": pos.get("symbol"),
                        "szi": size if side == "Buy" else -size,
                        "entryPx": entry_px,
                        "pnl": unrealized_pnl,
                        "notional_entry": abs(size) * entry_px,
                        "leverage": pos.get("leverage"),
                        "positionValue": float(pos.get("positionValue", 0)),
                    })

            return {
                "balance": balance,
                "total_value": total_value,
                "positions": positions,
            }
        except Exception as e:
            logger.error(f"Get user state error: {e}")
            return {"balance": 0.0, "total_value": 0.0, "positions": []}

    async def get_current_price(self, asset: str) -> float:
        """Get current price for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Current price
        """
        try:
            symbol = self._symbol(asset)
            result = await asyncio.to_thread(
                self.client.get_tickers,
                category="linear",
                symbol=symbol,
            )

            if result.get("retCode") == 0:
                tickers = result.get("result", {}).get("list", [])
                if tickers:
                    return float(tickers[0].get("lastPrice", 0))
            return 0.0
        except Exception as e:
            logger.error(f"Get price error for {asset}: {e}")
            return 0.0

    async def get_candles(self, asset: str, interval: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        """Get OHLCV candles.

        Args:
            asset: Asset symbol
            interval: Candle interval
            limit: Number of candles

        Returns:
            List of candle dicts
        """
        try:
            symbol = self._symbol(asset)

            # Map interval to Bybit format
            interval_map = {
                "1m": "1",
                "5m": "5",
                "15m": "15",
                "30m": "30",
                "1h": "60",
                "4h": "240",
                "1d": "D",
            }
            bybit_interval = interval_map.get(interval, "60")

            result = await asyncio.to_thread(
                self.client.get_kline,
                category="linear",
                symbol=symbol,
                interval=bybit_interval,
                limit=limit,
            )

            candles = []
            if result.get("retCode") == 0:
                for c in result.get("result", {}).get("list", []):
                    # Bybit returns: [startTime, open, high, low, close, volume, turnover]
                    candles.append({
                        "timestamp": int(c[0]),
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    })

            # Bybit returns newest first, reverse to oldest first
            candles.reverse()
            return candles
        except Exception as e:
            logger.error(f"Get candles error for {asset}: {e}")
            return []

    async def get_meta_and_ctxs(self) -> Any:
        """Get market metadata (cached).

        Returns:
            Instruments info
        """
        await self._fetch_instruments()
        return self._instruments_cache

    async def get_funding_rate(self, asset: str) -> Optional[float]:
        """Get funding rate for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Funding rate or None
        """
        try:
            symbol = self._symbol(asset)
            result = await asyncio.to_thread(
                self.client.get_tickers,
                category="linear",
                symbol=symbol,
            )

            if result.get("retCode") == 0:
                tickers = result.get("result", {}).get("list", [])
                if tickers:
                    fr = tickers[0].get("fundingRate")
                    return float(fr) if fr else None
            return None
        except Exception as e:
            logger.error(f"Get funding rate error for {asset}: {e}")
            return None

    async def get_open_interest(self, asset: str) -> Optional[float]:
        """Get open interest for an asset.

        Args:
            asset: Asset symbol

        Returns:
            Open interest or None
        """
        try:
            symbol = self._symbol(asset)
            result = await asyncio.to_thread(
                self.client.get_tickers,
                category="linear",
                symbol=symbol,
            )

            if result.get("retCode") == 0:
                tickers = result.get("result", {}).get("list", [])
                if tickers:
                    oi = tickers[0].get("openInterest")
                    return float(oi) if oi else None
            return None
        except Exception as e:
            logger.error(f"Get OI error for {asset}: {e}")
            return None

    async def get_recent_fills(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent fills.

        Args:
            limit: Max fills to return

        Returns:
            List of fill dicts
        """
        try:
            result = await asyncio.to_thread(
                self.client.get_executions,
                category="linear",
                limit=limit,
            )

            fills = []
            if result.get("retCode") == 0:
                for fill in result.get("result", {}).get("list", []):
                    fills.append({
                        "coin": self._asset_from_symbol(fill.get("symbol", "")),
                        "side": fill.get("side"),
                        "px": float(fill.get("execPrice", 0)),
                        "sz": float(fill.get("execQty", 0)),
                        "fee": float(fill.get("execFee", 0)),
                        "time": fill.get("execTime"),
                    })
            return fills
        except Exception as e:
            logger.error(f"Get recent fills error: {e}")
            return []
