"""High-level Hyperliquid exchange client with async retry helpers.

This module wraps the Hyperliquid `Exchange` and `Info` SDK classes to provide a
single entry point for submitting trades, managing orders, and retrieving market
state.  It normalizes retry behaviour, adds logging, and caches metadata so that
the trading agent can depend on predictable, non-blocking IO.

V6 CHANGES:
- Now extends BaseExchange for modular exchange support
- Added extract_oids and get_meta_and_ctxs to interface

V5 CHANGES:
- Added round_price() method to round prices to exchange tick size
- Added TICK_SIZES constant with per-asset tick sizes
- place_take_profit and place_stop_loss now automatically round prices
- Better error messages for price rejection

V4 CHANGES:
- Added response logging and validation for place_take_profit
- Added response logging and validation for place_stop_loss
- Both methods now log the actual API response and check for errors
- Added helper method _validate_order_response for consistent validation
"""

import asyncio
import logging
import aiohttp
from typing import TYPE_CHECKING, Dict, Any, Optional, Tuple, List
from src.backend.config_loader import CONFIG
from src.backend.trading.base_exchange import BaseExchange
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants  # For MAINNET/TESTNET
from hyperliquid.utils.error import ServerError  # For handling 502 errors
from eth_account import Account as _Account
from eth_account.signers.local import LocalAccount
from websocket._exceptions import WebSocketConnectionClosedException
import socket

if TYPE_CHECKING:
    # Type stubs for linter - eth_account's type stubs are incorrect
    class Account:
        @staticmethod
        def from_key(_private_key: str) -> LocalAccount: ...
        @staticmethod
        def from_mnemonic(_mnemonic: str) -> LocalAccount: ...
        @staticmethod
        def enable_unaudited_hdwallet_features() -> None: ...
else:
    Account = _Account


class HyperliquidAPI(BaseExchange):
    """Facade around Hyperliquid SDK clients with async convenience methods.

    The class owns wallet credentials, connection configuration, and provides
    coroutine helpers that keep retry semantics and logging consistent across
    the trading agent.

    Extends BaseExchange for modular multi-exchange support.
    """

    # ===== TICK SIZES FOR PRICE ROUNDING =====
    # Hyperliquid requires prices to be rounded to specific tick sizes
    # These values are from Hyperliquid's asset metadata
    # Reference: https://hyperliquid.gitbook.io/hyperliquid-docs/trading/asset-specifications
    TICK_SIZES = {
        "BTC": 1.0,      # $1.00 tick size
        "ETH": 0.1,      # $0.10 tick size
        "SOL": 0.01,     # $0.01 tick size
        "AVAX": 0.01,
        "MATIC": 0.0001,
        "LINK": 0.001,
        "DOGE": 0.00001,
        "ARB": 0.0001,
        "OP": 0.001,
        "APT": 0.01,
        "INJ": 0.01,
        "SUI": 0.0001,
        "SEI": 0.0001,
        "TIA": 0.001,
        "ATOM": 0.001,
        "DOT": 0.001,
        "FTM": 0.0001,
        "NEAR": 0.001,
        "WLD": 0.001,
        "BLUR": 0.0001,
        "LDO": 0.001,
        "GMX": 0.01,
        "STX": 0.0001,
        "ORDI": 0.01,
        "PEPE": 0.00000001,
        "WIF": 0.0001,
        "BONK": 0.000000001,
        "JUP": 0.0001,
        "PENDLE": 0.001,
        "W": 0.0001,
        "ENA": 0.0001,
        # Default for unknown assets
        "_DEFAULT": 0.01
    }

    def __init__(self):
        """Initialize wallet credentials and instantiate exchange clients.

        Raises:
            ValueError: If neither a private key nor mnemonic is present in the
                configuration.
        """
        self._meta_cache = None
        private_key = CONFIG.get("hyperliquid_private_key")
        mnemonic = CONFIG.get("mnemonic")
        
        # Check if we have valid credentials
        if private_key and private_key != "your_private_key_here":
            try:
                self.wallet = Account.from_key(private_key)
            except Exception as e:
                raise ValueError(f"Invalid HYPERLIQUID_PRIVATE_KEY format: {e}")
        elif mnemonic and mnemonic != "your_mnemonic_here":
            try:
                Account.enable_unaudited_hdwallet_features()
                self.wallet = Account.from_mnemonic(mnemonic)
            except Exception as e:
                raise ValueError(f"Invalid MNEMONIC format: {e}")
        else:
            raise ValueError("Either HYPERLIQUID_PRIVATE_KEY/LIGHTER_PRIVATE_KEY or MNEMONIC must be provided with valid values (not placeholder)")
        # Choose base URL: allow override via env-config; fallback to network selection
        network = (CONFIG.get("hyperliquid_network") or "mainnet").lower()
        base_url = CONFIG.get("hyperliquid_base_url")

        if not base_url:
            if network == "testnet":
                # Try to get testnet URL from constants, otherwise use hard-coded testnet endpoint
                try:
                    base_url = constants.TESTNET_API_URL
                    logging.info("Using Hyperliquid TESTNET (from SDK constants)")
                except AttributeError:
                    # SDK doesn't have TESTNET_API_URL, use hard-coded testnet endpoint
                    base_url = "https://api.hyperliquid-testnet.xyz"
                    logging.info("Using Hyperliquid TESTNET (hard-coded endpoint)")
            else:
                base_url = constants.MAINNET_API_URL
                logging.info("Using Hyperliquid MAINNET")
        else:
            logging.info(f"Using custom Hyperliquid URL: {base_url}")

        self.base_url = base_url
        self._build_clients()

    def _build_clients(self):
        """Instantiate exchange and info client instances for the active base URL."""
        self.info = Info(self.base_url)
        self.exchange = Exchange(self.wallet, self.base_url)

    def _reset_clients(self):
        """Recreate SDK clients after connection failures while logging failures."""
        try:
            self._build_clients()
            logging.warning("Hyperliquid clients re-instantiated after connection issue")
        except (ValueError, AttributeError, RuntimeError) as e:
            logging.error("Failed to reset Hyperliquid clients: %s", e)

    async def _retry(self, fn, *args, max_attempts: int = 3, backoff_base: float = 0.5, reset_on_fail: bool = True, to_thread: bool = True, **kwargs):
        """Retry helper with exponential backoff and optional thread offloading.

        Args:
            fn: Callable to invoke, either sync (supports `asyncio.to_thread`) or
                async depending on ``to_thread``. The callable should raise
                exceptions rather than returning sentinel values.
            *args: Positional arguments forwarded to ``fn``.
            max_attempts: Maximum number of attempts before surfacing the last
                exception.
            backoff_base: Initial delay in seconds, doubled after each failure.
            reset_on_fail: Whether to rebuild Hyperliquid clients after a
                failure.
            to_thread: If ``True`` the callable is executed in a worker thread.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            Result produced by ``fn``.

        Raises:
            Exception: Propagates any exception raised by ``fn`` after retries.
        """
        last_err = None
        for attempt in range(max_attempts):
            try:
                if to_thread:
                    return await asyncio.to_thread(fn, *args, **kwargs)
                return await fn(*args, **kwargs)
            except ServerError as e:
                # Hyperliquid SDK ServerError (includes 502, 503, etc.)
                last_err = e
                status_code = e.args[0] if e.args else 0
                error_msg = e.args[1] if len(e.args) > 1 else str(e)

                # More aggressive backoff for 5xx errors
                wait_time = backoff_base * (3 ** attempt)  # 0.5, 1.5, 4.5 seconds
                logging.warning(
                    "⚠️ Hyperliquid ServerError %s (attempt %s/%s): %s. Waiting %.1fs...",
                    status_code, attempt + 1, max_attempts, error_msg[:100], wait_time
                )

                if reset_on_fail:
                    self._reset_clients()
                await asyncio.sleep(wait_time)
                continue
            except (WebSocketConnectionClosedException, aiohttp.ClientError, ConnectionError, TimeoutError, socket.timeout) as e:
                last_err = e
                logging.warning("HL call failed (attempt %s/%s): %s", attempt + 1, max_attempts, e)
                if reset_on_fail:
                    self._reset_clients()
                await asyncio.sleep(backoff_base * (2 ** attempt))
                continue
            except (RuntimeError, ValueError, KeyError, AttributeError) as e:
                # Unknown errors: don't spin forever, but allow a quick reset once
                last_err = e
                logging.warning("HL call unexpected error (attempt %s/%s): %s", attempt + 1, max_attempts, e)
                if reset_on_fail and attempt == 0:
                    self._reset_clients()
                    await asyncio.sleep(backoff_base)
                    continue
                break
        raise last_err if last_err else RuntimeError("Hyperliquid retry: unknown error")

    def get_tick_size(self, asset: str) -> float:
        """Get the tick size (minimum price increment) for an asset.
        
        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            
        Returns:
            Tick size as float
        """
        return self.TICK_SIZES.get(asset.upper(), self.TICK_SIZES["_DEFAULT"])

    def round_price(self, asset: str, price: float) -> float:
        """Round a price to the correct tick size for the asset.
        
        CRITICAL: Hyperliquid rejects orders with prices not aligned to tick size.
        This method ensures all prices are properly rounded.
        
        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            price: Raw price to round
            
        Returns:
            Price rounded to the asset's tick size
            
        Examples:
            >>> api.round_price("BTC", 92700.51)
            92701.0
            >>> api.round_price("ETH", 3128.45)
            3128.5
            >>> api.round_price("SOL", 145.678)
            145.68
        """
        if price is None or price <= 0:
            return price
            
        tick_size = self.get_tick_size(asset)
        
        # Round to nearest tick
        rounded = round(price / tick_size) * tick_size
        
        # Determine decimal places from tick size
        if tick_size >= 1:
            decimals = 0
        else:
            # Count decimal places in tick size
            tick_str = f"{tick_size:.10f}".rstrip('0')
            if '.' in tick_str:
                decimals = len(tick_str.split('.')[1])
            else:
                decimals = 0
        
        # Final round to avoid floating point errors
        rounded = round(rounded, decimals)
        
        logging.debug(f"round_price({asset}, {price}) -> {rounded} (tick_size={tick_size})")
        return rounded

    def round_size(self, asset, amount):
        """Round order size to the asset precision defined by market metadata.

        Args:
            asset: Symbol of the market whose contract size we are rounding to.
            amount: Desired contract size before rounding.

        Returns:
            The input ``amount`` rounded to the market's ``szDecimals`` precision.
        """
        meta = self._meta_cache[0] if hasattr(self, '_meta_cache') and self._meta_cache else None
        if meta:
            universe = meta.get("universe", [])
            asset_info = next((u for u in universe if u.get("name") == asset), None)
            if asset_info:
                decimals = asset_info.get("szDecimals", 8)
                return round(amount, decimals)
        return round(amount, 8)

    def _validate_order_response(self, result: Any, asset: str, order_type: str) -> Tuple[bool, Optional[int], Optional[str]]:
        """Validate an order response from the exchange.
        
        Args:
            result: Raw response from exchange.order()
            asset: Asset symbol for logging
            order_type: "TP" or "SL" for logging
            
        Returns:
            Tuple of (success: bool, oid: Optional[int], error_message: Optional[str])
        """
        if not isinstance(result, dict):
            return False, None, f"Unexpected response type: {type(result)}"
        
        status = result.get("status")
        
        if status == "err":
            error_msg = result.get("response", "unknown error")
            logging.error(f"❌ {order_type} order REJECTED for {asset}: {error_msg}")
            return False, None, str(error_msg)
        
        if status == "ok":
            oids = self.extract_oids(result)
            if oids:
                logging.info(f"✅ {order_type} order CONFIRMED for {asset}, oid: {oids[0]}")
                return True, oids[0], None
            else:
                # Check if order was filled immediately (unlikely for TP/SL but possible)
                response_data = result.get("response", {})
                if isinstance(response_data, dict):
                    data = response_data.get("data", {})
                    statuses = data.get("statuses", [])
                    for st in statuses:
                        if "error" in st:
                            logging.error(f"❌ {order_type} order error for {asset}: {st['error']}")
                            return False, None, st["error"]
                
                logging.warning(f"⚠️ {order_type} order returned ok but no oid found for {asset}: {result}")
                return False, None, "No order ID in response"
        
        logging.warning(f"⚠️ {order_type} order unknown status for {asset}: {result}")
        return False, None, f"Unknown status: {status}"

    async def place_buy_order(self, asset, amount, slippage=0.01):
        """Submit a market buy order with exchange-side rounding and retry logic.

        Args:
            asset: Market symbol to open.
            amount: Contract size to open before rounding.
            slippage: Maximum acceptable slippage expressed as a decimal.

        Returns:
            Raw SDK response from :meth:`Exchange.market_open`.
        """
        amount = self.round_size(asset, amount)
        return await self._retry(lambda: self.exchange.market_open(asset, True, amount, None, slippage))

    async def place_sell_order(self, asset, amount, slippage=0.01):
        """Submit a market sell order with exchange-side rounding and retry logic.

        Args:
            asset: Market symbol to open.
            amount: Contract size to open before rounding.
            slippage: Maximum acceptable slippage expressed as a decimal.

        Returns:
            Raw SDK response from :meth:`Exchange.market_open`.
        """
        amount = self.round_size(asset, amount)
        return await self._retry(lambda: self.exchange.market_open(asset, False, amount, None, slippage))

    async def place_take_profit(self, asset, is_buy, amount, tp_price) -> Dict[str, Any]:
        """Create a reduce-only trigger order that executes a take-profit exit.

        Args:
            asset: Market symbol to trade.
            is_buy: ``True`` if the original position is long; dictates close
                direction.
            amount: Contract size to close.
            tp_price: Trigger price for the take-profit order.

        Returns:
            Dict with keys:
                - 'success': bool
                - 'oid': int or None
                - 'error': str or None  
                - 'raw_response': original API response
        """
        amount = self.round_size(asset, amount)
        
        # ===== CRITICAL: Round price to tick size =====
        original_price = tp_price
        tp_price = self.round_price(asset, tp_price)
        if original_price != tp_price:
            logging.info(f"📐 TP price rounded: {original_price} -> {tp_price} (tick size: {self.get_tick_size(asset)})")
        
        order_type = {"trigger": {"triggerPx": tp_price, "isMarket": True, "tpsl": "tp"}}
        
        logging.info(f"📤 Placing TP order: {asset} {'LONG→SELL' if is_buy else 'SHORT→BUY'} {amount} @ {tp_price}")
        
        result = await self._retry(lambda: self.exchange.order(asset, not is_buy, amount, tp_price, order_type, True))
        
        # Log full response for debugging
        logging.debug(f"TP order raw response for {asset}: {result}")
        
        # Validate and extract result
        success, oid, error = self._validate_order_response(result, asset, "TP")
        
        return {
            "success": success,
            "oid": oid,
            "error": error,
            "raw_response": result
        }

    async def place_stop_loss(self, asset, is_buy, amount, sl_price) -> Dict[str, Any]:
        """Create a reduce-only trigger order that executes a stop-loss exit.

        Args:
            asset: Market symbol to trade.
            is_buy: ``True`` if the original position is long; dictates close
                direction.
            amount: Contract size to close.
            sl_price: Trigger price for the stop-loss order.

        Returns:
            Dict with keys:
                - 'success': bool
                - 'oid': int or None
                - 'error': str or None
                - 'raw_response': original API response
        """
        amount = self.round_size(asset, amount)
        
        # ===== CRITICAL: Round price to tick size =====
        original_price = sl_price
        sl_price = self.round_price(asset, sl_price)
        if original_price != sl_price:
            logging.info(f"📐 SL price rounded: {original_price} -> {sl_price} (tick size: {self.get_tick_size(asset)})")
        
        order_type = {"trigger": {"triggerPx": sl_price, "isMarket": True, "tpsl": "sl"}}
        
        logging.info(f"📤 Placing SL order: {asset} {'LONG→SELL' if is_buy else 'SHORT→BUY'} {amount} @ {sl_price}")
        
        result = await self._retry(lambda: self.exchange.order(asset, not is_buy, amount, sl_price, order_type, True))
        
        # Log full response for debugging
        logging.debug(f"SL order raw response for {asset}: {result}")
        
        # Validate and extract result
        success, oid, error = self._validate_order_response(result, asset, "SL")
        
        return {
            "success": success,
            "oid": oid,
            "error": error,
            "raw_response": result
        }

    async def cancel_order(self, asset, oid):
        """Cancel a single order by identifier for a given asset.

        Args:
            asset: Market symbol associated with the order.
            oid: Hyperliquid order identifier to cancel.

        Returns:
            Raw SDK response from :meth:`Exchange.cancel`.
        """
        return await self._retry(lambda: self.exchange.cancel(asset, oid))

    async def cancel_all_orders(self, asset):
        """Cancel every open order for ``asset`` owned by the configured wallet."""
        try:
            open_orders = await self._retry(lambda: self.info.frontend_open_orders(self.wallet.address))
            for order in open_orders:
                if order.get("coin") == asset:
                    oid = order.get("oid")
                    if oid:
                        await self.cancel_order(asset, oid)
            return {"status": "ok", "cancelled_count": len([o for o in open_orders if o.get("coin") == asset])}
        except (RuntimeError, ValueError, KeyError, ConnectionError) as e:
            logging.error("Cancel all orders error for %s: %s", asset, e)
            return {"status": "error", "message": str(e)}

    async def get_open_orders(self):
        """Fetch and normalize open orders associated with the wallet.

        Returns:
            List of order dictionaries augmented with ``triggerPx`` when present.
        """
        try:
            orders = await self._retry(lambda: self.info.frontend_open_orders(self.wallet.address))
            
            # DIAGNOSTIC: Log raw response
            logging.debug(f"📋 get_open_orders: Received {len(orders)} orders from SDK")
            if orders:
                logging.debug(f"📋 get_open_orders: First order sample: {orders[0] if orders else 'N/A'}")
            
            # Normalize trigger price if present in orderType
            for o in orders:
                try:
                    ot = o.get("orderType")
                    if isinstance(ot, dict) and "trigger" in ot:
                        trig = ot.get("trigger") or {}
                        if "triggerPx" in trig:
                            o["triggerPx"] = float(trig["triggerPx"])
                except (ValueError, KeyError, TypeError):
                    continue
            return orders
        except (RuntimeError, ValueError, KeyError, ConnectionError) as e:
            logging.error("Get open orders error: %s", e)
            return []

    async def get_recent_fills(self, limit: int = 50):
        """Return the most recent fills when supported by the SDK variant.

        Args:
            limit: Maximum number of fills to return.

        Returns:
            List of fill dictionaries or an empty list if unsupported.
        """
        try:
            # Some SDK versions expose user_fills; fall back gracefully if absent
            if hasattr(self.info, 'user_fills'):
                fills = await self._retry(lambda: self.info.user_fills(self.wallet.address))
            elif hasattr(self.info, 'fills'):
                fills = await self._retry(lambda: self.info.fills(self.wallet.address))
            else:
                return []
            if isinstance(fills, list):
                return fills[-limit:]
            return []
        except (RuntimeError, ValueError, KeyError, ConnectionError, AttributeError) as e:
            logging.error("Get recent fills error: %s", e)
            return []

    def extract_oids(self, order_result):
        """Extract resting or filled order identifiers from an exchange response.

        Args:
            order_result: Raw order response payload returned by the exchange.

        Returns:
            List of order identifiers present in resting or filled status entries.
        """
        oids = []
        try:
            statuses = order_result["response"]["data"]["statuses"]
            for st in statuses:
                if "resting" in st and "oid" in st["resting"]:
                    oids.append(st["resting"]["oid"])
                if "filled" in st and "oid" in st["filled"]:
                    oids.append(st["filled"]["oid"])
        except (KeyError, TypeError, ValueError):
            pass
        return oids

    async def get_user_state(self):
        """Retrieve wallet state with enriched position PnL calculations.

        Returns:
            Dictionary with ``balance``, ``total_value``, and ``positions``.
        """
        state = await self._retry(lambda: self.info.user_state(self.wallet.address))
        positions = state.get("assetPositions", [])
        
        # Hyperliquid returns accountValue in crossMarginSummary
        cross_margin = state.get("crossMarginSummary", {})
        total_value = float(cross_margin.get("accountValue", 0.0) or state.get("accountValue", 0.0))
        
        enriched_positions = []
        for pos_wrap in positions:
            pos = pos_wrap["position"]
            entry_px = float(pos.get("entryPx", 0) or 0)
            size = float(pos.get("szi", 0) or 0)
            side = "long" if size > 0 else "short"
            current_px = await self.get_current_price(pos["coin"]) if entry_px and size else 0.0
            pnl = (current_px - entry_px) * abs(size) if side == "long" else (entry_px - current_px) * abs(size)
            pos["pnl"] = pnl
            pos["notional_entry"] = abs(size) * entry_px
            enriched_positions.append(pos)
        
        # withdrawable is also in crossMarginSummary
        balance = float(cross_margin.get("withdrawable", 0.0) or state.get("withdrawable", 0.0))
        
        if not total_value:
            total_value = balance + sum(max(p.get("pnl", 0.0), 0.0) for p in enriched_positions)
        return {"balance": balance, "total_value": total_value, "positions": enriched_positions}

    async def get_current_price(self, asset):
        """Return the latest mid-price for ``asset``.

        Args:
            asset: Market symbol to query.

        Returns:
            Mid-price as a float, or ``0.0`` when unavailable.
        """
        mids = await self._retry(self.info.all_mids)
        return float(mids.get(asset, 0.0))

    async def get_meta_and_ctxs(self):
        """Return cached meta/context information, fetching once per lifecycle.

        Returns:
            Cached metadata response as returned by
            :meth:`Info.meta_and_asset_ctxs`.
        """
        if not self._meta_cache:
            response = await self._retry(self.info.meta_and_asset_ctxs)
            self._meta_cache = response
        return self._meta_cache

    async def get_open_interest(self, asset):
        """Return open interest for ``asset`` if it exists in cached metadata.

        Args:
            asset: Market symbol to query.

        Returns:
            Rounded open interest or ``None`` if unavailable.
        """
        try:
            data = await self.get_meta_and_ctxs()
            if isinstance(data, list) and len(data) >= 2:
                meta, asset_ctxs = data[0], data[1]
                universe = meta.get("universe", [])
                asset_idx = next((i for i, u in enumerate(universe) if u.get("name") == asset), None)
                if asset_idx is not None and asset_idx < len(asset_ctxs):
                    oi = asset_ctxs[asset_idx].get("openInterest")
                    return round(float(oi), 2) if oi else None
            return None
        except (RuntimeError, ValueError, KeyError, ConnectionError, TypeError) as e:
            logging.error("OI fetch error for %s: %s", asset, e)
            return None

    async def get_funding_rate(self, asset):
        """Return the most recent funding rate for ``asset`` if available.

        Args:
            asset: Market symbol to query.

        Returns:
            Funding rate as a float or ``None`` when not present.
        """
        try:
            data = await self.get_meta_and_ctxs()
            if isinstance(data, list) and len(data) >= 2:
                meta, asset_ctxs = data[0], data[1]
                universe = meta.get("universe", [])
                asset_idx = next((i for i, u in enumerate(universe) if u.get("name") == asset), None)
                if asset_idx is not None and asset_idx < len(asset_ctxs):
                    funding = asset_ctxs[asset_idx].get("funding")
                    return round(float(funding), 8) if funding else None
            return None
        except (RuntimeError, ValueError, KeyError, ConnectionError, TypeError) as e:
            logging.error("Funding fetch error for %s: %s", asset, e)
            return None

    async def get_candles(self, asset: str, interval: str = "1h", limit: int = 100) -> list:
        """
        Fetch OHLCV candle data from Hyperliquid.

        Args:
            asset: Asset symbol (e.g., "BTC", "ETH")
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d")
            limit: Number of candles to fetch (max 5000)

        Returns:
            List of candle dicts with keys: timestamp, open, high, low, close, volume
            Returns empty list on error.
        """
        try:
            # Map interval strings to milliseconds
            interval_map = {
                "1m": 60 * 1000,
                "5m": 5 * 60 * 1000,
                "15m": 15 * 60 * 1000,
                "1h": 60 * 60 * 1000,
                "4h": 4 * 60 * 60 * 1000,
                "1d": 24 * 60 * 60 * 1000,
            }
            
            interval_ms = interval_map.get(interval, 60 * 60 * 1000)  # Default to 1h
            
            # Calculate time range
            import time
            end_time = int(time.time() * 1000)
            start_time = end_time - (limit * interval_ms)
            
            # Hyperliquid candleSnapshot request
            payload = {
                "type": "candleSnapshot",
                "req": {
                    "coin": asset,
                    "interval": interval,
                    "startTime": start_time,
                    "endTime": end_time
                }
            }
            
            # Make POST request to info endpoint
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/info",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        logging.warning(f"Candle fetch failed for {asset}: HTTP {resp.status}")
                        return []
                    
                    data = await resp.json()
            
            # Parse response into standard OHLCV format
            candles = []
            if isinstance(data, list):
                for candle in data:
                    try:
                        # Hyperliquid returns: [timestamp, open, high, low, close, volume]
                        if isinstance(candle, dict):
                            candles.append({
                                "timestamp": candle.get("t", 0),
                                "open": float(candle.get("o", 0)),
                                "high": float(candle.get("h", 0)),
                                "low": float(candle.get("l", 0)),
                                "close": float(candle.get("c", 0)),
                                "volume": float(candle.get("v", 0))
                            })
                        elif isinstance(candle, list) and len(candle) >= 6:
                            candles.append({
                                "timestamp": candle[0],
                                "open": float(candle[1]),
                                "high": float(candle[2]),
                                "low": float(candle[3]),
                                "close": float(candle[4]),
                                "volume": float(candle[5])
                            })
                    except (ValueError, TypeError, IndexError) as e:
                        logging.debug(f"Error parsing candle: {e}")
                        continue
            
            logging.debug(f"Fetched {len(candles)} candles for {asset} ({interval})")
            return candles
            
        except Exception as e:
            logging.error(f"Error fetching candles for {asset}: {e}")
            return []
