"""Abstract base class for exchange implementations.

This module defines the interface that all exchange implementations must follow,
enabling modular exchange support (Hyperliquid, Bybit, etc.) via the EXCHANGE env var.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseExchange(ABC):
    """Abstract base class for exchange implementations.

    All exchange implementations (HyperliquidAPI, BybitAPI, etc.) must inherit
    from this class and implement all abstract methods.
    """

    @abstractmethod
    async def place_buy_order(self, asset: str, amount: float, slippage: float = 0.01) -> Dict[str, Any]:
        """Submit a market buy order (open long position).

        Args:
            asset: Market symbol (e.g., "BTC", "ETH")
            amount: Contract size to open
            slippage: Maximum acceptable slippage as decimal (default 1%)

        Returns:
            Raw exchange response dict
        """
        pass

    @abstractmethod
    async def place_sell_order(self, asset: str, amount: float, slippage: float = 0.01) -> Dict[str, Any]:
        """Submit a market sell order (open short position).

        Args:
            asset: Market symbol (e.g., "BTC", "ETH")
            amount: Contract size to open
            slippage: Maximum acceptable slippage as decimal (default 1%)

        Returns:
            Raw exchange response dict
        """
        pass

    @abstractmethod
    async def place_take_profit(self, asset: str, is_buy: bool, amount: float, tp_price: float) -> Dict[str, Any]:
        """Place a take-profit order.

        Args:
            asset: Market symbol
            is_buy: True if the original position is long
            amount: Contract size to close
            tp_price: Trigger price for take-profit

        Returns:
            Dict with keys: 'success', 'oid', 'error', 'raw_response'
        """
        pass

    @abstractmethod
    async def place_stop_loss(self, asset: str, is_buy: bool, amount: float, sl_price: float) -> Dict[str, Any]:
        """Place a stop-loss order.

        Args:
            asset: Market symbol
            is_buy: True if the original position is long
            amount: Contract size to close
            sl_price: Trigger price for stop-loss

        Returns:
            Dict with keys: 'success', 'oid', 'error', 'raw_response'
        """
        pass

    @abstractmethod
    async def cancel_order(self, asset: str, oid: Any) -> Dict[str, Any]:
        """Cancel a specific order.

        Args:
            asset: Market symbol
            oid: Order identifier (type varies by exchange)

        Returns:
            Exchange response dict
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self, asset: str) -> Dict[str, Any]:
        """Cancel all open orders for an asset.

        Args:
            asset: Market symbol

        Returns:
            Dict with 'status' and 'cancelled_count' or 'message'
        """
        pass

    @abstractmethod
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders for the account.

        Returns:
            List of order dicts with normalized fields
        """
        pass

    @abstractmethod
    async def get_user_state(self) -> Dict[str, Any]:
        """Get account state including balance and positions.

        Returns:
            Dict with:
                - 'balance': float - Available balance
                - 'total_value': float - Total account value
                - 'positions': List[Dict] - Open positions with PnL
        """
        pass

    @abstractmethod
    async def get_current_price(self, asset: str) -> float:
        """Get current mid-price for an asset.

        Args:
            asset: Market symbol

        Returns:
            Current price as float, or 0.0 if unavailable
        """
        pass

    @abstractmethod
    async def get_candles(self, asset: str, interval: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch OHLCV candle data.

        Args:
            asset: Market symbol
            interval: Candle interval ("1m", "5m", "15m", "1h", "4h", "1d")
            limit: Number of candles to fetch

        Returns:
            List of candle dicts with keys: timestamp, open, high, low, close, volume
        """
        pass

    @abstractmethod
    def round_size(self, asset: str, amount: float) -> float:
        """Round order size to exchange precision.

        Args:
            asset: Market symbol
            amount: Raw amount before rounding

        Returns:
            Amount rounded to exchange's size precision
        """
        pass

    @abstractmethod
    def round_price(self, asset: str, price: float) -> float:
        """Round price to exchange tick size.

        Args:
            asset: Market symbol
            price: Raw price before rounding

        Returns:
            Price rounded to exchange's tick size
        """
        pass

    @abstractmethod
    def get_tick_size(self, asset: str) -> float:
        """Get the tick size (minimum price increment) for an asset.

        Args:
            asset: Market symbol

        Returns:
            Tick size as float
        """
        pass

    @abstractmethod
    def extract_oids(self, order_result: Dict[str, Any]) -> List[Any]:
        """Extract order IDs from an exchange response.

        Args:
            order_result: Raw order response from exchange

        Returns:
            List of order identifiers
        """
        pass

    @abstractmethod
    async def get_meta_and_ctxs(self) -> Any:
        """Get market metadata and context information.

        Returns:
            Exchange-specific metadata (cached)
        """
        pass

    # Optional methods with default implementations
    async def get_funding_rate(self, asset: str) -> Optional[float]:
        """Get funding rate for an asset (optional).

        Args:
            asset: Market symbol

        Returns:
            Funding rate as float, or None if not available
        """
        return None

    async def get_open_interest(self, asset: str) -> Optional[float]:
        """Get open interest for an asset (optional).

        Args:
            asset: Market symbol

        Returns:
            Open interest as float, or None if not available
        """
        return None

    async def get_recent_fills(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent fills/trades (optional).

        Args:
            limit: Maximum number of fills to return

        Returns:
            List of fill dicts, or empty list if not available
        """
        return []
