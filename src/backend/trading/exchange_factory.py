"""Factory for creating exchange instances based on configuration.

This module provides a factory function to create the appropriate exchange
client based on the EXCHANGE environment variable.
"""

import logging
from src.backend.config_loader import CONFIG
from src.backend.trading.base_exchange import BaseExchange

logger = logging.getLogger(__name__)


def create_exchange() -> BaseExchange:
    """Create and return an exchange instance based on EXCHANGE env var.

    Supported exchanges:
        - "hyperliquid" (default): Hyperliquid perpetual futures
        - "bybit": Bybit USDT perpetual futures

    Returns:
        BaseExchange: An instance of the configured exchange client

    Raises:
        ValueError: If EXCHANGE is set to an unknown value
        RuntimeError: If exchange credentials are missing or invalid
    """
    exchange_name = (CONFIG.get("exchange") or "hyperliquid").lower().strip()

    logger.info(f"Creating exchange client for: {exchange_name}")

    if exchange_name == "hyperliquid":
        from src.backend.trading.hyperliquid_api import HyperliquidAPI
        return HyperliquidAPI()

    elif exchange_name == "bybit":
        from src.backend.trading.bybit_api import BybitAPI
        return BybitAPI()

    else:
        raise ValueError(
            f"Unknown exchange: '{exchange_name}'. "
            f"Supported values: 'hyperliquid', 'bybit'"
        )


def get_exchange_name() -> str:
    """Get the configured exchange name.

    Returns:
        str: The exchange name from config (lowercase)
    """
    return (CONFIG.get("exchange") or "hyperliquid").lower().strip()
