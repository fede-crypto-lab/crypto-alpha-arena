"""Client helper for interacting with the TAAPI technical analysis API."""

import requests
import os
import time
import logging
from src.backend.config_loader import CONFIG
from src.backend.indicators.taapi_cache import get_cache


class TAAPIClient:
    """Fetches TA indicators with retry/backoff semantics for resilience."""

    # Rate limit for free plan (1 request per 15 seconds)
    FREE_PLAN_RATE_LIMIT = 15

    def __init__(self, enable_cache: bool = True, cache_ttl: int = 60):
        """
        Initialize TAAPI credentials and base URL.

        Args:
            enable_cache: Enable caching to reduce API calls
            cache_ttl: Cache time-to-live in seconds (default: 60s)
        """
        self.api_key = CONFIG["taapi_api_key"]
        self.base_url = "https://api.taapi.io/"
        self.bulk_url = "https://api.taapi.io/bulk"
        self.enable_cache = enable_cache
        self.cache = get_cache(ttl=cache_ttl) if enable_cache else None
        # Check if using paid plan (no rate limits, more indicators)
        self.is_paid = CONFIG.get("taapi_plan", "free").lower() == "paid"
        if self.is_paid:
            logging.info("TAAPI: Using PAID plan (no rate limits, all coins, extended indicators)")
        else:
            logging.info("TAAPI: Using FREE plan (15s rate limit, BTC/ETH only)")

    def _get_with_retry(self, url, params, retries=3, backoff=0.5):
        """Perform a GET request with exponential backoff retry logic."""
        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                # Retry on rate limit (429) or server errors (500+)
                if (e.response.status_code == 429 or e.response.status_code >= 500) and attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    if e.response.status_code == 429:
                        logging.warning(f"TAAPI rate limit (429) hit, retrying in {wait}s (attempt {attempt + 1}/{retries})")
                    else:
                        logging.warning(f"TAAPI {e.response.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
            except requests.Timeout as e:
                if attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    logging.warning(f"TAAPI timeout, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    def _post_with_retry(self, url, payload, retries=3, backoff=0.5):
        """Perform a POST request with exponential backoff retry logic."""
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=payload, timeout=15)
                resp.raise_for_status()
                return resp.json()
            except requests.HTTPError as e:
                # Retry on rate limit (429) or server errors (500+)
                if (e.response.status_code == 429 or e.response.status_code >= 500) and attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    if e.response.status_code == 429:
                        logging.warning(f"TAAPI Bulk rate limit (429), retrying in {wait}s (attempt {attempt + 1}/{retries})")
                    else:
                        logging.warning(f"TAAPI Bulk {e.response.status_code}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
            except requests.Timeout as e:
                if attempt < retries - 1:
                    wait = backoff * (2 ** attempt)
                    logging.warning(f"TAAPI Bulk timeout, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max retries exceeded")

    def fetch_bulk_indicators(self, symbol, interval, indicators_config):
        """
        Fetch multiple indicators in one bulk request to TAAPI.

        Args:
            symbol: Market pair (e.g., "BTC/USDT")
            interval: Timeframe (e.g., "5m", "4h")
            indicators_config: List of indicator configurations
                Example: [
                    {"id": "ema20", "indicator": "ema", "period": 20, "results": 10},
                    {"id": "macd", "indicator": "macd", "results": 10},
                    {"id": "rsi14", "indicator": "rsi", "period": 14, "results": 10}
                ]

        Returns:
            Dict mapping indicator IDs to their results
            Example: {"ema20": [values...], "macd": {...}, "rsi14": [values...]}
        """
        try:
            # Build bulk request payload
            indicators = []
            for config in indicators_config:
                indicator_def = {
                    "id": config.get("id", config["indicator"]),
                    "indicator": config["indicator"]
                }

                # Add optional parameters
                if "period" in config:
                    indicator_def["period"] = config["period"]
                if "results" in config:
                    indicator_def["results"] = config["results"]
                if "backtrack" in config:
                    indicator_def["backtrack"] = config["backtrack"]

                indicators.append(indicator_def)

            payload = {
                "secret": self.api_key,
                "construct": {
                    "exchange": "binance",
                    "symbol": symbol,
                    "interval": interval,
                    "indicators": indicators
                }
            }

            # Make bulk POST request
            response = self._post_with_retry(self.bulk_url, payload)

            # Parse results by ID
            results = {}
            if isinstance(response, dict) and "data" in response:
                for item in response["data"]:
                    indicator_id = item.get("id")
                    if indicator_id:
                        results[indicator_id] = item.get("result")

            return results

        except Exception as e:
            logging.error(f"TAAPI bulk fetch exception for {symbol} {interval}: {e}")
            return {}

    def fetch_asset_indicators(self, asset):
        """
        Fetch all required indicators for an asset using bulk requests.
        Makes 2 requests total (5m + 4h) instead of 10 individual requests.
        
        IMPORTANT: Free plan has 1 request per 15 seconds limit.
        This method adds 15s delay between requests to respect rate limits.
        
        Caching: Results are cached for 60 seconds by default to reduce API calls.

        Args:
            asset: Asset ticker (e.g., "BTC", "ETH")

        Returns:
            Dict with structure:
            {
                "5m": {"ema20": [...], "macd": [...], "rsi7": [...], "rsi14": [...]},
                "4h": {"ema20": value, "ema50": value, "atr3": value, "atr14": value,
                       "macd": [...], "rsi14": [...]}
            }
        """
        # Check cache first (for current interval from config)
        interval = CONFIG.get("interval", "1h")
        
        # Try to get cached data for both intervals
        if self.enable_cache and self.cache:
            cached_5m = self.cache.get(asset, "5m")
            cached_interval = self.cache.get(asset, interval)
            
            if cached_5m and cached_interval:
                logging.info(f"Using cached indicators for {asset} (5m + {interval})")
                return {"5m": cached_5m, interval: cached_interval}
        
        symbol = f"{asset}/USDT"
        result = {"5m": {}, interval: {}}

        # Bulk request for 5m indicators
        # Note: Free plan limit is 20 calculations per request
        # 4 indicators × 5 results = 20 calculations (at limit)
        indicators_5m = [
            {"id": "ema20", "indicator": "ema", "period": 20, "results": 5},
            {"id": "macd", "indicator": "macd", "results": 5},
            {"id": "rsi7", "indicator": "rsi", "period": 7, "results": 5},
            {"id": "rsi14", "indicator": "rsi", "period": 14, "results": 5}
        ]

        bulk_5m = self.fetch_bulk_indicators(symbol, "5m", indicators_5m)

        # Extract series data from bulk response
        result["5m"]["ema20"] = self._extract_series(bulk_5m.get("ema20"), "value")
        result["5m"]["macd"] = self._extract_series(bulk_5m.get("macd"), "valueMACD")
        result["5m"]["rsi7"] = self._extract_series(bulk_5m.get("rsi7"), "value")
        result["5m"]["rsi14"] = self._extract_series(bulk_5m.get("rsi14"), "value")

        # Wait for rate limit only on free plan
        if not self.is_paid:
            logging.info(f"Waiting {self.FREE_PLAN_RATE_LIMIT}s for TAAPI rate limit (Free plan)...")
            time.sleep(self.FREE_PLAN_RATE_LIMIT)
        else:
            time.sleep(0.5)  # Small delay to avoid hammering API

        # Bulk request for higher timeframe indicators
        # Paid plan: Extended indicators (Bollinger, Supertrend, Stoch RSI, ADX, OBV)
        if self.is_paid:
            indicators_ht = [
                {"id": "ema20", "indicator": "ema", "period": 20},
                {"id": "ema50", "indicator": "ema", "period": 50},
                {"id": "atr3", "indicator": "atr", "period": 3},
                {"id": "atr14", "indicator": "atr", "period": 14},
                {"id": "macd", "indicator": "macd", "results": 5},
                {"id": "rsi14", "indicator": "rsi", "period": 14, "results": 5},
                # Extended indicators for paid plan
                {"id": "bbands", "indicator": "bbands", "period": 20},
                {"id": "supertrend", "indicator": "supertrend", "period": 10, "multiplier": 3},
                {"id": "stochrsi", "indicator": "stochrsi", "period": 14},
                {"id": "adx", "indicator": "adx", "period": 14},
                {"id": "obv", "indicator": "obv"},
            ]
        else:
            # Free plan: Basic indicators only
            indicators_ht = [
                {"id": "ema20", "indicator": "ema", "period": 20},
                {"id": "ema50", "indicator": "ema", "period": 50},
                {"id": "atr3", "indicator": "atr", "period": 3},
                {"id": "atr14", "indicator": "atr", "period": 14},
                {"id": "macd", "indicator": "macd", "results": 5},
                {"id": "rsi14", "indicator": "rsi", "period": 14, "results": 5}
            ]

        bulk_ht = self.fetch_bulk_indicators(symbol, interval, indicators_ht)

        # Extract values and series - basic indicators
        result[interval]["ema20"] = self._extract_value(bulk_ht.get("ema20"))
        result[interval]["ema50"] = self._extract_value(bulk_ht.get("ema50"))
        result[interval]["atr3"] = self._extract_value(bulk_ht.get("atr3"))
        result[interval]["atr14"] = self._extract_value(bulk_ht.get("atr14"))
        result[interval]["macd"] = self._extract_series(bulk_ht.get("macd"), "valueMACD")
        result[interval]["rsi14"] = self._extract_series(bulk_ht.get("rsi14"), "value")

        # Extract extended indicators (paid plan only)
        if self.is_paid:
            # Bollinger Bands: {valueUpperBand, valueMiddleBand, valueLowerBand}
            bbands = bulk_ht.get("bbands", {})
            if isinstance(bbands, dict):
                result[interval]["bbands"] = {
                    "upper": self._extract_value(bbands, "valueUpperBand"),
                    "middle": self._extract_value(bbands, "valueMiddleBand"),
                    "lower": self._extract_value(bbands, "valueLowerBand"),
                }
            # Supertrend: {value, valueSupertrend, valueAdvice}
            supertrend = bulk_ht.get("supertrend", {})
            if isinstance(supertrend, dict):
                result[interval]["supertrend"] = {
                    "value": self._extract_value(supertrend, "valueSupertrend"),
                    "advice": supertrend.get("valueAdvice"),  # "buy" or "sell"
                }
            # Stochastic RSI: {valueFastK, valueFastD}
            stochrsi = bulk_ht.get("stochrsi", {})
            if isinstance(stochrsi, dict):
                result[interval]["stochrsi"] = {
                    "k": self._extract_value(stochrsi, "valueFastK"),
                    "d": self._extract_value(stochrsi, "valueFastD"),
                }
            # ADX: {value}
            result[interval]["adx"] = self._extract_value(bulk_ht.get("adx"))
            # OBV: {value}
            result[interval]["obv"] = self._extract_value(bulk_ht.get("obv"))

        # Cache the results
        if self.enable_cache and self.cache:
            self.cache.set(asset, "5m", result["5m"])
            self.cache.set(asset, interval, result[interval])
            logging.info(f"Cached indicators for {asset} (5m + {interval})")

        return result

    def _extract_series(self, data, value_key="value"):
        """Extract and normalize series data from TAAPI response."""
        if not data:
            return []
        if isinstance(data, dict) and value_key in data:
            values = data[value_key]
            if isinstance(values, list):
                return [round(v, 4) if isinstance(v, (int, float)) else v for v in values]
        return []

    def _extract_value(self, data, value_key="value"):
        """Extract and normalize single value from TAAPI response."""
        if not data:
            return None
        if isinstance(data, dict) and value_key in data:
            val = data[value_key]
            return round(val, 4) if isinstance(val, (int, float)) else val
        return None

    def get_indicators(self, asset, interval):
        """Return a curated bundle of intraday indicators for ``asset``."""
        params = {
            "secret": self.api_key,
            "exchange": "binance",
            "symbol": f"{asset}/USDT",
            "interval": interval
        }
        rsi_response = self._get_with_retry(f"{self.base_url}rsi", params)
        macd_response = self._get_with_retry(f"{self.base_url}macd", params)
        sma_response = self._get_with_retry(f"{self.base_url}sma", params)
        ema_response = self._get_with_retry(f"{self.base_url}ema", params)
        bbands_response = self._get_with_retry(f"{self.base_url}bbands", params)
        return {
            "rsi": rsi_response.get("value"),
            "macd": macd_response,
            "sma": sma_response.get("value"),
            "ema": ema_response.get("value"),
            "bbands": bbands_response
        }

    def get_historical_indicator(self, indicator, symbol, interval, results=10, params=None):
        """Fetch historical indicator data with optional overrides."""
        base_params = {
            "secret": self.api_key,
            "exchange": "binance",
            "symbol": symbol,
            "interval": interval,
            "results": results
        }
        if params:
            base_params.update(params)
        response = self._get_with_retry(f"{self.base_url}{indicator}", base_params)
        return response

    def fetch_series(self, indicator: str, symbol: str, interval: str, results: int = 10, params: dict | None = None, value_key: str = "value") -> list:
        """Fetch and normalize a historical indicator series.

        Args:
            indicator: TAAPI indicator slug (e.g. ``"ema"``).
            symbol: Market pair identifier (e.g. ``"BTC/USDT"``).
            interval: Candle interval requested from TAAPI.
            results: Number of datapoints to request.
            params: Additional TAAPI query parameters.
            value_key: Key to extract from the TAAPI response payload.

        Returns:
            List of floats rounded to 4 decimals, or an empty list on error.
        """
        try:
            data = self.get_historical_indicator(indicator, symbol, interval, results=results, params=params)
            if isinstance(data, dict):
                # Simple indicators: {"value": [1,2,3]}
                if value_key in data and isinstance(data[value_key], list):
                    return [round(v, 4) if isinstance(v, (int, float)) else v for v in data[value_key]]
                # Error response
                if "error" in data:
                    import logging
                    logging.error(f"TAAPI error for {indicator} {symbol} {interval}: {data.get('error')}")
                    return []
            return []
        except Exception as e:
            import logging
            logging.error(f"TAAPI fetch_series exception for {indicator}: {e}")
            return []

    def fetch_value(self, indicator: str, symbol: str, interval: str, params: dict | None = None, key: str = "value"):
        """Fetch a single indicator value for the latest candle."""
        try:
            base_params = {
                "secret": self.api_key,
                "exchange": "binance",
                "symbol": symbol,
                "interval": interval
            }
            if params:
                base_params.update(params)
            data = self._get_with_retry(f"{self.base_url}{indicator}", base_params)
            if isinstance(data, dict):
                val = data.get(key)
                return round(val, 4) if isinstance(val, (int, float)) else val
            return None
        except Exception:
            return None
