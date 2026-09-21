"""Venue adapters: historical funding rates and hourly marks from public APIs.

No API keys are required by anything in this module - every endpoint used here is
a public market-data endpoint. Responses are cached on disk so that re-running a
backtest does not re-hammer the exchanges.

Funding intervals differ across venues (Hyperliquid and dYdX settle hourly, most
CEX perps settle every 8h). Rates are therefore stored raw, *per interval*, and
normalisation is the caller's job - see `dataset.annualize`. Mixing a 1h rate with
an 8h rate without normalising inflates the apparent spread by 8x, which is the
single easiest way to backtest a strategy that does not exist.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
HOURS_PER_YEAR = 24 * 365  # 8760


@dataclass(frozen=True)
class FundingPoint:
    """One funding settlement: the rate actually applied at `time_ms`."""

    time_ms: int
    rate: float  # fraction per interval, e.g. 0.0001 == 1bp per interval


@dataclass(frozen=True)
class Mark:
    """Hourly close price, used to measure basis drift between the two legs."""

    time_ms: int
    close: float


# --------------------------------------------------------------------------
# HTTP with retry + disk cache
# --------------------------------------------------------------------------

class VenueError(RuntimeError):
    """Raised when a venue is unreachable or returns an unusable payload."""


def _cache_path(key: str) -> str:
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{digest}.json")


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    cache: bool = True,
    retries: int = 4,
    timeout: int = 30,
) -> Any:
    key = f"{method} {url} {json.dumps(payload, sort_keys=True) if payload else ''}"
    path = _cache_path(key)

    if cache and os.path.exists(path):
        with open(path, "r") as fh:
            return json.load(fh)

    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json", "User-Agent": "funding-arb-research/1.0"}

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode()
            data = json.loads(raw)
            if cache:
                os.makedirs(CACHE_DIR, exist_ok=True)
                with open(path, "w") as fh:
                    json.dump(data, fh)
            return data
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning("%s %s failed (%s), retry in %ss", method, url, exc, wait)
            time.sleep(wait)

    raise VenueError(f"{method} {url} failed after {retries} attempts: {last_exc}")


# --------------------------------------------------------------------------
# Venue interface
# --------------------------------------------------------------------------

class Venue(ABC):
    """A perpetual-futures venue we can read funding and marks from."""

    name: str = "abstract"
    #: Nominal settlement interval. Verified against the data by `detect_interval_hours`.
    funding_interval_hours: float = 8.0
    #: Taker fee in basis points of notional, one side. Defaults are public retail tiers.
    taker_fee_bps: float = 5.5
    #: True for cash instruments, which pay no funding and post full notional.
    is_spot: bool = False

    @abstractmethod
    def symbol(self, coin: str) -> str:
        """Map a bare coin ("BTC") onto this venue's instrument id."""

    @abstractmethod
    def fetch_funding(self, coin: str, start_ms: int, end_ms: int) -> List[FundingPoint]:
        """Funding settlements in [start_ms, end_ms], ascending by time."""

    @abstractmethod
    def fetch_marks(self, coin: str, start_ms: int, end_ms: int) -> List[Mark]:
        """Hourly closes in [start_ms, end_ms], ascending by time."""

    def detect_interval_hours(self, points: List[FundingPoint]) -> float:
        """Infer the settlement interval from the data itself.

        Several venues run different intervals per symbol (Bybit quotes some pairs
        at 1h or 4h rather than 8h), so trusting the declared constant silently
        mis-scales those series. The median gap is robust to the odd missing point.
        """
        if len(points) < 3:
            return self.funding_interval_hours
        gaps = [
            (b.time_ms - a.time_ms) / 3_600_000.0
            for a, b in zip(points, points[1:])
            if b.time_ms > a.time_ms
        ]
        if not gaps:
            return self.funding_interval_hours
        observed = statistics.median(gaps)
        if abs(observed - self.funding_interval_hours) > 0.1:
            logger.warning(
                "%s: declared funding interval %.1fh but observed %.1fh - using observed",
                self.name, self.funding_interval_hours, observed,
            )
        return observed


# --------------------------------------------------------------------------
# Hyperliquid - hourly funding
# --------------------------------------------------------------------------

class Hyperliquid(Venue):
    name = "hyperliquid"
    funding_interval_hours = 1.0
    taker_fee_bps = 4.5

    INFO_URL = "https://api.hyperliquid.xyz/info"
    _PAGE = 500  # server-side cap on fundingHistory rows per call

    def symbol(self, coin: str) -> str:
        return coin.upper()

    def fetch_funding(self, coin: str, start_ms: int, end_ms: int) -> List[FundingPoint]:
        out: List[FundingPoint] = []
        cursor = start_ms
        while cursor < end_ms:
            page = _http_json(
                self.INFO_URL,
                method="POST",
                payload={
                    "type": "fundingHistory",
                    "coin": self.symbol(coin),
                    "startTime": cursor,
                    "endTime": end_ms,
                },
            )
            if not page:
                break
            for row in page:
                t = int(row["time"])
                if start_ms <= t <= end_ms:
                    out.append(FundingPoint(t, float(row["fundingRate"])))
            last = int(page[-1]["time"])
            if len(page) < self._PAGE or last <= cursor:
                break
            cursor = last + 1
        return _dedupe(out)

    def fetch_marks(self, coin: str, start_ms: int, end_ms: int) -> List[Mark]:
        out: List[Mark] = []
        cursor = start_ms
        span = 4000 * 3_600_000  # stay well inside the venue's 5000-candle cap
        while cursor < end_ms:
            chunk_end = min(cursor + span, end_ms)
            page = _http_json(
                self.INFO_URL,
                method="POST",
                payload={
                    "type": "candleSnapshot",
                    "req": {
                        "coin": self.symbol(coin),
                        "interval": "1h",
                        "startTime": cursor,
                        "endTime": chunk_end,
                    },
                },
            )
            for row in page or []:
                out.append(Mark(int(row["t"]), float(row["c"])))
            if chunk_end >= end_ms:
                break
            cursor = chunk_end + 1
        return _dedupe(out)


# --------------------------------------------------------------------------
# OKX - 8h funding
# --------------------------------------------------------------------------

class OKX(Venue):
    name = "okx"
    funding_interval_hours = 8.0
    taker_fee_bps = 5.0

    BASE = "https://www.okx.com"
    _PAGE = 100

    def symbol(self, coin: str) -> str:
        return f"{coin.upper()}-USDT-SWAP"

    @staticmethod
    def _unwrap(payload: Dict[str, Any]) -> List[Any]:
        if payload.get("code") not in ("0", 0):
            raise VenueError(f"OKX error {payload.get('code')}: {payload.get('msg')}")
        return payload.get("data") or []

    def fetch_funding(self, coin: str, start_ms: int, end_ms: int) -> List[FundingPoint]:
        out: List[FundingPoint] = []
        # OKX paginates backwards: `after` returns rows strictly older than the ts.
        cursor = end_ms
        while cursor > start_ms:
            q = urllib.parse.urlencode(
                {"instId": self.symbol(coin), "after": cursor, "limit": self._PAGE}
            )
            rows = self._unwrap(_http_json(f"{self.BASE}/api/v5/public/funding-rate-history?{q}"))
            if not rows:
                break
            for row in rows:
                t = int(row["fundingTime"])
                # `realizedRate` is what was actually charged; `fundingRate` is the
                # forecast published ahead of settlement. Prefer the realised one.
                rate = float(row.get("realizedRate") or row["fundingRate"])
                if start_ms <= t <= end_ms:
                    out.append(FundingPoint(t, rate))
            oldest = min(int(r["fundingTime"]) for r in rows)
            if len(rows) < self._PAGE or oldest >= cursor:
                break
            cursor = oldest
        return _dedupe(out)

    def fetch_marks(self, coin: str, start_ms: int, end_ms: int) -> List[Mark]:
        out: List[Mark] = []
        cursor = end_ms
        while cursor > start_ms:
            q = urllib.parse.urlencode(
                {"instId": self.symbol(coin), "bar": "1H", "after": cursor, "limit": self._PAGE}
            )
            rows = self._unwrap(_http_json(f"{self.BASE}/api/v5/market/history-candles?{q}"))
            if not rows:
                break
            for row in rows:
                t = int(row[0])
                if start_ms <= t <= end_ms:
                    out.append(Mark(t, float(row[4])))
            oldest = min(int(r[0]) for r in rows)
            if len(rows) < self._PAGE or oldest >= cursor:
                break
            cursor = oldest
        return _dedupe(out)


# --------------------------------------------------------------------------
# Bybit / Binance - implemented but geo-restricted from some hosts
# --------------------------------------------------------------------------

class Bybit(Venue):
    """Bybit USDT perps.

    Note: api.bybit.com sits behind a CloudFront distribution that blocks a number
    of regions. If you get "configured to block access from your country", run the
    fetch from a host in a supported region - the adapter itself is fine.
    """

    name = "bybit"
    funding_interval_hours = 8.0
    taker_fee_bps = 5.5

    BASE = "https://api.bybit.com"
    _PAGE = 200

    def symbol(self, coin: str) -> str:
        return f"{coin.upper()}USDT"

    @staticmethod
    def _unwrap(payload: Dict[str, Any]) -> List[Any]:
        if payload.get("retCode") not in (0, "0"):
            raise VenueError(f"Bybit error {payload.get('retCode')}: {payload.get('retMsg')}")
        return (payload.get("result") or {}).get("list") or []

    def fetch_funding(self, coin: str, start_ms: int, end_ms: int) -> List[FundingPoint]:
        out: List[FundingPoint] = []
        cursor = end_ms
        while cursor > start_ms:
            q = urllib.parse.urlencode({
                "category": "linear", "symbol": self.symbol(coin),
                "startTime": start_ms, "endTime": cursor, "limit": self._PAGE,
            })
            rows = self._unwrap(_http_json(f"{self.BASE}/v5/market/funding/history?{q}"))
            if not rows:
                break
            for row in rows:
                t = int(row["fundingRateTimestamp"])
                if start_ms <= t <= end_ms:
                    out.append(FundingPoint(t, float(row["fundingRate"])))
            oldest = min(int(r["fundingRateTimestamp"]) for r in rows)
            if len(rows) < self._PAGE or oldest >= cursor:
                break
            cursor = oldest - 1
        return _dedupe(out)

    def fetch_marks(self, coin: str, start_ms: int, end_ms: int) -> List[Mark]:
        out: List[Mark] = []
        cursor = start_ms
        span = 200 * 3_600_000
        while cursor < end_ms:
            chunk_end = min(cursor + span, end_ms)
            q = urllib.parse.urlencode({
                "category": "linear", "symbol": self.symbol(coin), "interval": "60",
                "start": cursor, "end": chunk_end, "limit": 200,
            })
            rows = self._unwrap(_http_json(f"{self.BASE}/v5/market/kline?{q}"))
            for row in rows:
                out.append(Mark(int(row[0]), float(row[4])))
            cursor = chunk_end + 1
        return _dedupe(out)


class Binance(Venue):
    """Binance USD-M perps. Also geo-restricted from several regions."""

    name = "binance"
    funding_interval_hours = 8.0
    taker_fee_bps = 5.0

    BASE = "https://fapi.binance.com"
    _PAGE = 1000

    def symbol(self, coin: str) -> str:
        return f"{coin.upper()}USDT"

    def fetch_funding(self, coin: str, start_ms: int, end_ms: int) -> List[FundingPoint]:
        out: List[FundingPoint] = []
        cursor = start_ms
        while cursor < end_ms:
            q = urllib.parse.urlencode({
                "symbol": self.symbol(coin), "startTime": cursor,
                "endTime": end_ms, "limit": self._PAGE,
            })
            rows = _http_json(f"{self.BASE}/fapi/v1/fundingRate?{q}")
            if isinstance(rows, dict):
                raise VenueError(f"Binance error: {rows}")
            if not rows:
                break
            for row in rows:
                out.append(FundingPoint(int(row["fundingTime"]), float(row["fundingRate"])))
            last = int(rows[-1]["fundingTime"])
            if len(rows) < self._PAGE or last <= cursor:
                break
            cursor = last + 1
        return _dedupe(out)

    def fetch_marks(self, coin: str, start_ms: int, end_ms: int) -> List[Mark]:
        out: List[Mark] = []
        cursor = start_ms
        while cursor < end_ms:
            q = urllib.parse.urlencode({
                "symbol": self.symbol(coin), "interval": "1h",
                "startTime": cursor, "endTime": end_ms, "limit": 1000,
            })
            rows = _http_json(f"{self.BASE}/fapi/v1/klines?{q}")
            if isinstance(rows, dict):
                raise VenueError(f"Binance error: {rows}")
            if not rows:
                break
            for row in rows:
                out.append(Mark(int(row[0]), float(row[4])))
            last = int(rows[-1][0])
            if len(rows) < 1000 or last <= cursor:
                break
            cursor = last + 1
        return _dedupe(out)


REGISTRY = {v.name: v for v in (Hyperliquid(), OKX(), Bybit(), Binance())}


def get_venue(name: str) -> Venue:
    try:
        return REGISTRY[name.lower().strip()]
    except KeyError:
        raise ValueError(f"unknown venue '{name}'; known: {sorted(REGISTRY)}") from None


def _dedupe(rows: List[Any]) -> List[Any]:
    """Sort by time and drop duplicate timestamps introduced by overlapping pages."""
    seen: Dict[int, Any] = {}
    for row in rows:
        seen[row.time_ms] = row
    return [seen[t] for t in sorted(seen)]


# --------------------------------------------------------------------------
# Spot legs - the other half of a cash-and-carry
# --------------------------------------------------------------------------

class SpotLeg(Venue):
    """A spot instrument dressed up as a `Venue` with identically zero funding.

    Cross-venue carry only ever earns the *difference* between two funding rates,
    because the long leg is itself a perp that pays funding. Pairing a perp against
    spot removes that offset and earns the funding *level* instead - which in crypto
    is structurally positive, because leveraged retail is net long. That is a much
    larger and far more persistent number than any inter-venue spread.

    Modelling spot as a zero-funding venue means the existing pair engine handles
    it unchanged: `spread = perp_funding - 0`, so a positive rate puts us short the
    perp and long the coin, which is exactly the cash-and-carry.

    Spot taker fees are typically double the perp tier (10bp vs 5bp at OKX retail),
    and that asymmetry matters enough that it is carried per-venue rather than
    assumed.
    """

    funding_interval_hours = 1.0
    is_spot = True

    def fetch_funding(self, coin: str, start_ms: int, end_ms: int) -> List[FundingPoint]:
        first = ((start_ms + 3_600_000 - 1) // 3_600_000) * 3_600_000
        return [FundingPoint(t, 0.0) for t in range(first, end_ms + 1, 3_600_000)]


class OKXSpot(SpotLeg):
    name = "okx_spot"
    taker_fee_bps = 10.0

    BASE = OKX.BASE
    _PAGE = 100

    def symbol(self, coin: str) -> str:
        return f"{coin.upper()}-USDT"

    def fetch_marks(self, coin: str, start_ms: int, end_ms: int) -> List[Mark]:
        return OKX.fetch_marks(self, coin, start_ms, end_ms)  # same endpoint, spot instId

    _unwrap = staticmethod(OKX._unwrap)


class BinanceSpot(SpotLeg):
    """Binance spot. Geo-restricted from some hosts, same as the perp adapter."""

    name = "binance_spot"
    taker_fee_bps = 10.0

    BASE = "https://api.binance.com"

    def symbol(self, coin: str) -> str:
        return f"{coin.upper()}USDT"

    def fetch_marks(self, coin: str, start_ms: int, end_ms: int) -> List[Mark]:
        out: List[Mark] = []
        cursor = start_ms
        while cursor < end_ms:
            q = urllib.parse.urlencode({
                "symbol": self.symbol(coin), "interval": "1h",
                "startTime": cursor, "endTime": end_ms, "limit": 1000,
            })
            rows = _http_json(f"{self.BASE}/api/v3/klines?{q}")
            if isinstance(rows, dict):
                raise VenueError(f"Binance error: {rows}")
            if not rows:
                break
            for row in rows:
                out.append(Mark(int(row[0]), float(row[4])))
            last = int(rows[-1][0])
            if len(rows) < 1000 or last <= cursor:
                break
            cursor = last + 1
        return _dedupe(out)


REGISTRY.update({v.name: v for v in (OKXSpot(), BinanceSpot())})
