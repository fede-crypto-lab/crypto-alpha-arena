"""Discover the coins that can actually be carried, and load them as pairs.

Two filters decide membership, and both are liquidity filters wearing different
hats:

* **Open interest on the perp leg.** A rich funding rate on a perp with $200k of
  open interest is not an opportunity, it is a warning: the rate is high precisely
  because nobody will take the other side, and the size that clears it is smaller
  than the position you wanted.
* **A spot market on the hedge venue.** Without it there is no cash leg, and the
  trade degenerates into the perp-perp spread that the single-pair backtest
  already showed does not pay.

The universe is deliberately capped. Loading a pair costs tens of paginated
requests for the hourly marks, and a book that holds five slots does not need two
hundred candidates to choose from.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .dataset import Pair, load_pair
from .venues import Venue, _http_json, get_venue

logger = logging.getLogger(__name__)


def hyperliquid_open_interest() -> Dict[str, float]:
    """Open interest in USD per coin, from the venue's own context snapshot."""
    meta, ctxs = _http_json(
        "https://api.hyperliquid.xyz/info",
        method="POST",
        payload={"type": "metaAndAssetCtxs"},
        cache=False,  # a stale OI snapshot would silently freeze the universe
    )
    out: Dict[str, float] = {}
    for asset, ctx in zip(meta["universe"], ctxs):
        try:
            out[asset["name"]] = float(ctx.get("openInterest", 0)) * float(ctx.get("markPx") or 0)
        except (TypeError, ValueError):
            continue
    return out


def okx_spot_coins() -> set:
    """Coins with a USDT spot market on OKX."""
    rows = _http_json("https://www.okx.com/api/v5/public/instruments?instType=SPOT")["data"]
    return {r["instId"].split("-")[0] for r in rows if r["instId"].endswith("-USDT")}


def discover(min_open_interest: float = 5e6, limit: int = 24) -> List[str]:
    """Coins ranked by open interest that have both a perp and a spot hedge."""
    oi = hyperliquid_open_interest()
    spot = okx_spot_coins()
    eligible = [(c, v) for c, v in oi.items() if v >= min_open_interest and c in spot]
    eligible.sort(key=lambda kv: -kv[1])
    chosen = [c for c, _ in eligible[:limit]]
    logger.info("universe: %d eligible, taking %d", len(eligible), len(chosen))
    return chosen


def load_universe(
    coins: List[str],
    spot_venue: Venue,
    perp_venue: Venue,
    start_ms: int,
    end_ms: int,
    marks_spot: Optional[Venue] = None,
    marks_perp: Optional[Venue] = None,
) -> Dict[str, Pair]:
    """Load every coin as a spot/perp pair, skipping the ones that fail.

    A coin missing history on either venue is dropped rather than patched: a pair
    with a hole in its marks would silently mis-price the hedge for the hours it
    cannot see.
    """
    universe: Dict[str, Pair] = {}
    for coin in coins:
        try:
            universe[coin] = load_pair(spot_venue, perp_venue, coin, start_ms, end_ms,
                                       marks_a=marks_spot, marks_b=marks_perp)
        except Exception as exc:  # noqa: BLE001 - one bad listing must not stop the run
            logger.warning("skipping %s: %s", coin, exc)
    if not universe:
        raise ValueError("no coin in the universe could be loaded")
    return universe


def default_universe(
    start_ms: int,
    end_ms: int,
    spot: str = "okx_spot",
    perp: str = "hyperliquid",
    min_open_interest: float = 5e6,
    limit: int = 24,
    coins: Optional[List[str]] = None,
    marks_spot: Optional[str] = None,
    marks_perp: Optional[str] = None,
) -> Dict[str, Pair]:
    chosen = coins or discover(min_open_interest=min_open_interest, limit=limit)
    return load_universe(
        chosen, get_venue(spot), get_venue(perp), start_ms, end_ms,
        marks_spot=get_venue(marks_spot) if marks_spot else None,
        marks_perp=get_venue(marks_perp) if marks_perp else None,
    )
