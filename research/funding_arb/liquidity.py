"""Measure real execution cost by walking live order books.

The backtest's one unmeasurable input was slippage. Historical candles cannot
give it: a 1h bar says where the price went, not what it would have cost to move
$10k through the book at that moment. So the backtests carried it as an
assumption (2bp optimistic, 10bp pessimistic) and the whole verdict swung on it -
at 2bp every swept configuration was profitable, at 10bp three went negative.

This module removes the assumption. Both venues publish their full L2 book for
free and without authentication, so the cost of a given size is not an estimate:
it is arithmetic on the levels that are standing there right now.

Two things worth being clear about:

* **Testnet cannot answer this.** A testnet book is a handful of synthetic orders;
  walking it measures the testnet, not the market. Mainnet's *public* book is the
  honest source, and reading it risks nothing.
* **One snapshot is not the answer either.** Depth at 3am on a Sunday and depth
  during a liquidation cascade differ by an order of magnitude, and a carry gets
  closed precisely during the second kind of moment. `sample_to_file` exists so
  snapshots accumulate into a distribution; the number to plan against is a bad
  percentile, not the median.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .venues import Venue, _http_json

logger = logging.getLogger(__name__)

Level = Tuple[float, float]  # (price, size in base units)


@dataclass
class Fill:
    """What it would cost to execute `notional` on one side of one book."""

    mid: float
    vwap: float
    slippage_bps: float
    #: Notional actually available on that side of the book.
    depth_usd: float
    #: False when the book could not absorb the order - slippage is then a
    #: lower bound, not a measurement.
    filled: bool

    @property
    def usable(self) -> bool:
        return self.filled


def walk(levels: Sequence[Level], notional_usd: float, mid: float,
         is_buy: bool) -> Fill:
    """Consume the book until `notional_usd` is filled; report the VWAP cost.

    Slippage is measured against the mid, not the touch, because the half-spread
    is a real cost of crossing and pretending otherwise flatters every taker
    strategy.
    """
    remaining = notional_usd
    spent = 0.0
    base = 0.0
    depth = sum(px * sz for px, sz in levels)

    for px, sz in levels:
        level_usd = px * sz
        take = min(level_usd, remaining)
        if take <= 0:
            break
        spent += take
        base += take / px
        remaining -= take
        if remaining <= 0:
            break

    if base == 0:
        return Fill(mid=mid, vwap=mid, slippage_bps=0.0, depth_usd=depth, filled=False)

    vwap = spent / base
    # Buying above the mid and selling below it are both costs, hence the sign flip.
    slip = (vwap / mid - 1.0) if is_buy else (1.0 - vwap / mid)
    return Fill(
        mid=mid,
        vwap=vwap,
        slippage_bps=slip * 10_000.0,
        depth_usd=depth,
        filled=remaining <= 0,
    )


# --------------------------------------------------------------------------
# Book fetchers - public endpoints, never cached (a stale book is a lie)
# --------------------------------------------------------------------------

#: Hyperliquid's l2Book serves 20 levels a side and no more, so that is a real
#: limit on what can be measured there, not a truncation we choose. OKX serves up
#: to 400, and its spot books are finely priced - taking only the top 20 there
#: reads a sliver of the book and reports slippage several times the truth.
HL_MAX_LEVELS = 20
OKX_MAX_LEVELS = 400


def hyperliquid_book(coin: str, depth: int = HL_MAX_LEVELS) -> Tuple[List[Level], List[Level]]:
    payload = _http_json(
        "https://api.hyperliquid.xyz/info",
        method="POST",
        payload={"type": "l2Book", "coin": coin.upper()},
        cache=False,
    )
    bids_raw, asks_raw = payload["levels"][0], payload["levels"][1]
    to_levels = lambda rows: [(float(r["px"]), float(r["sz"])) for r in rows[:depth]]
    return to_levels(bids_raw), to_levels(asks_raw)


def okx_book(inst_id: str, depth: int = OKX_MAX_LEVELS) -> Tuple[List[Level], List[Level]]:
    q = urllib.parse.urlencode({"instId": inst_id, "sz": min(depth, OKX_MAX_LEVELS)})
    payload = _http_json(f"https://www.okx.com/api/v5/market/books?{q}", cache=False)
    if payload.get("code") not in ("0", 0):
        raise RuntimeError(f"OKX book error {payload.get('code')}: {payload.get('msg')}")
    row = payload["data"][0]
    to_levels = lambda rows: [(float(r[0]), float(r[1])) for r in rows[:depth]]
    return to_levels(row["bids"]), to_levels(row["asks"])


def fetch_book(venue: Venue, coin: str, depth: int = OKX_MAX_LEVELS):
    """Dispatch to the right endpoint, clamping to what each venue actually serves."""
    name = venue.name
    if name == "hyperliquid":
        return hyperliquid_book(coin, min(depth, HL_MAX_LEVELS))
    if name in ("okx", "okx_spot"):
        return okx_book(venue.symbol(coin), depth)
    raise NotImplementedError(f"no L2 book reader for venue '{name}'")


# --------------------------------------------------------------------------
# Round-trip cost of one carry
# --------------------------------------------------------------------------

@dataclass
class CarryLiquidity:
    coin: str
    notional: float
    #: Sum of the four crossings' slippage, in bp of one leg's notional.
    round_trip_slippage_bps: float
    spot_buy_bps: float
    spot_sell_bps: float
    perp_sell_bps: float
    perp_buy_bps: float
    spot_depth_usd: float
    perp_depth_usd: float
    #: False if any of the four sides could not absorb the size.
    complete: bool
    timestamp_ms: int


def measure_carry(spot_venue: Venue, perp_venue: Venue, coin: str,
                  notional: float, depth: int = OKX_MAX_LEVELS) -> CarryLiquidity:
    """Cost of opening and closing one spot/perp carry, at current depth.

    Four crossings: buy spot and sell perp to open, sell spot and buy perp to
    close. Every one of them pays the half-spread plus impact, which is why a
    carry's break-even holding period is measured in weeks rather than days.
    """
    spot_bids, spot_asks = fetch_book(spot_venue, coin, depth)
    perp_bids, perp_asks = fetch_book(perp_venue, coin, depth)

    spot_mid = (spot_bids[0][0] + spot_asks[0][0]) / 2
    perp_mid = (perp_bids[0][0] + perp_asks[0][0]) / 2

    buy_spot = walk(spot_asks, notional, spot_mid, is_buy=True)
    sell_spot = walk(spot_bids, notional, spot_mid, is_buy=False)
    sell_perp = walk(perp_bids, notional, perp_mid, is_buy=False)
    buy_perp = walk(perp_asks, notional, perp_mid, is_buy=True)

    return CarryLiquidity(
        coin=coin,
        notional=notional,
        round_trip_slippage_bps=(
            buy_spot.slippage_bps + sell_spot.slippage_bps
            + sell_perp.slippage_bps + buy_perp.slippage_bps
        ),
        spot_buy_bps=buy_spot.slippage_bps,
        spot_sell_bps=sell_spot.slippage_bps,
        perp_sell_bps=sell_perp.slippage_bps,
        perp_buy_bps=buy_perp.slippage_bps,
        spot_depth_usd=min(buy_spot.depth_usd, sell_spot.depth_usd),
        perp_depth_usd=min(sell_perp.depth_usd, buy_perp.depth_usd),
        complete=all(f.filled for f in (buy_spot, sell_spot, sell_perp, buy_perp)),
        timestamp_ms=int(time.time() * 1000),
    )


def scan(spot_venue: Venue, perp_venue: Venue, coins: Sequence[str],
         notional: float, depth: int = OKX_MAX_LEVELS) -> List[CarryLiquidity]:
    out: List[CarryLiquidity] = []
    for coin in coins:
        try:
            out.append(measure_carry(spot_venue, perp_venue, coin, notional, depth))
        except Exception as exc:  # noqa: BLE001 - a missing listing must not stop the scan
            logger.warning("liquidity scan failed for %s: %s", coin, exc)
    return out


def sample_to_file(path: str, rows: Sequence[CarryLiquidity]) -> None:
    """Append one snapshot per coin as JSON lines.

    Appending rather than overwriting is the point: run this on a schedule and
    the file becomes the distribution the single snapshot cannot be.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "a") as fh:
        for row in rows:
            fh.write(json.dumps(asdict(row)) + "\n")


def load_samples(path: str) -> Dict[str, List[float]]:
    """Round-trip slippage per coin, from an accumulated sample file."""
    per_coin: Dict[str, List[float]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("complete"):
                per_coin.setdefault(row["coin"], []).append(row["round_trip_slippage_bps"])
    return per_coin


def format_scan(rows: Sequence[CarryLiquidity], assumed_bps: Optional[float] = None) -> str:
    """Table of measured cost against the figure the backtest assumed."""
    if not rows:
        return "no liquidity measured"

    rows = sorted(rows, key=lambda r: r.round_trip_slippage_bps)
    notional = rows[0].notional
    lines = [
        "=" * 76,
        f" MEASURED EXECUTION COST   ${notional:,.0f} per leg, live books",
        "=" * 76,
        "",
        f"{'coin':<9}{'round trip':>12}{'spot buy':>10}{'spot sell':>11}"
        f"{'perp sell':>11}{'perp buy':>10}{'book depth':>12}",
        "-" * 76,
    ]
    for r in rows:
        flag = "" if r.complete else "  <- book too thin for this size"
        depth = min(r.spot_depth_usd, r.perp_depth_usd)
        lines.append(
            f"{r.coin:<9}{r.round_trip_slippage_bps:>11.1f}b{r.spot_buy_bps:>9.1f}b"
            f"{r.spot_sell_bps:>10.1f}b{r.perp_sell_bps:>10.1f}b{r.perp_buy_bps:>9.1f}b"
            f"{depth / 1000:>10.0f}k{flag}"
        )

    complete = [r for r in rows if r.complete]
    if complete:
        values = sorted(r.round_trip_slippage_bps for r in complete)
        median = values[len(values) // 2]
        lines += [
            "-" * 76,
            f" median round-trip slippage: {median:.1f}bp"
            f"   (per-order equivalent: {median / 4:.1f}bp)",
        ]
        if assumed_bps is not None:
            verdict = "OPTIMISTIC" if median / 4 > assumed_bps else "conservative"
            lines.append(
                f" backtest assumed {assumed_bps:.1f}bp per order -> that assumption is {verdict}"
            )
        lines.append(
            " NOTE: one snapshot. Depth collapses exactly when a carry needs to exit,"
        )
        lines.append(
            "       so plan against a bad percentile, not this median. Use --sample-to."
        )
    lines.append("=" * 76)
    return "\n".join(lines)
