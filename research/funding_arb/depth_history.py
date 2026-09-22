"""Historical order-book depth, from Binance's public data dumps.

`liquidity.py` measures execution cost from a live book, which answers "what does
this cost right now" and nothing about what it cost during the moments that
matter. The advice that followed - sample on a cron for a month - was wrong, or
at least unnecessary: Binance publishes daily `bookDepth` archives going back to
2023, free and unauthenticated, covering every symbol in this universe.

Each file holds a snapshot roughly twice a minute of the *cumulative notional*
standing within 1%, 2%, 3%, 4% and 5% of mid, on each side. That is exactly the
input needed to price an order: walk out until the cumulative notional covers the
size, and the average distance travelled is the slippage.

Two honest limits on this:

* **It is Binance, not Hyperliquid or OKX.** Spreads and depth correlate strongly
  across venues but are not identical, so the right use is the *shape* of the
  distribution - how much worse a bad hour is than a median one - calibrated
  against the live measurement from `liquidity.py` for the level.
* **Depth is not the whole cost.** It misses the spread inside 1% and the
  queue/latency effects of actually hitting it. For sizes that reach beyond the
  touch, which is the case this module exists for, impact dominates.
"""

from __future__ import annotations

import csv
import io
import logging
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterator, List, Optional, Sequence

logger = logging.getLogger(__name__)

BASE = "https://data.binance.vision/data/futures/um/daily/bookDepth"
#: The percentage bands Binance publishes, one side. The 0.2% band matters most
#: here: a $10k order on a liquid perp never leaves it, and treating the first
#: band as 1% wide would overstate that order's cost roughly fivefold.
BUCKETS = (0.2, 1.0, 2.0, 3.0, 4.0, 5.0)


@dataclass
class DepthSnapshot:
    """Cumulative notional standing within each percentage band, one moment."""

    timestamp: str
    #: {percent_from_mid: cumulative notional USD} for the ask side (positive).
    asks: Dict[float, float]
    #: Same for the bid side, keyed by the positive percentage.
    bids: Dict[float, float]


def archive_url(symbol: str, day: date) -> str:
    return f"{BASE}/{symbol}/{symbol}-bookDepth-{day.isoformat()}.zip"


#: A real book's cumulative notional grows as you walk out from the mid. The
#: archive occasionally carries placeholder days where every band reports the
#: same tiny figure (NEARUSDT, 2026-09-07 to 09-11, reports $13 at 0.2% and $13
#: at 5%). Those are not thin books, they are corrupt rows, and left in they
#: masquerade as liquidity crises: they alone produced a "30% of snapshots could
#: not absorb $10k" reading on a perp that trades hundreds of millions a day.
MIN_BAND_GROWTH = 1.05


def is_plausible(snapshot: "DepthSnapshot") -> bool:
    """Reject snapshots whose depth does not grow with distance from the mid."""
    for side in (snapshot.asks, snapshot.bids):
        if not side:
            return False
        bands = sorted(side)
        inner, outer = side[bands[0]], side[bands[-1]]
        if inner <= 0 or outer < inner * MIN_BAND_GROWTH:
            return False
    return True


def fetch_day(symbol: str, day: date, timeout: int = 90) -> List[DepthSnapshot]:
    """Download and parse one day. Returns [] when the archive does not exist."""
    url = archive_url(symbol, day)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            blob = resp.read()
    except Exception as exc:  # noqa: BLE001 - a missing day is normal, not fatal
        logger.info("no bookDepth for %s on %s (%s)", symbol, day, exc)
        return []

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode()

    # The percentage column is written as "-5.00", so it parses as a float and
    # not as an int - reading it strictly as an int silently drops every row.
    grouped: Dict[str, Dict[float, float]] = defaultdict(dict)
    for row in csv.DictReader(io.StringIO(text)):
        try:
            pct = float(row["percentage"])
            grouped[row["timestamp"]][pct] = float(row["notional"])
        except (KeyError, ValueError):
            continue

    out: List[DepthSnapshot] = []
    rejected = 0
    for ts, levels in grouped.items():
        asks = {p: levels[p] for p in BUCKETS if p in levels}
        bids = {p: levels[-p] for p in BUCKETS if -p in levels}
        if not (asks and bids):
            continue
        snapshot = DepthSnapshot(timestamp=ts, asks=asks, bids=bids)
        if is_plausible(snapshot):
            out.append(snapshot)
        else:
            rejected += 1
    if rejected:
        logger.warning("%s %s: dropped %d implausible snapshots of %d",
                       symbol, day, rejected, rejected + len(out))
    out.sort(key=lambda s: s.timestamp)
    return out


def slippage_bps(levels: Dict[float, float], notional: float) -> Optional[float]:
    """Average distance from mid, in bp, needed to fill `notional` on one side.

    The archive gives cumulative notional at each 1% band, so depth is treated as
    uniform *within* a band and the fill is integrated across the bands it spans.
    Filling entirely inside the first band costs half of it on average, not a
    full percent - assuming otherwise would overstate every small order by 2x.

    Returns None when the published depth (5% out) cannot cover the size; the
    answer is then "more than 250bp", not a number.
    """
    if notional <= 0:
        return 0.0

    previous_pct = 0.0
    previous_cum = 0.0
    cost = 0.0  # notional-weighted distance from mid, in percent

    for pct in sorted(levels):
        cum = levels[pct]
        if cum <= previous_cum:
            continue
        band_notional = cum - previous_cum
        remaining = notional - previous_cum

        if remaining <= band_notional:
            # Partial fill of this band: the average price sits halfway into the
            # portion consumed, not at the band's far edge.
            fraction = remaining / band_notional
            reached = previous_pct + fraction * (pct - previous_pct)
            cost += remaining * (previous_pct + reached) / 2
            return cost / notional * 100.0  # percent -> bp

        cost += band_notional * (previous_pct + pct) / 2
        previous_pct, previous_cum = float(pct), cum

    return None  # book does not cover the size within the published bands


def round_trip_bps(snapshot: DepthSnapshot, notional: float) -> Optional[float]:
    """Cost of entering and leaving one perp leg: cross the ask, later the bid."""
    buy = slippage_bps(snapshot.asks, notional)
    sell = slippage_bps(snapshot.bids, notional)
    if buy is None or sell is None:
        return None
    return buy + sell


@dataclass
class DepthStats:
    symbol: str
    notional: float
    n_snapshots: int
    #: Share of snapshots where 5% of book could not absorb the size.
    uncovered_share: float
    median_bps: float
    p90_bps: float
    p99_bps: float
    worst_bps: float

    @property
    def stress_multiple(self) -> float:
        """How much worse a bad hour is than a typical one.

        This is the number worth carrying across venues: the level is
        Binance-specific, the shape much less so.
        """
        # Below a tenth of a basis point the median is rounding noise, and a
        # ratio against it says nothing - a cost that small is simply free.
        if self.median_bps < 0.1:
            return float("nan")
        return self.p99_bps / self.median_bps


def summarize(symbol: str, snapshots: Sequence[DepthSnapshot],
              notional: float) -> Optional[DepthStats]:
    values: List[float] = []
    uncovered = 0
    for snap in snapshots:
        rt = round_trip_bps(snap, notional)
        if rt is None:
            uncovered += 1
        else:
            values.append(rt)
    if not values:
        return None

    values.sort()
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return DepthStats(
        symbol=symbol,
        notional=notional,
        n_snapshots=len(snapshots),
        uncovered_share=uncovered / len(snapshots),
        median_bps=pick(0.50),
        p90_bps=pick(0.90),
        p99_bps=pick(0.99),
        worst_bps=values[-1],
    )


def days_back(n: int, end: Optional[date] = None) -> Iterator[date]:
    """Binance publishes with a day or two of lag, so start two days back."""
    end = end or (date.today() - timedelta(days=2))
    for i in range(n):
        yield end - timedelta(days=i)


def scan_history(symbols: Sequence[str], notional: float, n_days: int = 30,
                 end: Optional[date] = None) -> List[DepthStats]:
    """Depth statistics per symbol over the last `n_days` of published archives."""
    out: List[DepthStats] = []
    for symbol in symbols:
        snapshots: List[DepthSnapshot] = []
        for day in days_back(n_days, end):
            snapshots.extend(fetch_day(symbol, day))
        stats = summarize(symbol, snapshots, notional)
        if stats:
            out.append(stats)
        else:
            logger.warning("%s: no usable depth history", symbol)
    return out


def format_history(rows: Sequence[DepthStats], n_days: int) -> str:
    if not rows:
        return "no depth history retrieved"
    rows = sorted(rows, key=lambda r: r.median_bps)
    lines = [
        "=" * 78,
        f" HISTORICAL DEPTH   ${rows[0].notional:,.0f} per leg, "
        f"Binance USD-M, last {n_days} days",
        "=" * 78,
        "",
        f"{'symbol':<11}{'snapshots':>11}{'median':>9}{'p90':>9}{'p99':>9}"
        f"{'worst':>10}{'p99/med':>9}{'uncovered':>11}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"{r.symbol:<11}{r.n_snapshots:>11,}{r.median_bps:>8.1f}b"
            f"{r.p90_bps:>8.1f}b{r.p99_bps:>8.1f}b{r.worst_bps:>9.1f}b"
            + (f"{r.stress_multiple:>8.1f}x" if r.stress_multiple == r.stress_multiple
               else f"{'free':>9}")
            + f"{r.uncovered_share * 100:>10.1f}%"
        )
    lines += [
        "-" * 78,
        " p99/med is the stress multiple: how much worse a bad moment is than a",
        " typical one. Size the strategy against p99, not the median - a carry gets",
        " unwound precisely when depth has gone.",
        " Binance is a proxy for HL/OKX levels; carry the SHAPE across, and take the",
        " LEVEL from `--liquidity` on the venues actually being traded.",
        "=" * 78,
    ]
    return "\n".join(lines)
