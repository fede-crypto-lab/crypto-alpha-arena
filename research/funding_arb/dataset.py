"""Align two venues onto a common, strictly causal hourly grid.

Two independent traps live in here, and both of them manufacture fake profit:

1. **Interval mismatch.** A 1h Hyperliquid rate and an 8h OKX rate are not
   comparable numbers. Everything used for *signals* is annualised first.

2. **Lookahead.** At grid time t the strategy may only see funding that has
   already settled (`event_time <= t`). Forward-filling the last known rate is
   safe; interpolating between settlements is not, because it leaks the next
   print backwards into the decision.

Funding *cash flows* are deliberately NOT resampled - they are replayed as the
discrete events the exchange actually charged, so a position that straddles three
OKX settlements is credited exactly three times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .venues import HOURS_PER_YEAR, FundingPoint, Mark, Venue

HOUR_MS = 3_600_000


def annualize(rate: float, interval_hours: float) -> float:
    """Convert a per-interval funding rate into a simple annualised rate.

    Simple (not compounded) scaling is used throughout: it is the convention every
    venue quotes in, and compounding a rate that flips sign several times a day
    would be more precise about the wrong thing.
    """
    if interval_hours <= 0:
        raise ValueError("interval_hours must be positive")
    return rate * (HOURS_PER_YEAR / interval_hours)


@dataclass
class VenueSeries:
    """One venue's funding and marks, plus the interval actually observed."""

    venue: Venue
    funding: List[FundingPoint]
    marks: List[Mark]
    interval_hours: float

    @property
    def name(self) -> str:
        return self.venue.name

    def annualized_at(self, time_ms: int) -> float:
        """Most recent *settled* funding as of `time_ms`, annualised."""
        rate = _last_at(self.funding, time_ms)
        return annualize(rate, self.interval_hours) if rate is not None else 0.0

    def mark_at(self, time_ms: int) -> float:
        price = _last_mark_at(self.marks, time_ms)
        if price is None:
            raise KeyError(f"{self.name}: no mark at or before {time_ms}")
        return price

    def funding_between(self, start_ms: int, end_ms: int) -> List[FundingPoint]:
        """Settlements in (start_ms, end_ms] - exclusive of entry, inclusive of exit.

        Exclusive at the open so that entering exactly on a settlement timestamp
        does not collect a payment the position was not yet open for.
        """
        return [p for p in self.funding if start_ms < p.time_ms <= end_ms]


@dataclass
class Pair:
    """Two venues' series for the same coin on a shared hourly grid."""

    coin: str
    a: VenueSeries  # reference leg
    b: VenueSeries
    grid: List[int]

    def spread_apr_at(self, time_ms: int) -> float:
        """Annualised funding of `b` minus that of `a`, as known at `time_ms`.

        Positive means b pays more than a: short b, long a, collect the difference.
        """
        return self.b.annualized_at(time_ms) - self.a.annualized_at(time_ms)

    def spread_series(self) -> List[Tuple[int, float]]:
        return [(t, self.spread_apr_at(t)) for t in self.grid]


def load_pair(
    venue_a: Venue,
    venue_b: Venue,
    coin: str,
    start_ms: int,
    end_ms: int,
) -> Pair:
    """Fetch both venues and build the aligned pair."""
    series = []
    for venue in (venue_a, venue_b):
        funding = venue.fetch_funding(coin, start_ms, end_ms)
        if not funding:
            raise ValueError(f"{venue.name}: no funding data for {coin} in window")
        marks = venue.fetch_marks(coin, start_ms, end_ms)
        if not marks:
            raise ValueError(f"{venue.name}: no marks for {coin} in window")
        series.append(
            VenueSeries(
                venue=venue,
                funding=funding,
                marks=marks,
                interval_hours=venue.detect_interval_hours(funding),
            )
        )

    a, b = series
    # Start the grid only once BOTH venues have a settled print and a mark to show;
    # before that the "spread" would be measured against a zero-filled leg.
    grid_start = max(
        a.funding[0].time_ms, b.funding[0].time_ms,
        a.marks[0].time_ms, b.marks[0].time_ms,
    )
    grid_end = min(
        a.funding[-1].time_ms, b.funding[-1].time_ms,
        a.marks[-1].time_ms, b.marks[-1].time_ms,
    )
    if grid_end <= grid_start:
        raise ValueError(f"{coin}: venue histories do not overlap")

    first = ((grid_start + HOUR_MS - 1) // HOUR_MS) * HOUR_MS
    grid = list(range(first, grid_end + 1, HOUR_MS))
    return Pair(coin=coin, a=a, b=b, grid=grid)


def _last_at(points: Sequence[FundingPoint], time_ms: int):
    """Rate of the latest settlement with time <= `time_ms`, or None."""
    lo, hi, found = 0, len(points) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if points[mid].time_ms <= time_ms:
            found = points[mid].rate
            lo = mid + 1
        else:
            hi = mid - 1
    return found


def _last_mark_at(marks: Sequence[Mark], time_ms: int):
    lo, hi, found = 0, len(marks) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if marks[mid].time_ms <= time_ms:
            found = marks[mid].close
            lo = mid + 1
        else:
            hi = mid - 1
    return found
