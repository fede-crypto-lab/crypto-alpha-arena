"""Does the funding ranking persist? The test the whole strategy rests on.

The single-pair backtest established that a coin's own funding cannot be timed:
the EWMA forecast's entry selectivity came out below 1.0, so acting on it was
worse than doing nothing. A cross-sectional strategy makes a different claim -
that the coins paying the most *now* will still be paying the most *next week* -
and that claim has to be falsified before any book is built on it.

Two statistics answer it:

* **Spearman rank correlation** between trailing funding and the funding actually
  realised over the following window. Near zero means the ranking is noise and the
  strategy is a fee generator.
* **Realised funding by quintile.** The rank correlation can be positive while the
  top quintile still earns nothing extra; what pays the trade is the *spread*
  between what the top of the ranking goes on to earn and what the universe earns
  on average.

Both are computed on non-overlapping windows, so the periods are independent and
the dispersion across them is meaningful rather than an artefact of a rolling
window reusing the same data.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .dataset import HOUR_MS, annualize
from .venues import FundingPoint, Venue

logger = logging.getLogger(__name__)

DAY_MS = 24 * HOUR_MS


@dataclass
class PersistenceResult:
    window_days: int
    n_periods: int
    mean_rho: float
    min_rho: float
    max_rho: float
    top_quintile_apr: float
    median_apr: float
    bottom_quintile_apr: float

    @property
    def spread_apr(self) -> float:
        """What the ranking is worth: top quintile minus bottom quintile."""
        return self.top_quintile_apr - self.bottom_quintile_apr

    @property
    def edge_over_median(self) -> float:
        """What selection is worth over just holding the average coin."""
        return self.top_quintile_apr - self.median_apr


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation, written out so the module stays dependency-free.

    Ranks rather than levels, because funding distributions have a hard floor at
    the venue's minimum and a long right tail - a Pearson correlation on the raw
    values would be dominated by a handful of hype episodes.

    Ties take the average rank, and this is not a detail here: venues clamp
    funding at a baseline, so a large block of coins sits at *exactly* the same
    rate. Breaking those ties by sort order would hand them the same arbitrary
    ordering in both windows and manufacture agreement that is not in the data -
    inflating rho precisely where the ranking actually carries no information.
    """
    if len(xs) != len(ys) or len(xs) < 3:
        return 0.0

    def ranks(values: Sequence[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0  # average rank across the tied block
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def mean_apr(points: Sequence[FundingPoint], start_ms: int, end_ms: int,
             interval_hours: float) -> Optional[float]:
    vals = [annualize(p.rate, interval_hours) for p in points if start_ms <= p.time_ms < end_ms]
    return statistics.fmean(vals) if vals else None


def fetch_funding_panel(
    venue: Venue,
    coins: Sequence[str],
    start_ms: int,
    end_ms: int,
    min_points: int = 24 * 60,
) -> Dict[str, List[FundingPoint]]:
    """Funding history for many coins, dropping those with too little history.

    A coin listed halfway through the sample would otherwise be ranked against
    coins with twice the history, and short series produce wild trailing means.
    """
    panel: Dict[str, List[FundingPoint]] = {}
    for coin in coins:
        try:
            points = venue.fetch_funding(coin, start_ms, end_ms)
        except Exception as exc:  # noqa: BLE001 - a delisted ticker must not stop the scan
            logger.warning("skipping %s: %s", coin, exc)
            continue
        if len(points) >= min_points:
            panel[coin] = points
        else:
            logger.info("skipping %s: only %d settlements", coin, len(points))
    return panel


def measure(
    panel: Dict[str, List[FundingPoint]],
    start_ms: int,
    end_ms: int,
    window_days: int,
    interval_hours: float = 1.0,
    min_coins: int = 20,
) -> Optional[PersistenceResult]:
    """Rank on the trailing window, score on the next one, over disjoint periods."""
    window = window_days * DAY_MS
    rhos: List[float] = []
    tops: List[float] = []
    medians: List[float] = []
    bottoms: List[float] = []

    t = start_ms + window
    while t + window <= end_ms:
        past: List[float] = []
        future: List[float] = []
        for points in panel.values():
            p = mean_apr(points, t - window, t, interval_hours)
            f = mean_apr(points, t, t + window, interval_hours)
            if p is not None and f is not None:
                past.append(p)
                future.append(f)

        if len(past) >= min_coins:
            rhos.append(spearman(past, future))
            order = sorted(range(len(past)), key=lambda i: -past[i])
            size = max(1, len(order) // 5)
            tops.append(statistics.fmean([future[i] for i in order[:size]]))
            bottoms.append(statistics.fmean([future[i] for i in order[-size:]]))
            medians.append(statistics.median(future))

        t += window  # non-overlapping

    if not rhos:
        return None

    return PersistenceResult(
        window_days=window_days,
        n_periods=len(rhos),
        mean_rho=statistics.fmean(rhos),
        min_rho=min(rhos),
        max_rho=max(rhos),
        top_quintile_apr=statistics.fmean(tops),
        median_apr=statistics.fmean(medians),
        bottom_quintile_apr=statistics.fmean(bottoms),
    )


def format_persistence(results: Sequence[PersistenceResult], n_coins: int) -> str:
    pct = lambda x: f"{x * 100:+.1f}%"
    lines = [
        "=" * 68,
        f" FUNDING RANK PERSISTENCE   {n_coins} coins",
        "=" * 68,
        "",
        f"{'window':>8}{'periods':>9}{'rho':>8}{'rho min':>9}"
        f"{'top Q':>9}{'median':>9}{'bottom Q':>10}{'Q1-Q5':>9}",
        "-" * 71,
    ]
    for r in results:
        lines.append(
            f"{r.window_days:>6}d{r.n_periods:>9}{r.mean_rho:>8.3f}{r.min_rho:>9.2f}"
            f"{pct(r.top_quintile_apr):>9}{pct(r.median_apr):>9}"
            f"{pct(r.bottom_quintile_apr):>10}{pct(r.spread_apr):>9}"
        )
    lines += [
        "",
        " rho is the Spearman correlation between trailing and subsequent funding.",
        " Near zero would mean the ranking is noise and any book built on it is a",
        " fee generator. 'top Q' is what the richest fifth went on to actually pay.",
        "=" * 68,
    ]
    return "\n".join(lines)
