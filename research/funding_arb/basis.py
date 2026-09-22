"""How well does the hedge actually hold? The question the earlier modules missed.

A cash-and-carry is called delta-neutral because the long spot and the short perp
are the same asset. They are not the same *instrument*, and the gap between them -
the basis - moves. What lands in the P&L is not the basis itself but its **change
over the holding period**: enter at one basis, leave at another, and the
difference is yours whether you wanted it or not.

On BTC that difference is noise: MEXC perp sits within 0.08% of MEXC spot from the
1st to the 99th percentile. On a thin alt it is not: HYPE reaches 6.5% and PONS
5.4%. A carry earning 20% annualised over a forty-day hold collects about 2.2% -
so a basis that moves 3% against it during that hold erases the trade and more.

This is a different failure from thin execution, and the earlier measures could
not see it. `liquidity.py` and `depth_history.py` both ask what it costs to *cross
the book*; a coin can have a perfectly serviceable book and still carry a hedge
that drifts. Filtering the universe on execution cost alone therefore admits coins
whose carry is structurally uninsurable.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .dataset import HOUR_MS
from .venues import Mark, Venue

logger = logging.getLogger(__name__)


@dataclass
class BasisStats:
    coin: str
    n_observations: int
    hold_hours: int
    #: Level of perp/spot - 1, for reference.
    median_level: float
    #: What actually hits P&L: the change in basis across a holding period.
    median_move: float
    p90_move: float
    p99_move: float
    worst_move: float

    @property
    def hedge_is_reliable(self) -> bool:
        """A rough gate: a 1% adverse basis move eats a month of typical carry."""
        return self.p99_move < 0.01

    def carry_months_lost(self, carry_apr: float = 0.20) -> float:
        """A p99 basis move, expressed in months of carry on the notional."""
        if carry_apr <= 0:
            return float("inf")
        return self.p99_move / carry_apr * 12


def basis_series(spot: Sequence[Mark], perp: Sequence[Mark]) -> List[Tuple[int, float]]:
    """perp/spot - 1 at every timestamp both venues report.

    Intersecting timestamps rather than forward-filling matters here: a stale
    mark on one leg invents a basis move that never happened, and on a thin alt
    those artefacts would swamp the real signal.
    """
    spot_by_time: Dict[int, float] = {m.time_ms: m.close for m in spot}
    out: List[Tuple[int, float]] = []
    for mark in perp:
        reference = spot_by_time.get(mark.time_ms)
        if reference and reference > 0:
            out.append((mark.time_ms, mark.close / reference - 1.0))
    out.sort()
    return out


def holding_move(series: Sequence[Tuple[int, float]], hold_hours: int) -> List[float]:
    """|basis(t + hold) - basis(t)| over every start point in the sample."""
    by_time = dict(series)
    step = hold_hours * HOUR_MS
    moves: List[float] = []
    for t, start_basis in series:
        end_basis = by_time.get(t + step)
        if end_basis is not None:
            moves.append(abs(end_basis - start_basis))
    return moves


def measure(coin: str, spot: Sequence[Mark], perp: Sequence[Mark],
            hold_hours: int = 480, min_observations: int = 200) -> Optional[BasisStats]:
    series = basis_series(spot, perp)
    if len(series) < min_observations:
        logger.info("%s: only %d aligned marks, skipping", coin, len(series))
        return None

    moves = holding_move(series, hold_hours)
    if len(moves) < min_observations:
        logger.info("%s: only %d holding windows, skipping", coin, len(moves))
        return None

    moves.sort()
    pick = lambda q: moves[min(len(moves) - 1, int(q * len(moves)))]
    return BasisStats(
        coin=coin,
        n_observations=len(moves),
        hold_hours=hold_hours,
        median_level=statistics.median(b for _, b in series),
        median_move=pick(0.50),
        p90_move=pick(0.90),
        p99_move=pick(0.99),
        worst_move=moves[-1],
    )


def scan(coins: Sequence[str], spot_venue: Venue, perp_venue: Venue,
         start_ms: int, end_ms: int, hold_hours: int = 480) -> List[BasisStats]:
    out: List[BasisStats] = []
    for coin in coins:
        try:
            spot = spot_venue.fetch_marks(coin, start_ms, end_ms)
            perp = perp_venue.fetch_marks(coin, start_ms, end_ms)
        except Exception as exc:  # noqa: BLE001 - one bad listing must not stop the scan
            logger.warning("basis scan failed for %s: %s", coin, exc)
            continue
        stats = measure(coin, spot, perp, hold_hours)
        if stats:
            out.append(stats)
    return out


def format_scan(rows: Sequence[BasisStats], carry_apr: float = 0.20) -> str:
    if not rows:
        return "no basis measured"
    rows = sorted(rows, key=lambda r: r.p99_move)
    hold = rows[0].hold_hours
    lines = [
        "=" * 78,
        f" HEDGE QUALITY   basis move over a {hold}h ({hold / 24:.0f}d) hold,"
        f" perp vs spot",
        "=" * 78,
        "",
        f"{'coin':<9}{'windows':>9}{'level':>9}{'median':>9}{'p90':>9}{'p99':>9}"
        f"{'worst':>9}{'carry months lost':>19}",
        "-" * 78,
    ]
    for r in rows:
        flag = "" if r.hedge_is_reliable else "  <-"
        lines.append(
            f"{r.coin:<9}{r.n_observations:>9,}{r.median_level * 100:>8.2f}%"
            f"{r.median_move * 100:>8.2f}%{r.p90_move * 100:>8.2f}%"
            f"{r.p99_move * 100:>8.2f}%{r.worst_move * 100:>8.2f}%"
            f"{r.carry_months_lost(carry_apr):>15.1f}{flag}"
        )
    reliable = [r for r in rows if r.hedge_is_reliable]
    lines += [
        "-" * 78,
        f" {len(reliable)} of {len(rows)} coins hold a hedge worth the name"
        f" (p99 basis move under 1%).",
        f" 'carry months lost' is a p99 move priced against {carry_apr:.0%} annualised"
        f" carry: how",
        " long the position must run just to earn back one bad exit on the hedge.",
        " This is NOT execution cost. A coin can have a fine book and still drift.",
        "=" * 78,
    ]
    return "\n".join(lines)
