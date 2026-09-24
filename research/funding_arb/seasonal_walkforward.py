"""Does a seasonal spread that "always won" keep winning? A walk-forward test.

Seasonal scanners (SeasonAlgo and similar) show spreads that won, say, 13 of the
last 15 years between two calendar dates. What they cannot show is what happened
the year *after* a pattern qualified - and that is the only number that matters to
someone who trades it. This module reproduces the scanner's selection honestly:

1. For each test year Y, look only at the `lookback` years before it.
2. Scan every (entry day, holding period, direction) window, exactly as a scanner
   does, and keep the ones whose historical win rate clears the threshold.
3. Trade those windows in year Y and record the result.

The comparison that decides it is the out-of-sample win rate of the selected
windows against the out-of-sample win rate of *all* windows. If selection by past
reliability carried information, the first would sit clearly above the second.

The false-positive arithmetic is included because it is the reason this test
exists: with ~1,000 windows per spread, a fair coin clears 12 of 15 years about
1.8% of the time, so a scanner surfaces ~18 "reliable" patterns per spread per year
from noise alone - before any real seasonality is involved.
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

DailySeries = Dict[date, float]
#: (entry day-of-year, holding days, direction: +1 long the spread, -1 short)
Candidate = Tuple[int, int, int]

DEFAULT_HOLDS = (10, 15, 20, 30, 45, 60, 90)


class _Index:
    """Sorted view of a series for 'first observation on or after' lookups."""

    def __init__(self, series: DailySeries):
        self.dates = sorted(series)
        self.values = [series[d] for d in self.dates]

    def on_or_after(self, target: date, max_gap_days: int = 7) -> Optional[Tuple[date, float]]:
        i = bisect.bisect_left(self.dates, target)
        if i >= len(self.dates):
            return None
        found = self.dates[i]
        # A long gap means missing data, not a holiday; refuse rather than
        # silently shifting the trade by weeks.
        if (found - target).days > max_gap_days:
            return None
        return found, self.values[i]


def candidate_grid(entry_step: int = 5, holds: Sequence[int] = DEFAULT_HOLDS,
                   directions: Sequence[int] = (1, -1)) -> List[Candidate]:
    """Every window a scanner would test."""
    return [(doy, h, d) for doy in range(1, 366, entry_step)
            for h in holds for d in directions]


def trade_pnl(index: _Index, year: int, candidate: Candidate) -> Optional[float]:
    """Spread change from entry to exit, signed by direction. None if no data."""
    doy, hold, direction = candidate
    target = date(year, 1, 1) + timedelta(days=doy - 1)
    entry = index.on_or_after(target)
    if entry is None:
        return None
    exit_ = index.on_or_after(entry[0] + timedelta(days=hold))
    if exit_ is None:
        return None
    return direction * (exit_[1] - entry[1])


def pnl_table(series: DailySeries, candidates: Sequence[Candidate],
              years: Sequence[int]) -> Dict[Candidate, Dict[int, float]]:
    index = _Index(series)
    table: Dict[Candidate, Dict[int, float]] = {}
    for cand in candidates:
        row = {}
        for y in years:
            pnl = trade_pnl(index, y, cand)
            if pnl is not None:
                row[y] = pnl
        table[cand] = row
    return table


def chance_of_qualifying(lookback: int, min_wins: int, p: float = 0.5) -> float:
    """Probability a window with NO edge wins at least `min_wins` of `lookback`."""
    return sum(comb(lookback, k) * p ** k * (1 - p) ** (lookback - k)
               for k in range(min_wins, lookback + 1))


@dataclass
class WalkForwardResult:
    name: str
    lookback: int
    min_wins: int
    test_years: int = 0
    candidates_per_year: int = 0
    #: every (candidate, test-year) where the candidate qualified
    selected_pnl: List[float] = field(default_factory=list)
    selected_in_sample_win: List[float] = field(default_factory=list)
    #: every (candidate, test-year), qualified or not - the no-skill baseline
    all_pnl: List[float] = field(default_factory=list)
    #: the single best-looking window per year, as a trader would pick it
    best_pick_pnl: List[float] = field(default_factory=list)
    #: (test year, candidate, out-of-sample pnl) for each best pick, so the
    #: chosen windows can be read back and checked against a physical story
    best_picks: List[Tuple[int, Candidate, float]] = field(default_factory=list)

    @staticmethod
    def _win(xs: Sequence[float]) -> float:
        return sum(1 for x in xs if x > 0) / len(xs) if xs else float("nan")

    @property
    def selected_oos_win(self) -> float:
        return self._win(self.selected_pnl)

    @property
    def baseline_oos_win(self) -> float:
        return self._win(self.all_pnl)

    @property
    def best_pick_oos_win(self) -> float:
        return self._win(self.best_pick_pnl)

    @property
    def expected_false_per_year(self) -> float:
        return self.candidates_per_year * chance_of_qualifying(self.lookback, self.min_wins)

    @property
    def qualified_per_year(self) -> float:
        return len(self.selected_pnl) / self.test_years if self.test_years else 0.0


def walk_forward(name: str, series: DailySeries, lookback: int = 15,
                 min_wins: int = 12, cost: float = 0.0,
                 candidates: Optional[Sequence[Candidate]] = None,
                 min_history: Optional[int] = None) -> WalkForwardResult:
    """Select windows on the trailing `lookback` years, trade them the next year.

    `cost` is subtracted from every trade in the spread's own units, so it can be
    set to a realistic round trip for the instrument. `min_history` is how many of
    the lookback years must have data for a window to be judged; it defaults to
    the full lookback, because judging "12 of 15" on 9 observed years is a
    different and weaker claim.
    """
    candidates = list(candidates or candidate_grid())
    min_history = min_history or lookback
    years = sorted({d.year for d in series})
    table = pnl_table(series, candidates, years)
    result = WalkForwardResult(name=name, lookback=lookback, min_wins=min_wins,
                               candidates_per_year=len(candidates))

    for test_year in years[lookback:]:
        train = range(test_year - lookback, test_year)
        best: Optional[Tuple[float, float, float, Candidate]] = None  # (win, mean, oos, cand)
        any_traded = False
        for cand in candidates:
            row = table[cand]
            oos = row.get(test_year)
            if oos is None:
                continue
            oos -= cost
            any_traded = True
            result.all_pnl.append(oos)
            hist = [row[y] - cost for y in train if y in row]
            if len(hist) < min_history:
                continue
            wins = sum(1 for x in hist if x > 0)
            if wins >= min_wins:
                result.selected_pnl.append(oos)
                result.selected_in_sample_win.append(wins / len(hist))
                key = (wins / len(hist), statistics.fmean(hist), oos, cand)
                if best is None or key[:2] > best[:2]:
                    best = key
        if any_traded:
            result.test_years += 1
        if best is not None:
            result.best_pick_pnl.append(best[2])
            result.best_picks.append((test_year, best[3], best[2]))
    return result


def format_results(rows: Sequence[WalkForwardResult], unit: str = "$/bbl") -> str:
    if not rows:
        return "no results"
    r0 = rows[0]
    lines = [
        "=" * 96,
        f" SEASONAL WALK-FORWARD   select on {r0.lookback}y, require >= "
        f"{r0.min_wins}/{r0.lookback} wins, trade the following year",
        "=" * 96,
        "",
        f"{'spread':<22}{'years':>6}{'qualify/yr':>11}{'by chance':>10}"
        f"{'IS win':>8}{'OOS win':>9}{'baseline':>9}{'best pick':>10}"
        f"{'avg win':>9}{'avg loss':>9}",
        "-" * 96,
    ]
    for r in rows:
        wins = [x for x in r.selected_pnl if x > 0]
        losses = [x for x in r.selected_pnl if x <= 0]
        is_win = statistics.fmean(r.selected_in_sample_win) if r.selected_in_sample_win else float("nan")
        lines.append(
            f"{r.name:<22}{r.test_years:>6}{r.qualified_per_year:>11.1f}"
            f"{r.expected_false_per_year:>10.1f}{is_win:>8.0%}{r.selected_oos_win:>9.0%}"
            f"{r.baseline_oos_win:>9.0%}{r.best_pick_oos_win:>10.0%}"
            f"{(statistics.fmean(wins) if wins else 0):>9.2f}"
            f"{(statistics.fmean(losses) if losses else 0):>9.2f}"
        )
    lines += [
        "-" * 96,
        " qualify/yr: windows that passed the filter each year (what a scanner shows).",
        " by chance:  how many would pass per year with NO seasonal edge at all.",
        " IS win:     their historical win rate - the number the scanner advertises.",
        " OOS win:    what they did the following year. baseline: all windows, no",
        f" selection. best pick: the single best-looking window each year. P&L in {unit}.",
        "=" * 96,
    ]
    return "\n".join(lines)
