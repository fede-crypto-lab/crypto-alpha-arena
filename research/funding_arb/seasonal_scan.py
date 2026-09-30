#!/usr/bin/env python3
"""Every commodity spread, every seasonal window: does picking by the past work?

A scanner over thousands of spreads always finds hundreds that "won 13 of 15
years": chance alone guarantees it (MEMORY.md §11-quater and §11-quinquies, where
four corn spreads at 87-93% went to 27-40% on the years before the scanner's
sample). So the output here is not a list of winners. It answers one question
first, pooled over all spreads:

    when a window is chosen using ONLY the years before Y, does it win in Y
    more often than a window chosen at random?

That is a walk-forward: for every test cycle, qualify windows on the previous
`lookback` cycles, take the best one, trade it in the test cycle, record the
result. If the pooled out-of-sample win rate of those picks is no better than
the baseline of all windows, seasonal selection carries no information in this
data and no individual spread from the scan should be traded - however good its
history looks.

Spreads, each on the contracts a trader would hold (no rolled series):
* calendar: one root, front month m1 against a later month of the same root;
* inter-commodity: two roots, same delivery month and year, one contract each,
  P&L in dollars (price x $/point per leg).

A "cycle" is one instance of a spread (e.g. CL June 2019 vs CL Dec 2019). Windows
are placed relative to the front leg's last trading day, so the same window means
the same point in every contract's life even when expiry dates drift.

    python -m research.funding_arb.seasonal_scan research/funding_arb/data/databento
"""

from __future__ import annotations

import argparse
import bisect
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .futures_contracts import (
    MONTH_CODES, SPECS, ContractKey, DailySeries, read_dir, round_trip_cost,
)
from .metrics import wilson_interval
from .seasonal_walkforward import chance_of_qualifying

#: Window grid. Entry is `offset` days before the front leg's last print; exit
#: `hold` days later. Coarser than a commercial scanner, which only makes the
#: multiple-comparisons problem smaller, not the test weaker.
DEFAULT_OFFSETS = tuple(range(35, 330, 7))
DEFAULT_HOLDS = (20, 30, 45, 60, 90)
#: A price older than this is not a price you could have traded at.
MAX_GAP_DAYS = 5
#: Exit no later than this many days before the first leg's last print: the
#: final days of a contract are thin and dominated by delivery logistics.
EXPIRY_BUFFER_DAYS = 5


@dataclass(frozen=True)
class SpreadDef:
    name: str
    legs: Tuple[Tuple[str, str, int], ...]   # (root, month code, year offset vs cycle year)

    @property
    def roots(self) -> Tuple[str, ...]:
        return tuple(r for r, _, _ in self.legs)


Window = Tuple[int, int, int]   # (offset days before front expiry, hold days, direction)


@dataclass
class Cycle:
    year: int
    anchor: date                 # front leg's last print
    values: Dict[date, float]    # spread in dollars per 1:1 contract set


def calendar_defs(roots: Sequence[str], max_gap_months: int = 12) -> List[SpreadDef]:
    out = []
    for root in roots:
        for i, m1 in enumerate(MONTH_CODES):
            for gap in range(1, max_gap_months + 1):
                j = i + gap
                m2, yoff = MONTH_CODES[j % 12], j // 12
                out.append(SpreadDef(f"{root}{m1}-{root}{m2}{'+1' if yoff else ''}",
                                     ((root, m1, 0), (root, m2, yoff))))
    return out


def inter_defs(roots: Sequence[str]) -> List[SpreadDef]:
    out = []
    for a in range(len(roots)):
        for b in range(a + 1, len(roots)):
            for m in MONTH_CODES:
                out.append(SpreadDef(f"{roots[a]}{m}-{roots[b]}{m}",
                                     ((roots[a], m, 0), (roots[b], m, 0))))
    return out


#: A leg whose last print is this close to the end of the data is still trading:
#: its last print is today, not its expiry, and windows anchored on it would
#: sit in the wrong place in the contract's life.
LIVE_MARGIN_DAYS = 10


def ends_at_expiry(key: ContractKey, s: DailySeries) -> bool:
    """Does the series stop where this contract's life ends?

    Every product here stops trading between ~2 months before its delivery
    month (Brent: last business day of the second preceding month) and the end
    of the delivery month (gold, cattle). A series that stops elsewhere is a
    far-dated contract's stray prints filed under a reused symbol (they showed
    up as "expired" 2028 and 2030 cycles in the first full scan) or a truncated
    history; either would anchor windows at the wrong point in the contract's life.
    """
    _, month, year = key
    first = date(year, MONTH_CODES.index(month) + 1, 1)
    end_of_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return first - timedelta(days=75) <= max(s) <= end_of_month + timedelta(days=3)


def build_cycles(sd: SpreadDef, data: Dict[ContractKey, DailySeries],
                 as_of: Optional[date] = None) -> List[Cycle]:
    """One cycle per front-leg delivery year where every leg has data and has
    expired by `as_of`. Values exist only on days all legs printed: a
    forward-filled leg is a fake spread."""
    years = sorted({y for (r, m, y) in data if (r, m) == sd.legs[0][:2]})
    cycles = []
    for y in years:
        series = []
        for root, month, yoff in sd.legs:
            s = data.get((root, month, y + yoff))
            if not s or (as_of and not ends_at_expiry((root, month, y + yoff), s)
                         and max(s) <= as_of - timedelta(days=LIVE_MARGIN_DAYS)):
                break
            series.append((root, s))
        else:
            front = series[0][1]
            if as_of and any(max(s) > as_of - timedelta(days=LIVE_MARGIN_DAYS) for _, s in series):
                continue
            # The window must close before the FIRST leg stops trading: for
            # inter-commodity pairs that is not always the front (CL expires
            # about a week before RB of the same month).
            anchor = min(max(s) for _, s in series)
            common = set(front)
            for _, s in series[1:]:
                common &= set(s)
            sign = [1, -1] + [0] * (len(series) - 2)
            values = {d: sum(sg * s[d] * SPECS[r].dollars_per_point
                             for sg, (r, s) in zip(sign, series)) for d in common}
            if values:
                cycles.append(Cycle(y, anchor, values))
    return cycles


class _Lookup:
    def __init__(self, values: Dict[date, float]):
        self.dates = sorted(values)
        self.values = values

    def at_or_after(self, target: date) -> Optional[Tuple[date, float]]:
        i = bisect.bisect_left(self.dates, target)
        if i == len(self.dates) or (self.dates[i] - target).days > MAX_GAP_DAYS:
            return None
        return self.dates[i], self.values[self.dates[i]]


def window_pnl(cycle: Cycle, lookup: _Lookup, w: Window) -> Optional[float]:
    offset, hold, direction = w
    entry = lookup.at_or_after(cycle.anchor - timedelta(days=offset))
    if entry is None:
        return None
    exit_ = lookup.at_or_after(entry[0] + timedelta(days=hold))
    if exit_ is None or exit_[0] > cycle.anchor - timedelta(days=EXPIRY_BUFFER_DAYS):
        return None
    return direction * (exit_[1] - entry[1])


def windows(offsets=DEFAULT_OFFSETS, holds=DEFAULT_HOLDS) -> List[Window]:
    return [(o, h, d) for o in offsets for h in holds if h < o for d in (1, -1)]


@dataclass
class Pick:
    spread: str
    test_year: int
    window: Window
    prior_wins: int
    prior_mean: float
    oos_net: float


@dataclass
class Tally:
    """Running win/mean without keeping every value: the baseline alone is
    tens of millions of window-years."""
    n: int = 0
    wins: int = 0
    total: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        self.wins += x > 0
        self.total += x


@dataclass
class ScanResult:
    spreads_tested: int = 0
    windows_per_spread: int = 0
    baseline_oos: Tally = field(default_factory=Tally)    # every window, every test year
    qualified_oos: Tally = field(default_factory=Tally)
    best_picks: List[Pick] = field(default_factory=list)


def scan(defs: Sequence[SpreadDef], data: Dict[ContractKey, DailySeries],
         lookback: int, min_wins: int, grid: Sequence[Window]) -> ScanResult:
    res = ScanResult(windows_per_spread=len(grid))
    as_of = max(max(s) for s in data.values())
    for sd in defs:
        cycles = build_cycles(sd, data, as_of)
        if len(cycles) <= lookback:
            continue
        res.spreads_tested += 1
        cost = round_trip_cost(sd.roots)
        table: Dict[Window, List[Optional[float]]] = {}
        for c in cycles:
            lk = _Lookup(c.values)
            for w in grid:
                pnl = window_pnl(c, lk, w)
                table.setdefault(w, []).append(None if pnl is None else pnl - cost)
        for t in range(lookback, len(cycles)):
            best: Optional[Pick] = None
            for w, row in table.items():
                oos = row[t]
                if oos is None:
                    continue
                res.baseline_oos.add(oos)
                prior = [x for x in row[t - lookback:t] if x is not None]
                if len(prior) < lookback:
                    continue
                wins = sum(x > 0 for x in prior)
                if wins < min_wins:
                    continue
                res.qualified_oos.add(oos)
                mean = statistics.mean(prior)
                if best is None or (wins, mean) > (best.prior_wins, best.prior_mean):
                    best = Pick(sd.name, cycles[t].year, w, wins, mean, oos)
            if best:
                res.best_picks.append(best)
    return res


def _line(label: str, t: Tally) -> str:
    if not t.n:
        return f"{label:28s} none"
    lo, hi = wilson_interval(t.wins, t.n)
    return (f"{label:28s} n={t.n:9d}  win {t.wins / t.n:5.1%} [{lo:.1%}-{hi:.1%}]  "
            f"mean net {t.total / t.n:8.0f} $")


def _tally(xs: Sequence[float]) -> Tally:
    t = Tally()
    for x in xs:
        t.add(x)
    return t


def format_report(res: ScanResult, lookback: int, min_wins: int) -> str:
    p = chance_of_qualifying(lookback, min_wins)
    by_chance = p * res.windows_per_spread * res.spreads_tested
    lines = [
        f"{res.spreads_tested} spreads x {res.windows_per_spread} windows, lookback {lookback}, "
        f"qualify at >= {min_wins}/{lookback} net wins (costs: estimate, commissions + 1 tick per leg)",
        f"a window with no edge qualifies {p:.2%} of the time -> ~{by_chance:.0f} qualifiers "
        f"per test year from chance alone",
        "",
        _line("baseline: every window", res.baseline_oos),
        _line("qualified windows", res.qualified_oos),
        _line("best pick per spread-year", _tally([b.oos_net for b in res.best_picks])),
    ]
    by_year = defaultdict(list)
    for b in res.best_picks:
        by_year[b.test_year].append(b.oos_net)
    lines += ["", "best picks by test year (years are not independent across spreads):"]
    for y in sorted(by_year):
        xs = by_year[y]
        lines.append(f"  {y}: {len(xs):5d} picks, win {sum(x > 0 for x in xs) / len(xs):5.1%}, "
                     f"mean net {statistics.mean(xs):8.0f} $")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory", help="folder written by fetch_databento")
    ap.add_argument("--lookback", type=int, default=8)
    ap.add_argument("--min-wins", type=int, default=7)
    ap.add_argument("--kind", choices=("calendar", "inter", "both"), default="both")
    ap.add_argument("--roots", nargs="+", help="restrict to these roots")
    a = ap.parse_args(argv)
    data = read_dir(a.directory)
    roots = sorted({k[0] for k in data} & set(a.roots or SPECS))
    defs = []
    if a.kind in ("calendar", "both"):
        defs += calendar_defs(roots)
    if a.kind in ("inter", "both"):
        defs += inter_defs(roots)
    res = scan(defs, data, a.lookback, a.min_wins, windows())
    print(format_report(res, a.lookback, a.min_wins))
    return 0


if __name__ == "__main__":
    sys.exit(main())
