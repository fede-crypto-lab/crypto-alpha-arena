#!/usr/bin/env python3
"""A fixed, non-seasonal calendar-spread rule tested on every product.

The rule that survived on copper (MEMORY.md §11-nonies) uses no seasonal choice:
for each liquid ("active") delivery month, trade it against the next active
month, always in the same direction, entering a fixed number of days before the
last day a retail account can hold the front and exiting a fixed number of days
later. Nothing is selected from the data except the direction, and both
directions are reported, so the multiple-comparison count is small and known
(products x 2). Criterion: MEMORY.md §11-decies, fixed before the first run.

    python -m research.funding_arb.fixed_rule_scan research/funding_arb/data/databento
    python -m research.funding_arb.fixed_rule_scan research/funding_arb/data/databento_settle
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .futures_contracts import MONTH_CODES, read_dir, round_trip_cost
from .seasonal_scan import SpreadDef, _Lookup, build_cycles, window_pnl

#: Delivery months with real liquidity. Off-cycle months trade thinly and their
#: quotes are not prices a retail spread order would get.
ACTIVE: Dict[str, str] = {
    "GC": "GJMQVZ", "SI": "HKNUZ", "HG": "HKNUZ", "PL": "FJNV", "PA": "HMUZ",
    "CL": "FGHJKMNQUVXZ", "BZ": "FGHJKMNQUVXZ", "HO": "FGHJKMNQUVXZ",
    "RB": "FGHJKMNQUVXZ", "NG": "FGHJKMNQUVXZ",
    "ZC": "HKNUZ", "ZW": "HKNUZ", "KE": "HKNUZ", "ZS": "FHKNQUX", "ZM": "FHKNQUVZ",
    "ZL": "FHKNQUVZ", "ZO": "HKNUZ", "ZR": "FHKNUX",
    "LE": "GJMQVZ", "GF": "FHJKQUVX", "HE": "GJKMNQVZ",
}
WINDOWS: Tuple[Tuple[int, int], ...] = ((90, 60), (120, 90), (180, 90), (240, 120))
KEY_WINDOW = (120, 90)
SPLIT_YEAR = 2018
FIRST_YEAR = 2011


def next_active_pairs(root: str) -> List[SpreadDef]:
    months = ACTIVE[root]
    out = []
    for i, m in enumerate(months):
        nxt, yoff = (months[i + 1], 0) if i + 1 < len(months) else (months[0], 1)
        out.append(SpreadDef(f"{root}{m}-{root}{nxt}{'+1' if yoff else ''}",
                             ((root, m, 0), (root, nxt, yoff))))
    return out


@dataclass
class Cell:
    """Results of one (product, direction, window)."""
    by_year: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list))

    @property
    def all(self) -> List[float]:
        return [x for v in self.by_year.values() for x in v]

    def ok(self) -> bool:
        xs = self.all
        return bool(xs) and sum(x > 0 for x in xs) / len(xs) >= 0.55 and statistics.mean(xs) > 0

    def half_mean(self, first: bool) -> Optional[float]:
        xs = [x for y, v in self.by_year.items() for x in v if (y <= SPLIT_YEAR) == first]
        return statistics.mean(xs) if xs else None


def run(data, roots: Sequence[str]) -> Dict[Tuple[str, int, Tuple[int, int]], Cell]:
    as_of = max(max(s) for s in data.values())
    cells: Dict[Tuple[str, int, Tuple[int, int]], Cell] = defaultdict(Cell)
    for root in roots:
        for sd in next_active_pairs(root):
            cost = round_trip_cost(sd.roots)
            for c in build_cycles(sd, data, as_of):
                lk = _Lookup(c.values)
                for off, hold in WINDOWS:
                    entry = lk.at_or_after(c.anchor - timedelta(days=off))
                    if entry is None or entry[0].year < FIRST_YEAR:
                        continue
                    for direction in (-1, 1):
                        p = window_pnl(c, lk, (off, hold, direction))
                        if p is not None:
                            cells[(root, direction, (off, hold))].by_year[entry[0].year].append(p - cost)
    return cells


def is_candidate(cells, root: str, direction: int) -> bool:
    passed = sum(cells[(root, direction, w)].ok() for w in WINDOWS if (root, direction, w) in cells)
    key = cells.get((root, direction, KEY_WINDOW))
    halves = key and all((m := key.half_mean(h)) is not None and m > 0 for h in (True, False))
    return passed >= 3 and bool(halves)


def report(cells, roots: Sequence[str]) -> str:
    lines = ["direction -1 = sell near / buy next, +1 = buy near / sell next; costs are an estimate",
             f"{'product':7s} {'dir':>3s}  " + "  ".join(f"{o}/{h}".center(22) for o, h in WINDOWS)
             + f"  {'120/90 halves':>18s}  verdict"]
    for root in roots:
        for d in (-1, 1):
            cols = []
            for w in WINDOWS:
                c = cells.get((root, d, w))
                if not c or not c.all:
                    cols.append("-".center(22))
                    continue
                xs = c.all
                cols.append(f"{sum(x > 0 for x in xs) / len(xs):4.0%} {statistics.mean(xs):6.0f}$ n={len(xs):3d}".center(22))
            key = cells.get((root, d, KEY_WINDOW))
            h1 = key.half_mean(True) if key else None
            h2 = key.half_mean(False) if key else None
            halves = f"{h1 or 0:7.0f}$ {h2 or 0:7.0f}$" if key else "-"
            lines.append(f"{root:7s} {d:+3d}  " + "  ".join(cols) + f"  {halves:>18s}  "
                         + ("CANDIDATE" if is_candidate(cells, root, d) else ""))
    return "\n".join(lines)


def nth_weekday(year: int, month: int, n: int) -> date:
    """n-th Monday-to-Friday of a month (exchange holidays ignored: an entry
    or exit then slides to the next print, which `_Lookup` does anyway)."""
    d, seen = date(year, month, 1), 0
    while True:
        if d.weekday() < 5:
            seen += 1
            if seen == n:
                return d
        d += timedelta(days=1)


def roll_window_test(data, roots: Sequence[str], entry_bd: int = 2, exit_bd: int = 9):
    """Sell the front, buy the next active month, from the entry_bd-th to the
    exit_bd-th business day of the month before the front's delivery month -
    around the GSCI roll (5th-9th business day). Returns {root: {year: [net]}}."""
    out: Dict[str, Dict[int, List[float]]] = {}
    for root in roots:
        by_year: Dict[int, List[float]] = defaultdict(list)
        for sd in next_active_pairs(root):
            cost = round_trip_cost(sd.roots)
            month = MONTH_CODES.index(sd.legs[0][1]) + 1
            for c in build_cycles(sd, data, None):
                y, m = (c.year, month - 1) if month > 1 else (c.year - 1, 12)
                if y < FIRST_YEAR:
                    continue
                lk = _Lookup(c.values)
                e = lk.at_or_after(nth_weekday(y, m, entry_bd))
                x = lk.at_or_after(nth_weekday(y, m, exit_bd))
                if e and x and x[0] > e[0] and x[0] <= c.anchor and (x[0] - e[0]).days < 20:
                    by_year[e[0].year].append(-(x[1] - e[1]) - cost)
        out[root] = by_year
    return out


def roll_report(results) -> str:
    lines = ["GSCI-roll window: sell front / buy next, 2nd -> 9th business day of the month "
             "before delivery; costs are an estimate"]
    for root, by_year in results.items():
        halves = []
        for first in (True, False):
            xs = [x for y, v in by_year.items() for x in v if (y <= SPLIT_YEAR) == first]
            halves.append(xs)
        cells = []
        ok = True
        for xs in halves:
            if not xs:
                cells.append("n=0".ljust(30)); ok = False; continue
            w, m = sum(x > 0 for x in xs) / len(xs), statistics.mean(xs)
            ok &= w >= 0.55 and m > 0
            cells.append(f"n={len(xs):3d} win {w:4.0%} mean {m:6.0f}$".ljust(30))
        lines.append(f"  {root:3s} 2011-18: {cells[0]} 2019-26: {cells[1]} {'CANDIDATE' if ok else ''}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory")
    ap.add_argument("--roots", nargs="+")
    ap.add_argument("--roll", action="store_true", help="GSCI-roll window test instead")
    a = ap.parse_args(argv)
    data = read_dir(a.directory)
    roots = [r for r in (a.roots or ACTIVE) if r in {k[0] for k in data}]
    if a.roll:
        print(roll_report(roll_window_test(data, roots)))
        return 0
    print(report(run(data, roots), roots))
    return 0


if __name__ == "__main__":
    sys.exit(main())
