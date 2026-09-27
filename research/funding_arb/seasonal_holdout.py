#!/usr/bin/env python3
"""Check a SeasonAlgo spread against the years the scanner did not look at.

SeasonAlgo ranks spreads by their win rate over the last N years (15 by default).
Those N years are the selection sample, so the win rate shown there is in-sample
by construction. The strategy page, however, also lists every earlier year of the
same fixed window - years the scanner never used to pick it. If the seasonal
pattern is real, it should show up there too (allowing for regime changes, which
this test cannot rule out). That is a free out-of-sample check.

Usage: open the strategy's backtest page, copy the per-year table (header row
"Year  Enter date  Enter price ..." and the rows below it) into a text file, then

    python -m research.funding_arb.seasonal_holdout research/funding_arb/data/zc.txt

`--cost` is the round-trip cost in dollars per spread (commissions plus one tick
of slippage); a two-leg corn calendar at IBKR is roughly $22.5 (MEMORY.md).
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass
from math import sqrt
from typing import List, Optional, Sequence

from .metrics import wilson_interval

_ROW = re.compile(r"^\s*(\d{4})\s+(\d{4}-\d{2}-\d{2})\s+(-?[\d.]+)\s+"
                  r"(\d{4}-\d{2}-\d{2})\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)")


@dataclass(frozen=True)
class YearResult:
    year: int
    enter_price: float
    profit: float


def parse_table(text: str) -> List[YearResult]:
    """Rows of SeasonAlgo's per-year table; summary rows and headers are skipped."""
    out = []
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            out.append(YearResult(int(m.group(1)), float(m.group(3)), float(m.group(7))))
    return sorted(out, key=lambda r: r.year)


@dataclass
class Block:
    label: str
    n: int
    wins: int          # strictly positive after costs; a flat year is not a win
    mean_gross: float
    mean_net: float
    t_net: Optional[float]

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else float("nan")


def block(label: str, rows: Sequence[YearResult], cost: float) -> Block:
    net = [r.profit - cost for r in rows]
    t = None
    if len(net) > 2 and statistics.stdev(net) > 0:
        t = statistics.mean(net) / (statistics.stdev(net) / sqrt(len(net)))
    return Block(label, len(rows), sum(x > 0 for x in net),
                 statistics.mean(r.profit for r in rows) if rows else float("nan"),
                 statistics.mean(net) if net else float("nan"), t)


def split(rows: Sequence[YearResult], recent: int, cost: float) -> List[Block]:
    """Selection sample (last `recent` years), the same length just before it,
    and everything before the selection sample."""
    sel = rows[-recent:]
    prior = rows[:-recent]
    return [block(f"selection (last {len(sel)})", sel, cost),
            block(f"holdout (previous {min(recent, len(prior))})", prior[-recent:], cost),
            block(f"holdout (all {len(prior)} earlier)", prior, cost)]


def would_have_qualified(rows: Sequence[YearResult], lookback: int,
                         threshold: float, cost: float) -> List[YearResult]:
    """Years in which the prior `lookback` years already cleared `threshold`
    (net of costs): the only years this exact window was selectable ex ante."""
    out = []
    for i in range(lookback, len(rows)):
        prior = rows[i - lookback:i]
        if sum(r.profit - cost > 0 for r in prior) / lookback >= threshold:
            out.append(rows[i])
    return out


def format_report(rows: Sequence[YearResult], recent: int, cost: float,
                  threshold: float) -> str:
    lines = [f"{len(rows)} years ({rows[0].year}-{rows[-1].year}), cost ${cost:.2f} per round trip",
             "", f"{'sample':30s} {'n':>3s} {'win (net)':>10s} {'Wilson 95%':>12s} "
             f"{'mean $':>8s} {'net $':>8s} {'t':>6s}"]
    for b in split(rows, recent, cost):
        lo, hi = wilson_interval(b.wins, b.n)
        t = f"{b.t_net:6.2f}" if b.t_net is not None else "     -"
        lines.append(f"{b.label:30s} {b.n:3d} {b.wins:3d}/{b.n:<3d} {b.win_rate:4.0%}"
                     f" {lo:5.0%}-{hi:4.0%} {b.mean_gross:8.0f} {b.mean_net:8.0f} {t}")
    lines += ["", "by decade:"]
    for d in range(rows[0].year // 10 * 10, rows[-1].year + 1, 10):
        dec = [r for r in rows if d <= r.year < d + 10]
        if dec:
            b = block(str(d), dec, cost)
            lines.append(f"  {d}s  {b.wins}/{b.n} won, mean net {b.mean_net:6.0f} $")
    q = would_have_qualified(rows, recent, threshold, cost)
    lines += ["", f"years in which the prior {recent} already showed >= {threshold:.0%} "
              f"net wins (selectable ex ante): {len(q)}"]
    if q:
        b = block("", q, cost)
        lines.append(f"  traded those years: {b.wins}/{b.n} won, mean net {b.mean_net:.0f} $ "
                     f"({', '.join(str(r.year) for r in q)})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("table", help="text file with the pasted per-year table ('-' = stdin)")
    ap.add_argument("--cost", type=float, default=22.5, help="round-trip $ per spread")
    ap.add_argument("--recent", type=int, default=15, help="scanner's history length")
    ap.add_argument("--threshold", type=float, default=0.8,
                    help="net win rate a scanner would have required")
    a = ap.parse_args(argv)
    text = sys.stdin.read() if a.table == "-" else open(a.table, encoding="utf-8").read()
    rows = parse_table(text)
    if len(rows) <= a.recent:
        print(f"only {len(rows)} years parsed: need more than --recent {a.recent}", file=sys.stderr)
        return 1
    print(format_report(rows, a.recent, a.cost, a.threshold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
