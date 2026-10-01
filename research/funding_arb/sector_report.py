#!/usr/bin/env python3
"""Seasonal spread walk-forward, reported per commodity sector.

Same selection as `seasonal_scan` (pick windows on past cycles only, trade the
next one), split into the groups a commodity desk would use. Two kinds of
spread per group:

* calendar: two delivery months of one product;
* intra-sector: two related products, same delivery month, one contract each
  (crack spreads, wheat vs corn, soy complex, gold vs silver, cattle vs hogs).
  One-to-one contracts is how the exchange lists most of these spreads, but it
  is not notional-neutral (e.g. one HO contract is ~1.5x the notional of one
  CL), so part of their P&L is directional.

The sector split was asked for after the pooled results were seen, so every
number here is descriptive. The 2018-2022 / 2023-2026 columns show whether a
sector's result holds in both halves; they are not a fresh out-of-sample test.

    python -m research.funding_arb.sector_report research/funding_arb/data/databento
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .futures_contracts import read_dir
from .metrics import wilson_interval
from .seasonal_scan import Cycle, Pick, SpreadDef, calendar_defs, inter_defs, scan, windows

SECTORS: Dict[str, Tuple[str, ...]] = {
    "metals": ("GC", "SI", "HG", "PL", "PA"),
    "petroleum": ("CL", "BZ", "HO", "RB"),
    "natural gas": ("NG",),
    "grains & oilseeds": ("ZC", "ZW", "KE", "ZS", "ZM", "ZL", "ZO", "ZR"),
    "livestock": ("LE", "GF", "HE"),
}
SPLIT_YEAR = 2022
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


@dataclass
class Trade:
    spread: str
    roots: Tuple[str, ...]
    year: int
    entry_month: int
    direction: int
    net: float


def collect(defs: Sequence[SpreadDef], data, lookback: int, min_wins: int):
    trades: List[Trade] = []

    def on_pick(sd: SpreadDef, cycles: List[Cycle], t: int, pick: Pick) -> None:
        entry = cycles[t].anchor - timedelta(days=pick.window[0])
        trades.append(Trade(sd.name, sd.roots, pick.test_year, entry.month,
                            pick.window[2], pick.oos_net))

    res = scan(defs, data, lookback, min_wins, windows(), on_pick=on_pick)
    return res, trades


def _row(label: str, xs: Sequence[float]) -> str:
    if not xs:
        return f"    {label:22s} n=0"
    k = sum(x > 0 for x in xs)
    lo, _ = wilson_interval(k, len(xs))
    return (f"    {label:22s} n={len(xs):5d} win {k / len(xs):5.1%} (Wilson low {lo:4.1%}) "
            f"mean {statistics.mean(xs):6.0f} $ median {statistics.median(xs):5.0f} $")


def describe(name: str, res, trades: List[Trade]) -> List[str]:
    if not trades:
        return [f"  {name}: no spread with enough history"]
    base = res.baseline_oos.wins / res.baseline_oos.n
    xs = [t.net for t in trades]
    s = sorted(xs)
    by_year = defaultdict(list)
    for t in trades:
        by_year[t.year].append(t.net)
    beat = sum(sum(x > 0 for x in v) / len(v) > base for v in by_year.values())
    lines = [f"  {name}: {res.spreads_tested} spreads, random-window baseline {base:.1%}",
             _row("best picks, all years", xs),
             _row(f"  <= {SPLIT_YEAR}", [t.net for t in trades if t.year <= SPLIT_YEAR]),
             _row(f"  >  {SPLIT_YEAR}", [t.net for t in trades if t.year > SPLIT_YEAR]),
             f"    years beating baseline {beat}/{len(by_year)};  per-year mean $: "
             + ", ".join(f"{y}:{statistics.mean(v):.0f}" for y, v in sorted(by_year.items())),
             f"    risk per trade: stdev {statistics.stdev(xs):.0f} $, p5 {s[len(s) // 20]:.0f} $, "
             f"worst {s[0]:.0f} $, best {s[-1]:.0f} $"]
    roots = defaultdict(list)
    for t in trades:
        roots["-".join(dict.fromkeys(t.roots))].append(t.net)
    if len(roots) > 1:
        lines.append("    by product: " + "; ".join(
            f"{r} {sum(x > 0 for x in v) / len(v):.0%}/{statistics.mean(v):.0f}$ (n={len(v)})"
            for r, v in sorted(roots.items(), key=lambda kv: -statistics.mean(kv[1]))))
    months = Counter(t.entry_month for t in trades)
    shorts = sum(t.direction < 0 for t in trades) / len(trades)
    lines.append("    entry months of the picks: " + ", ".join(
        f"{MONTHS[m - 1]} {c}" for m, c in sorted(months.items())) + f";  short-spread share {shorts:.0%}")
    return lines


def recurring(trades: List[Trade], min_years: int = 6, top: int = 5) -> List[str]:
    """Spreads picked in at least `min_years` test years, best by mean.
    Exploratory: ranking 1,000+ spreads by their out-of-sample record is itself
    a selection, so this list is where to look, not what to trade."""
    by = defaultdict(list)
    for t in trades:
        by[t.spread].append(t.net)
    rows = [(s, v) for s, v in by.items() if len(v) >= min_years]
    rows.sort(key=lambda r: -statistics.mean(r[1]))
    return [f"      {s:14s} {sum(x > 0 for x in v)}/{len(v)} years won, mean {statistics.mean(v):6.0f} $"
            for s, v in rows[:top]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory")
    ap.add_argument("--lookback", type=int, default=8)
    ap.add_argument("--min-wins", type=int, default=7)
    a = ap.parse_args(argv)
    data = read_dir(a.directory)
    have = {k[0] for k in data}
    print(f"walk-forward lookback {a.lookback}, qualify >= {a.min_wins}/{a.lookback} net wins; "
          f"costs are an estimate (commissions + 1 tick per leg)\n")
    for sector, roots in SECTORS.items():
        roots = tuple(r for r in roots if r in have)
        print(f"=== {sector.upper()} ({', '.join(roots)})")
        res, tr = collect(calendar_defs(roots), data, a.lookback, a.min_wins)
        print("\n".join(describe("calendar", res, tr)))
        print("    most recurring calendar spreads (exploratory):")
        print("\n".join(recurring(tr)) or "      none")
        if len(roots) > 1:
            res, tr = collect(inter_defs(roots), data, a.lookback, a.min_wins)
            print("\n".join(describe("intra-sector", res, tr)))
            print("    most recurring intra-sector spreads (exploratory):")
            print("\n".join(recurring(tr)) or "      none")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
