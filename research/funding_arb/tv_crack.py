#!/usr/bin/env python3
"""Gasoline crack (RB x 42 - CL) on real futures, from TradingView CSV exports.

The EIA spot test (MEMORY.md §11-quater) chose one window with walk-forward:
long the gasoline crack from Jan 21-26 for ~90 days, 23/26 years out of sample.
Spot is not tradable. The question left open is whether the futures curve
already prices that seasonality in January. This script answers it on the
specific contracts a trader would hold, with the window fixed in advance - the
window was chosen on spot data, so every futures year here is out of sample.

Input: one CSV per contract, exported by hand from TradingView (chart menu ->
"Export chart data", daily bars, ISO time). File names are left as TradingView
writes them; the contract is read from the name, e.g.
"NYMEX_DL_RBM2019, 1D_1a2b3.csv" -> RB, June 2019.

    python -m research.funding_arb.tv_crack research/funding_arb/data/tv --month M

June (M) is the default: the May CL contract stops trading around April 20,
the same days the 90-day window exits, and the last days of a contract are thin.

RB is quoted in $/gallon and CL in $/barrel; one contract of each is 1,000
barrels, so a $1/bbl move in the crack is $1,000 per spread.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .seasonal_holdout import YearResult, block
from .metrics import wilson_interval

GALLONS_PER_BARREL = 42
BARRELS_PER_CONTRACT = 1000
#: IBKR commissions for two legs in and out (~$10) plus one tick of slippage on
#: each leg (CL $10, RB $4.20). Labelled an estimate wherever it is reported.
DEFAULT_COST = 30.0

_NAME = re.compile(r"(RB|CL)([FGHJKMNQUVXZ])(\d{4})")

Contract = Tuple[str, str, int]  # (root, month code, year)
DailySeries = Dict[date, float]


def contract_from_filename(name: str) -> Optional[Contract]:
    m = _NAME.search(os.path.basename(name))
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def trading_date(raw: str) -> date:
    """Session date of a daily bar, whatever timestamp convention was exported.

    TradingView stamps a daily bar either at exchange midnight or at the session
    open the evening before (17:00 Chicago). Shifting by +12h in UTC maps both
    onto the trading day; reading the date naively would put half the bars on
    the previous day and misalign the two legs.
    """
    raw = raw.strip()
    if re.fullmatch(r"\d+(\.\d+)?", raw):
        ts = datetime.fromtimestamp(float(raw), tz=timezone.utc)
    else:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            return ts.date()
    return (ts.astimezone(timezone.utc) + timedelta(hours=12)).date()


def read_closes(path: str) -> DailySeries:
    out: DailySeries = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            close = row.get("close") or row.get("Close")
            if close in (None, "", "NaN"):
                continue
            out[trading_date(row.get("time") or row["Time"])] = float(close)
    return out


def load_dir(path: str) -> Dict[Contract, DailySeries]:
    out = {}
    for name in sorted(os.listdir(path)):
        c = contract_from_filename(name)
        if c and name.lower().endswith(".csv"):
            out[c] = read_closes(os.path.join(path, name))
    return out


def crack_series(rb: DailySeries, cl: DailySeries) -> DailySeries:
    """$/bbl, on days both legs printed. No forward-fill: a stale leg is a fake spread."""
    return {d: rb[d] * GALLONS_PER_BARREL - cl[d] for d in rb.keys() & cl.keys()}


def _on_or_after(series: DailySeries, target: date, max_gap: int = 7) -> Optional[date]:
    days = sorted(d for d in series if d >= target)
    return days[0] if days and (days[0] - target).days <= max_gap else None


def _on_or_before(series: DailySeries, target: date, max_gap: int = 7) -> Optional[date]:
    days = sorted(d for d in series if d <= target)
    return days[-1] if days and (target - days[-1]).days <= max_gap else None


def trade(series: DailySeries, year: int, entry_md: Tuple[int, int],
          hold_days: int) -> Optional[Tuple[date, float, date, float]]:
    """Long the crack from the first print on/after the entry date to the last
    print on/before entry + hold. None when either end has no data."""
    entry = _on_or_after(series, date(year, *entry_md))
    if entry is None:
        return None
    exit_ = _on_or_before(series, entry + timedelta(days=hold_days))
    if exit_ is None or exit_ <= entry:
        return None
    return entry, series[entry], exit_, series[exit_]


def yearly(contracts: Dict[Contract, DailySeries], month: str,
           entry_md: Tuple[int, int], hold_days: int) -> List[Tuple[int, date, float, date, float]]:
    rows = []
    years = sorted({y for (root, m, y) in contracts if m == month})
    for y in years:
        rb, cl = contracts.get(("RB", month, y)), contracts.get(("CL", month, y))
        if not rb or not cl:
            continue
        t = trade(crack_series(rb, cl), y, entry_md, hold_days)
        if t:
            rows.append((y,) + t)
    return rows


def report(contracts: Dict[Contract, DailySeries], month: str, cost: float) -> str:
    base = yearly(contracts, month, (1, 21), 90)
    if not base:
        return f"no year has both RB{month} and CL{month} covering Jan 21 + 90 days"
    lines = [f"Long RB{month}x42 - CL{month}, Jan 21 + 90 days, cost ${cost:.0f} (estimate)",
             f"{'year':>4s} {'entry':>10s} {'$/bbl':>7s} {'exit':>10s} {'$/bbl':>7s} {'P&L $':>8s}"]
    results = []
    for y, d0, v0, d1, v1 in base:
        pnl = (v1 - v0) * BARRELS_PER_CONTRACT
        results.append(YearResult(y, v0, pnl))
        lines.append(f"{y:4d} {d0} {v0:7.2f} {d1} {v1:7.2f} {pnl:8.0f}")
    b = block("all", results, cost)
    lo, hi = wilson_interval(b.wins, b.n)
    t = f"{b.t_net:.2f}" if b.t_net is not None else "-"
    worst = min(r.profit for r in results)
    lines += ["", f"net wins {b.wins}/{b.n} ({b.win_rate:.0%}, Wilson {lo:.0%}-{hi:.0%}), "
                  f"mean gross {b.mean_gross:.0f} $, mean net {b.mean_net:.0f} $, t {t}, "
                  f"worst year {worst:.0f} $"]
    ex2020 = [r for r in results if r.year != 2020]
    if len(ex2020) < len(results) and ex2020:
        b2 = block("ex2020", ex2020, cost)
        lines.append(f"without 2020: {b2.wins}/{b2.n} net wins, mean net {b2.mean_net:.0f} $")
    lines += ["", "robustness (same verdict must hold on neighbouring windows; this is a check, not a search):"]
    for md in ((1, 11), (1, 21), (2, 1)):
        for hold in (60, 90):
            rs = [YearResult(y, v0, (v1 - v0) * BARRELS_PER_CONTRACT)
                  for y, _, v0, _, v1 in yearly(contracts, month, md, hold)]
            if rs:
                bb = block("", rs, cost)
                lines.append(f"  entry {md[1]:02d}/{md[0]:02d} hold {hold:3d}d: "
                             f"{bb.wins}/{bb.n} net wins, mean net {bb.mean_net:7.0f} $")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory", help="folder with the TradingView CSV exports")
    ap.add_argument("--month", default="M", help="contract month code for both legs (default M = June)")
    ap.add_argument("--cost", type=float, default=DEFAULT_COST, help="round-trip $ per spread")
    a = ap.parse_args(argv)
    contracts = load_dir(a.directory)
    if not contracts:
        print(f"no RB/CL contract files found in {a.directory}", file=sys.stderr)
        return 1
    have = sorted(f"{r}{m}{y}" for r, m, y in contracts)
    print(f"loaded {len(have)} contracts: {', '.join(have)}\n")
    print(report(contracts, a.month, a.cost))
    return 0


if __name__ == "__main__":
    sys.exit(main())
