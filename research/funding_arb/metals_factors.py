#!/usr/bin/env python3
"""Gold and copper calendar spreads against an external anchor: T-bill yields.

A calendar spread on a storable metal prices the cost of carrying the metal
from one delivery month to the next. For gold that cost is almost entirely the
financing rate, so the carry implied by the two nearest active contracts tracks
the 3-month T-bill closely (MEMORY.md §11-nonies: level correlation 0.95). For
copper, storage and above all inventory tightness matter more (0.46).

The signal is the gap between the two: residual = implied carry - T-bill. When
it is unusually far from its own one-year norm, the spread is expected to move
back. Parameters were fixed before the first run (MEMORY.md §11-nonies) and
are not tuned here.

    python -m research.funding_arb.metals_factors research/funding_arb/data/databento \
        --fred research/funding_arb/data/fred
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .futures_contracts import MONTH_CODES, SPECS, ContractKey, DailySeries, read_dir, round_trip_cost

ACTIVE = {"GC": "GJMQVZ", "HG": "HKNUZ"}
#: Roll to the next pair this many days before the front contract's last print:
#: the delivery period is thin and its prices are not the carry market.
MIN_DAYS_TO_EXPIRY = 45
Z_WINDOW = 252
Z_ENTRY = 1.5
MAX_HOLD = 30
SPLIT_YEAR = 2018


def load_fred(path: str, series_id: str) -> Dict[date, float]:
    out = {}
    with open(os.path.join(path, f"{series_id}.csv"), newline="") as fh:
        for r in csv.DictReader(fh):
            v = r.get(series_id, "")
            if v not in ("", "."):
                out[date.fromisoformat(r["observation_date"])] = float(v)
    return out


def _delivery(key: ContractKey) -> date:
    return date(key[2], MONTH_CODES.index(key[1]) + 1, 15)


@dataclass
class CarryPoint:
    implied: float          # annualised, decimal
    front: ContractKey
    back: ContractKey


def implied_carry(data: Dict[ContractKey, DailySeries], root: str) -> Dict[date, CarryPoint]:
    keys = sorted((k for k in data if k[0] == root and k[1] in ACTIVE[root]), key=_delivery)
    last = {k: max(data[k]) for k in keys}
    out: Dict[date, CarryPoint] = {}
    for d in sorted({d for k in keys for d in data[k]}):
        live = [k for k in keys if last[k] > d + timedelta(days=MIN_DAYS_TO_EXPIRY) and d in data[k]]
        if len(live) < 2:
            continue
        a, b = live[0], live[1]
        years = (_delivery(b) - _delivery(a)).days / 365
        pa, pb = data[a][d], data[b][d]
        if years > 0 and pa > 0 and pb > 0:
            out[d] = CarryPoint(math.log(pb / pa) / years, a, b)
    return out


def residual_z(carry: Dict[date, CarryPoint], bill: Dict[date, float]) -> Dict[date, float]:
    """z-score of (implied carry - T-bill) against its trailing Z_WINDOW days,
    using only days before each date (the day itself is excluded from its norm)."""
    days = [d for d in sorted(carry) if d in bill]
    res = [carry[d].implied - bill[d] / 100 for d in days]
    out = {}
    for i in range(Z_WINDOW, len(days)):
        hist = res[i - Z_WINDOW:i]
        sd = statistics.stdev(hist)
        if sd > 0:
            out[days[i]] = (res[i] - statistics.mean(hist)) / sd
    return out


@dataclass
class Trade:
    entry: date
    exit: date
    direction: int          # +1 = long front / short back
    net: float


def backtest(data: Dict[ContractKey, DailySeries], root: str, carry: Dict[date, CarryPoint],
             z: Dict[date, float]) -> List[Trade]:
    """z <= -Z_ENTRY: carry unusually low vs rates -> sell front, buy back (-1).
    z >= +Z_ENTRY: the opposite (+1). Exit when z crosses 0, after MAX_HOLD
    trading days, or before the front leg's delivery period."""
    mult = SPECS[root].dollars_per_point
    cost = round_trip_cost([root, root])
    days = sorted(z)
    trades: List[Trade] = []
    i = 0
    while i < len(days):
        d, zi = days[i], z[days[i]]
        if abs(zi) < Z_ENTRY:
            i += 1
            continue
        direction = 1 if zi > 0 else -1
        a, b = carry[d].front, carry[d].back
        # The signal uses day d's close, so the trade is entered at the next
        # close both legs print - acting on the same close would be optimistic.
        k = i + 1
        while k < len(days) and (days[k] not in data[a] or days[k] not in data[b]):
            k += 1
        if k >= len(days):
            break
        i, d = k, days[k]
        entry_px = data[a][d] - data[b][d]
        j, exit_day = i, d
        while j + 1 < len(days):
            j += 1
            dj = days[j]
            if dj not in data[a] or dj not in data[b]:
                continue
            exit_day = dj
            near_expiry = max(data[a]) <= dj + timedelta(days=MIN_DAYS_TO_EXPIRY // 2)
            if direction * z[dj] <= 0 or j - i >= MAX_HOLD or near_expiry:
                break
        if exit_day == d:
            i += 1
            continue
        pnl = direction * ((data[a][exit_day] - data[b][exit_day]) - entry_px) * mult - cost
        trades.append(Trade(d, exit_day, direction, pnl))
        i = j + 1
    return trades


def max_drawdown(pnls: Sequence[float]) -> float:
    peak = equity = dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    return dd


def summary(label: str, trades: Sequence[Trade]) -> Tuple[str, bool]:
    if not trades:
        return f"  {label:12s} no trades", False
    xs = [t.net for t in trades]
    total, dd = sum(xs), max_drawdown(xs)
    win = sum(x > 0 for x in xs) / len(xs)
    ok = total > 0 and win >= 0.55 and total >= 2 * dd
    return (f"  {label:12s} n={len(xs):3d} win {win:5.1%} total {total:8.0f} $ mean {statistics.mean(xs):6.0f} $ "
            f"max drawdown {dd:7.0f} $ worst {min(xs):7.0f} $  -> {'PASS' if ok else 'fail'}"), ok


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory")
    ap.add_argument("--fred", required=True, help="folder with FRED CSVs (DTB3.csv)")
    a = ap.parse_args(argv)
    data = read_dir(a.directory)
    bill = load_fred(a.fred, "DTB3")
    print(f"carry-anomaly rule: |z| >= {Z_ENTRY} on {Z_WINDOW}-day residual vs 3m T-bill, "
          f"exit at z=0 or {MAX_HOLD} days; 1 spread; costs are an estimate")
    for root in ACTIVE:
        carry = implied_carry(data, root)
        z = residual_z(carry, bill)
        trades = backtest(data, root, carry, z)
        first = [t for t in trades if t.entry.year <= SPLIT_YEAR]
        second = [t for t in trades if t.entry.year > SPLIT_YEAR]
        l1, ok1 = summary(f"<= {SPLIT_YEAR}", first)
        l2, ok2 = summary(f">  {SPLIT_YEAR}", second)
        print(f"{root}: {'PASSES' if ok1 and ok2 else 'does not pass'} the pre-set criterion\n{l1}\n{l2}")
        for side, lab in ((-1, "sell front"), (1, "buy front")):
            xs = [t.net for t in trades if t.direction == side]
            if xs:
                print(f"    {lab:10s} n={len(xs):3d} win {sum(x > 0 for x in xs) / len(xs):5.1%} "
                      f"mean {statistics.mean(xs):6.0f} $")
    return 0


if __name__ == "__main__":
    sys.exit(main())
