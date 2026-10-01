"""Per-contract futures history: product specs, symbol parsing, CSV storage.

Shared by `fetch_databento.py` (writes) and `seasonal_scan.py` (reads). Kept
free of third-party imports so the scanner and its tests run on the standard
library alone.
"""

from __future__ import annotations

import csv
import gzip
import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

MONTH_CODES = "FGHJKMNQUVXZ"


@dataclass(frozen=True)
class Spec:
    root: str
    sector: str
    dollars_per_point: float   # $ P&L per 1.0 move in the quoted price, one contract
    tick_value: float          # $ per minimum tick, one contract
    plausible: Tuple[float, float]  # quoted-price range; outside it the units are wrong


#: CME Group products on Databento GLBX.MDP3. Grains and livestock are quoted in
#: cents, so their $/point is per cent. `plausible` exists because a units slip
#: (cents read as dollars, a display factor applied twice) does not raise - it
#: produces a spread 100x too large that still looks like a spread.
SPECS: Dict[str, Spec] = {s.root: s for s in [
    Spec("CL", "energy", 1000, 10.0, (-50, 250)),
    Spec("BZ", "energy", 1000, 10.0, (10, 250)),
    Spec("HO", "energy", 42000, 4.2, (0.2, 8)),
    Spec("RB", "energy", 42000, 4.2, (0.2, 8)),
    Spec("NG", "energy", 10000, 10.0, (0.5, 25)),
    Spec("GC", "metals", 100, 10.0, (500, 10000)),
    Spec("SI", "metals", 5000, 25.0, (5, 200)),
    Spec("HG", "metals", 25000, 12.5, (1, 12)),
    Spec("PL", "metals", 50, 5.0, (200, 5000)),
    Spec("PA", "metals", 100, 5.0, (200, 5000)),
    Spec("ZC", "grains", 50, 12.5, (150, 1200)),
    Spec("ZW", "grains", 50, 12.5, (250, 1600)),
    Spec("KE", "grains", 50, 12.5, (250, 1600)),
    Spec("ZS", "grains", 50, 12.5, (600, 2200)),
    Spec("ZM", "grains", 100, 10.0, (150, 800)),
    Spec("ZL", "grains", 600, 6.0, (15, 120)),
    Spec("ZO", "grains", 50, 12.5, (100, 1000)),
    Spec("ZR", "grains", 2000, 10.0, (5, 40)),
    Spec("LE", "livestock", 400, 10.0, (60, 350)),
    Spec("GF", "livestock", 500, 12.5, (80, 450)),
    Spec("HE", "livestock", 400, 10.0, (25, 180)),
]}

#: IBKR-like commission, all fees, per contract per side. An estimate: it is
#: labelled as one in every report that uses it.
COMMISSION_PER_SIDE = 2.5

ContractKey = Tuple[str, str, int]      # (root, month code, delivery year)
DailySeries = Dict[date, float]

_RAW = re.compile(r"^([A-Z]{2})([FGHJKMNQUVXZ])(\d{1,2})$")


def parse_raw_symbol(symbol: str, seen_on: date) -> Optional[ContractKey]:
    """'CLM5' seen trading on 2015-01-21 -> ('CL', 'M', 2015).

    CME raw symbols carry one year digit, so the decade comes from the trade
    date: the delivery year is the first year >= the trade year ending in that
    digit. Outright contracts are never listed ten years out in the products
    above, so this is unambiguous. Spreads ('CLM5-CLN5') and options return None.
    """
    m = _RAW.match(symbol.strip())
    if not m or m.group(1) not in SPECS:
        return None
    digits = m.group(3)
    if len(digits) == 2:
        year = 2000 + int(digits)
    else:
        year = seen_on.year + (int(digits) - seen_on.year) % 10
    return m.group(1), m.group(2), year


def resolve_contract(symbols: Iterable[str], last_seen: date) -> Optional[ContractKey]:
    """Delivery year of ONE exchange instrument, from every raw symbol it was
    listed under and the last day it was seen.

    CME reuses a one-digit symbol ten years later: once NG March 2013 expired,
    'NGH3' became NG March 2023. Products listed more than ten years out (NG,
    CL) settle that new contract every day, so resolving by the date of each
    print (`parse_raw_symbol`) glues 2023 prices onto the 2013 series (measured
    on NG settlements). Per instrument the answer is unambiguous: a two-digit
    symbol gives the year outright; otherwise it is the first year with that
    last digit whose delivery month had not ended when the instrument was last seen.
    """
    one_digit = None
    for sym in symbols:
        m = _RAW.match(sym.strip())
        if not m or m.group(1) not in SPECS:
            continue
        if len(m.group(3)) == 2:
            return m.group(1), m.group(2), 2000 + int(m.group(3))
        one_digit = (m.group(1), m.group(2), int(m.group(3)))
    if one_digit is None:
        return None
    root, month, digit = one_digit
    y = last_seen.year - 1
    while True:
        first = date(y, MONTH_CODES.index(month) + 1, 1)
        end = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if y % 10 == digit and end >= last_seen:
            return root, month, y
        y += 1


def round_trip_cost(roots: Iterable[str]) -> float:
    """Commissions in and out on every leg plus one tick of slippage per leg."""
    return sum(2 * COMMISSION_PER_SIDE + SPECS[r].tick_value for r in roots)


def check_plausible(root: str, series: DailySeries, where: str = "") -> None:
    lo, hi = SPECS[root].plausible
    bad = [v for v in series.values() if not lo <= v <= hi]
    # A handful of bad prints is a data problem worth dropping; most of them
    # means the units are wrong and every spread built on this root is too.
    if len(bad) > max(3, len(series) // 100):
        raise ValueError(f"{where or root}: {len(bad)} of {len(series)} closes outside "
                         f"{lo}-{hi} (e.g. {bad[:3]}): wrong units or wrong product")


FIELDS = ["date", "root", "month", "year", "close", "volume"]


def write_rows(path: str, rows: Iterable[Tuple[date, ContractKey, float, float]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for d, (root, month, year), close, volume in rows:
            w.writerow([d.isoformat(), root, month, year, repr(close), repr(volume)])
            n += 1
    return n


#: Trailing prints this far from the rest of a contract's history belong to a
#: different contract that inherited the symbol: CME reuses 'CLM9' for June 2029
#: once June 2019 expires, and a stray June-2029 trade landed in the June-2019
#: series a month after it expired (measured on CL). Near expiry every contract
#: here trades daily, so a gap this long before the final prints is never real.
STRAGGLER_GAP_DAYS = 14


def drop_stragglers(series: DailySeries, gap_days: int = STRAGGLER_GAP_DAYS) -> DailySeries:
    days = sorted(series)
    cut = len(days)
    for i in range(len(days) - 1, 0, -1):
        if (days[i] - days[i - 1]).days > gap_days:
            cut = i
        elif i < len(days) - 5:
            break   # only the tail is inspected; early-life gaps are normal
    return {d: series[d] for d in days[:cut]}


def read_dir(path: str) -> Dict[ContractKey, DailySeries]:
    """Every *.csv.gz written by `write_rows`, merged per contract (a root may
    span several files), stragglers removed, validated per root."""
    out: Dict[ContractKey, DailySeries] = {}
    for name in sorted(os.listdir(path)):
        if not name.endswith(".csv.gz"):
            continue
        with gzip.open(os.path.join(path, name), "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["root"], r["month"], int(r["year"]))
                out.setdefault(key, {})[date.fromisoformat(r["date"])] = float(r["close"])
    out = {k: drop_stragglers(s) for k, s in out.items()}
    by_root: Dict[str, List[float]] = {}
    for k, s in out.items():
        by_root.setdefault(k[0], []).extend(s.values())
    for root, values in by_root.items():
        check_plausible(root, dict(enumerate(values)), root)
    return out


def roots_in(data: Dict[ContractKey, DailySeries]) -> List[str]:
    return sorted({k[0] for k in data})
