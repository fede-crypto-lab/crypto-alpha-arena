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
from datetime import date
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


def read_dir(path: str) -> Dict[ContractKey, DailySeries]:
    """Every *.csv.gz written by `write_rows`, validated per root."""
    out: Dict[ContractKey, DailySeries] = {}
    for name in sorted(os.listdir(path)):
        if not name.endswith(".csv.gz"):
            continue
        with gzip.open(os.path.join(path, name), "rt", newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["root"], r["month"], int(r["year"]))
                out.setdefault(key, {})[date.fromisoformat(r["date"])] = float(r["close"])
    by_root: Dict[str, List[float]] = {}
    for k, s in out.items():
        by_root.setdefault(k[0], []).extend(s.values())
    for root, values in by_root.items():
        check_plausible(root, dict(enumerate(values)), root)
    return out


def roots_in(data: Dict[ContractKey, DailySeries]) -> List[str]:
    return sorted({k[0] for k in data})
