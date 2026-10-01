#!/usr/bin/env python3
"""Weekly alert: which calendar spreads the tested rule says to open or close.

ALERTS ONLY. This module computes dates and reads public data; it does not
connect to a broker and contains no order code, by design (CLAUDE.md,
Boundaries). Orders, if any, are placed by a person.

The rule (MEMORY.md §11-decies and the scarcity sections):
* products: copper (HG), natural gas (NG), feeder cattle (GF), rough rice (ZR);
* for each liquid delivery month, SELL that contract and BUY the next liquid
  month, one contract each, from 180 days before the last day a retail account
  can hold the front, for 90 days;
* NG: do not open if EIA storage was more than 5% below the same week of
  earlier years in the last report before the entry day;
* ZR: do not open if WASDE projected stocks-to-use was 15% or more below the
  two earlier years in the last report before the entry day;
* GF: no filter (the cattle outlook failed its test); larger drawdowns, smaller size.

It keeps no state: open positions are re-derived from the rule each run, using
the indicator values that were public on each entry day.

    python -m research.funding_arb.signals                 # report for today
    python -m research.funding_arb.signals --date 2026-11-02 --days 14
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .fixed_rule_scan import ACTIVE
from .futures_contracts import MONTH_CODES
from .scarcity import EIA_STORAGE_URL, Published, parse_eia_storage, parse_rice, storage_deficit

PRODUCTS = ("HG", "NG", "GF", "ZR")
NAMES = {"HG": "Rame COMEX", "NG": "Gas naturale NYMEX", "GF": "Bovini da ingrasso CME",
         "ZR": "Riso CBOT"}
ENTRY_DAYS_BEFORE = 180
HOLD_DAYS = 90
NG_STORAGE_LIMIT = -0.05
ZR_STOCKS_LIMIT = -0.15
ESMIS = "https://esmis.nal.usda.gov"
UA = {"User-Agent": "Mozilla/5.0 (research alert; low rate)"}

Contract = Tuple[str, str, int]


def _first_of(month: str, year: int) -> date:
    return date(year, MONTH_CODES.index(month) + 1, 1)


def _business_days_before(d: date, n: int) -> date:
    while n:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def last_retail_day(c: Contract) -> date:
    """Last day a retail account can hold the contract, from exchange rules.

    HG and ZR are physically delivered and trade into the delivery month: out by
    the end of the month before it (first notice day). NG stops trading three
    business days before the delivery month. GF is cash-settled and trades to
    the last Thursday of the contract month. Within a couple of days of the real
    calendar, which is all a 180-day-ahead entry needs; holidays are ignored.
    """
    root, month, year = c
    first = _first_of(month, year)
    if root in ("HG", "ZR"):
        return first - timedelta(days=1)
    if root == "NG":
        return _business_days_before(first, 3)
    if root == "GF":
        return _feeder_cattle_last_day(first)
    raise ValueError(root)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    return date(year, month, (h + l - 7 * m + 114) % 31 + 1)


def _feeder_cattle_last_day(first: date) -> date:
    """Last Thursday of the month, moved earlier around US holidays.

    Checked against every GF contract 2011-2026 (the plain last-Thursday rule was
    off by a week in 30 of 134 contracts): November ends the Thursday before
    Thanksgiving; May the Thursday before when Memorial Day falls in the same
    week; March/April the Thursday before Good Friday when it falls after the
    15th of the contract month.
    """
    nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    last_thu = nxt - timedelta(days=1)
    while last_thu.weekday() != 3:
        last_thu -= timedelta(days=1)
    if first.month == 11:
        thanksgiving = first + timedelta(days=(3 - first.weekday()) % 7 + 21)
        return thanksgiving - timedelta(days=7)
    if first.month == 5:
        memorial = nxt - timedelta(days=1)
        while memorial.weekday() != 0:
            memorial -= timedelta(days=1)
        if (last_thu - memorial).days == 3:
            return last_thu - timedelta(days=7)
    good_friday = _easter(first.year) - timedelta(days=2)
    if good_friday.month == first.month and good_friday.day > 15:
        return good_friday - timedelta(days=1)
    return last_thu


def _next_business_day(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


@dataclass
class Trade:
    root: str
    front: Contract
    back: Contract
    entry: date
    exit: date
    blocked: Optional[str] = None      # reason the scarcity filter blocks it

    def legs(self) -> str:
        f, b = self.front, self.back
        return (f"VENDI 1 {f[0]}{f[1]}{f[2] % 100:02d} ({MONTH_NAMES[f[1]]} {f[2]}) / "
                f"COMPRA 1 {b[0]}{b[1]}{b[2] % 100:02d} ({MONTH_NAMES[b[1]]} {b[2]})")


MONTH_NAMES = dict(zip(MONTH_CODES, "gen feb mar apr mag giu lug ago set ott nov dic".split()))


def schedule(root: str, years: Sequence[int]) -> List[Trade]:
    months = ACTIVE[root]
    out = []
    for y in years:
        for i, m in enumerate(months):
            front = (root, m, y)
            back = (root, months[(i + 1) % len(months)], y + (1 if i + 1 == len(months) else 0))
            entry = _next_business_day(last_retail_day(front) - timedelta(days=ENTRY_DAYS_BEFORE))
            out.append(Trade(root, front, back, entry, _next_business_day(entry + timedelta(days=HOLD_DAYS))))
    return sorted(out, key=lambda t: t.entry)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    last = None
    for attempt in range(4):
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:  # transient network errors: retry, then fail loudly
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"cannot fetch {url}: {last}")


def fetch_storage() -> Published:
    return Published(storage_deficit(parse_eia_storage(_get(EIA_STORAGE_URL))))


def fetch_rice(since: date, until: date) -> Published:
    """WASDE rice stocks-to-use tightness for every report released in [since, until]."""
    values: Dict[date, float] = {}
    y, m = since.year, since.month
    while (y, m) <= (until.year, until.month):
        html = _get(f"{ESMIS}/publication/world-agricultural-supply-and-demand-estimates?date={y}-{m:02d}")
        body = html.decode("utf-8", "ignore")
        body = body[body.find("<tbody"):body.find("</tbody>")]
        for row in body.split("<tr")[1:]:
            d = re.search(r'datetime="(\d{4}-\d{2}-\d{2})', row)
            t = re.search(r'href="([^"]+\.txt)"', row)
            if d and t:
                parsed = parse_rice(_get(ESMIS + t.group(1)).decode("utf-8", "ignore"))
                if parsed:
                    values[date.fromisoformat(d.group(1))] = parsed[0] / parsed[1] - 1
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        time.sleep(0.8)
    return Published(values)


def apply_filters(trades: List[Trade], storage: Optional[Published], rice: Optional[Published]) -> None:
    for t in trades:
        if t.root == "NG" and storage is not None:
            v = storage.as_of(t.entry)
            if v is not None and v <= NG_STORAGE_LIMIT:
                t.blocked = f"scorte gas {v:+.1%} rispetto agli anni precedenti (limite {NG_STORAGE_LIMIT:+.0%})"
        if t.root == "ZR" and rice is not None:
            v = rice.as_of(t.entry)
            if v is not None and v <= ZR_STOCKS_LIMIT:
                t.blocked = f"scorte riso previste {v:+.0%} rispetto ai 2 anni prima (limite {ZR_STOCKS_LIMIT:+.0%})"


def report(today: date, days: int, trades: List[Trade], storage: Optional[Published],
           rice: Optional[Published]) -> str:
    horizon = today + timedelta(days=days)
    to_open = [t for t in trades if today <= t.entry <= horizon]
    to_close = [t for t in trades if today <= t.exit <= horizon and t.entry < today and not t.blocked]
    open_now = [t for t in trades if t.entry < today < t.exit and not t.blocked]
    lines = [f"AVVISI SPREAD — {today.isoformat()} (prossimi {days} giorni). Solo avvisi: nessun ordine viene inviato.", ""]
    s = storage.as_of(today + timedelta(days=1)) if storage else None
    r = rice.as_of(today + timedelta(days=1)) if rice else None
    lines.append("Indicatori di scarsità (ultimo dato pubblicato):")
    lines.append(f"  gas, scorte EIA vs anni precedenti: {s:+.1%}" if s is not None else "  gas: dato non disponibile")
    lines.append(f"  riso, scorte/consumi WASDE vs 2 anni prima: {r:+.0%}" if r is not None else "  riso: dato non disponibile")
    lines.append("")
    lines.append("DA APRIRE:" if to_open else "DA APRIRE: niente in questo periodo.")
    for t in to_open:
        tag = f"NON APRIRE — {t.blocked}" if t.blocked else "APRI"
        lines.append(f"  {t.entry} {NAMES[t.root]}: {tag}\n      {t.legs()}\n      uscita prevista {t.exit}"
                     + ("\n      nota: drawdown storici alti, dimensione ridotta" if t.root == "GF" and not t.blocked else ""))
    lines.append("DA CHIUDERE:" if to_close else "DA CHIUDERE: niente in questo periodo.")
    for t in to_close:
        lines.append(f"  {t.exit} {NAMES[t.root]}: CHIUDI {t.legs().replace('VENDI', 'riacquista').replace('COMPRA', 'rivendi')}"
                     f" (aperta il {t.entry})")
    lines.append(f"APERTE secondo la regola ({len(open_now)}):")
    for t in open_now:
        lines.append(f"  {NAMES[t.root]}: {t.legs()} — dal {t.entry}, uscita {t.exit}")
    lines += ["", "Ricorda: le date di scadenza sono calcolate dalle regole di borsa e possono differire di 1-2 giorni"
              " (festività); verifica sul contratto in TWS. Costi e margini reali non sono ancora stati misurati."]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--date", type=date.fromisoformat, default=date.today())
    ap.add_argument("--days", type=int, default=7, help="look-ahead window for open/close alerts")
    ap.add_argument("--offline", action="store_true", help="skip the scarcity data downloads")
    a = ap.parse_args(argv)
    years = range(a.date.year - 1, a.date.year + 2)
    trades = [t for root in PRODUCTS for t in schedule(root, years)]
    storage = rice = None
    if not a.offline:
        try:
            storage = fetch_storage()
        except RuntimeError as e:
            print(f"ATTENZIONE: scorte gas non scaricate ({e}); filtro NG non applicato", file=sys.stderr)
        try:
            rice = fetch_rice(a.date - timedelta(days=HOLD_DAYS + 45), a.date + timedelta(days=a.days))
        except RuntimeError as e:
            print(f"ATTENZIONE: WASDE non scaricato ({e}); filtro riso non applicato", file=sys.stderr)
    apply_filters(trades, storage, rice)
    print(report(a.date, a.days, trades, storage, rice))
    return 0


if __name__ == "__main__":
    sys.exit(main())
