#!/usr/bin/env python3
"""Download futures curve history from IBKR/TWS into a flat CSV.

    python fetch_ib_curves.py --port 7497 --years 5 --out data/ib_curves.csv

WRITTEN BLIND. This was authored in a container with no IBKR access and has never
run against a live TWS. Treat it as a draft to get working, not as tested code.
The parts most likely to need fixing are the contract qualification (symbol,
exchange and multiplier conventions differ per product) and the historical-data
parameters (some products want SETTLEMENT rather than TRADES, and some have far
less history than requested).

It reads market data only. It does not import, reference or contain any order
placement, by design - see CLAUDE.md, "Boundaries".

Two things it does deliberately:

* **Paces requests.** IBKR allows roughly 60 historical requests per ten minutes
  and penalises bursts. A throttle that is not handled does not raise - it
  silently drops contracts, which is how a sample stops being reproducible. This
  is the same failure that cost a day on MEXC (MEMORY.md, bug #5).
* **Appends and resumes.** Each contract is written as it arrives and an existing
  CSV is read back on startup, so an interrupted run continues instead of
  restarting. A full basket takes tens of minutes at the required pacing.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("fetch_ib_curves")

#: Basket chosen for sector spread and liquid curves. Livestock (LE/HE) is the
#: most likely to disappoint on history depth; drop it first if IBKR is stingy.
DEFAULT_BASKET: List[Tuple[str, str]] = [
    ("CL", "NYMEX"), ("NG", "NYMEX"), ("HO", "NYMEX"), ("RB", "NYMEX"),
    ("GC", "COMEX"), ("SI", "COMEX"), ("HG", "COMEX"),
    ("ZC", "CBOT"), ("ZS", "CBOT"), ("ZW", "CBOT"),
    ("LE", "CME"), ("HE", "CME"),
]

FIELDNAMES = ["date", "symbol", "exchange", "expiry", "close", "volume"]

#: IBKR permits ~60 historical requests per 10 minutes. One every 10.5s stays
#: inside that with margin. Lowering this is the fastest way to get throttled and
#: end up with a silently incomplete sample.
DEFAULT_PACING_SECONDS = 10.5


@dataclass
class Row:
    date: str
    symbol: str
    exchange: str
    expiry: str
    close: float
    volume: float


def load_existing(path: str) -> Set[Tuple[str, str]]:
    """(symbol, expiry) pairs already present, so a rerun resumes."""
    done: Set[Tuple[str, str]] = set()
    if not os.path.exists(path):
        return done
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["symbol"], row["expiry"]))
    if done:
        logger.info("resuming: %d contracts already downloaded", len(done))
    return done


def append_rows(path: str, rows: List[Row]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        for r in rows:
            writer.writerow({
                "date": r.date, "symbol": r.symbol, "exchange": r.exchange,
                "expiry": r.expiry, "close": f"{r.close:.6f}",
                "volume": f"{r.volume:.0f}",
            })


def list_expiries(ib, symbol: str, exchange: str, max_expiries: int) -> List[str]:
    """Contract months IBKR knows about, nearest first.

    `reqContractDetails` on an unqualified Future returns one entry per listed
    expiry. Deferred months thin out quickly, so only the first few are worth
    pulling - and the carry only needs two adjacent ones anyway.
    """
    from ib_async import Future

    details = ib.reqContractDetails(Future(symbol=symbol, exchange=exchange))
    expiries = sorted({d.contract.lastTradeDateOrContractMonth[:6] for d in details})
    today = datetime.utcnow().strftime("%Y%m")
    future = [e for e in expiries if e >= today]
    return future[:max_expiries]


def fetch_contract(ib, symbol: str, exchange: str, expiry: str,
                   years: int, what_to_show: str) -> List[Row]:
    from ib_async import Future

    contract = Future(symbol=symbol, exchange=exchange,
                      lastTradeDateOrContractMonth=expiry)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        logger.warning("%s %s: could not qualify contract", symbol, expiry)
        return []

    bars = ib.reqHistoricalData(
        qualified[0],
        endDateTime="",
        durationStr=f"{years} Y",
        barSizeSetting="1 day",
        whatToShow=what_to_show,
        useRTH=True,
        formatDate=1,
    )
    if not bars:
        # Almost always a market-data permission gap or a product with less
        # history than requested - not a code fault. Check the TWS log.
        logger.warning("%s %s: no bars returned", symbol, expiry)
        return []

    return [
        Row(date=str(b.date), symbol=symbol, exchange=exchange, expiry=expiry,
            close=float(b.close), volume=float(b.volume or 0))
        for b in bars
    ]


def summarise(path: str) -> None:
    """Sanity-check the download before anyone builds a carry series on it.

    Checks the three things that silently ruin the next step: too little history,
    and curves with fewer than two expiries on a given day (carry is undefined
    there), and contracts that returned nothing.
    """
    by_symbol: Dict[str, Dict[str, Set[str]]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            sym = by_symbol.setdefault(row["symbol"], {})
            sym.setdefault(row["date"], set()).add(row["expiry"])

    print(f"\n{'symbol':<9}{'days':>8}{'first':>13}{'last':>13}"
          f"{'expiries/day':>15}{'days w/ <2':>12}")
    print("-" * 70)
    for symbol in sorted(by_symbol):
        days = by_symbol[symbol]
        dates = sorted(days)
        widths = [len(v) for v in days.values()]
        thin = sum(1 for w in widths if w < 2)
        print(f"{symbol:<9}{len(dates):>8}{dates[0]:>13}{dates[-1]:>13}"
              f"{sum(widths) / len(widths):>15.1f}{thin:>12}")
    print("-" * 70)
    print(" 'days w/ <2' must be near zero: carry needs two expiries on the same")
    print(" day. A large count there means the basket or the expiry cap is wrong.")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497,
                   help="7497 = paper, 7496 = live. Use paper.")
    p.add_argument("--client-id", type=int, default=17)
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--max-expiries", type=int, default=4,
                   help="contract months per symbol, nearest first")
    p.add_argument("--what-to-show", default="TRADES",
                   help="TRADES, or SETTLEMENT where a product supports it")
    p.add_argument("--pacing", type=float, default=DEFAULT_PACING_SECONDS)
    p.add_argument("--out", default="research/funding_arb/data/ib_curves.csv")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="override the default basket, e.g. CL:NYMEX GC:COMEX")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.port == 7496:
        print("Refusing to connect to the live port. Use 7497 (paper) - this "
              "script is for data only.", file=sys.stderr)
        return 2

    try:
        from ib_async import IB
    except ImportError:
        print("ib_async is not installed. Run: pip install ib_async", file=sys.stderr)
        return 1

    basket = DEFAULT_BASKET
    if args.symbols:
        basket = [tuple(s.split(":", 1)) for s in args.symbols]

    done = load_existing(args.out)
    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id)
    print(f"connected to {args.host}:{args.port}")

    try:
        for symbol, exchange in basket:
            try:
                expiries = list_expiries(ib, symbol, exchange, args.max_expiries)
            except Exception as exc:  # noqa: BLE001 - one bad product must not stop the run
                logger.warning("%s: could not list expiries (%s)", symbol, exc)
                continue
            if not expiries:
                logger.warning("%s: no future expiries found", symbol)
                continue
            print(f"{symbol}: {len(expiries)} expiries -> {', '.join(expiries)}")

            for expiry in expiries:
                if (symbol, expiry) in done:
                    continue
                try:
                    rows = fetch_contract(ib, symbol, exchange, expiry,
                                          args.years, args.what_to_show)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s %s: %s", symbol, expiry, exc)
                    rows = []
                if rows:
                    append_rows(args.out, rows)
                    print(f"  {symbol} {expiry}: {len(rows)} bars "
                          f"({rows[0].date} .. {rows[-1].date})")
                time.sleep(args.pacing)
    finally:
        ib.disconnect()

    if os.path.exists(args.out):
        summarise(args.out)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
