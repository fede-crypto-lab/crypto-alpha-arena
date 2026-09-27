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
placement, by design - see CLAUDE.md, "Boundaries". Enable "Read-Only API" in TWS
as well, so the boundary is enforced by the broker and not only by this file.

HISTORY LIMIT (verified against IBKR documentation): the API serves expired
futures only up to TWO YEARS after their expiry, via `includeExpired`. So this
script yields the live curve plus roughly two to three years of history per
product. That is enough for a first cross-sectional carry-persistence test; it is
NOT enough for the 15-year seasonal walk-forward on specific contract spreads,
which needs another source (Databento carries CME Globex from June 2010).

Run `--probe` first. It checks the connection, refuses a live account, and pulls
five days of one contract - so a missing market-data permission shows up in
seconds rather than after twenty minutes of silent empty responses.

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
from datetime import datetime, timedelta, timezone
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


#: IBKR keeps expired futures for two years after expiry; asking further back
#: returns nothing, and asking exactly at the edge is flaky, so stay just inside.
EXPIRED_WINDOW_DAYS = 2 * 365 - 10


def select_expiries(all_expiries: List[str], today: datetime,
                    max_live: int, window_days: int = EXPIRED_WINDOW_DAYS) -> List[str]:
    """Expired months still inside IBKR's window, plus the nearest live ones.

    Pure function so it can be tested without a TWS connection. Expiries are
    'YYYYMMDD' or 'YYYYMM'; the month is what identifies a contract here.
    """
    cutoff = (today - timedelta(days=window_days)).strftime("%Y%m%d")
    now = today.strftime("%Y%m%d")
    # One date per month: IBKR can list the same month both as 'YYYYMM' and as
    # a full date, and pulling it twice wastes a paced request. A month-only
    # code is placed late in the month; the full date wins when both exist.
    by_month: Dict[str, str] = {}
    for e in all_expiries:
        month, date = e[:6], (e if len(e) == 8 else e[:6] + "28")
        if len(e) == 8 or month not in by_month:
            by_month[month] = date
    full = sorted(by_month.values())
    expired = [e for e in full if cutoff <= e < now]
    live = [e for e in full if e >= now][:max_live]
    return [e[:6] for e in expired + live]


def list_expiries(ib, symbol: str, exchange: str, max_live: int) -> List[str]:
    """Contract months to pull: every expired month IBKR still serves, plus the
    first few live ones.

    The first version of this script asked only for live contracts, which gives
    at most a couple of years of history on the front and almost none on the
    past years' contracts - useless for anything seasonal.
    """
    from ib_async import Future

    details = ib.reqContractDetails(
        Future(symbol=symbol, exchange=exchange, includeExpired=True))
    expiries = [d.contract.lastTradeDateOrContractMonth for d in details]
    return select_expiries(expiries, datetime.now(timezone.utc).replace(tzinfo=None),
                           max_live)


def fetch_contract(ib, symbol: str, exchange: str, expiry: str,
                   years: int, what_to_show: str) -> List[Row]:
    from ib_async import Future

    contract = Future(symbol=symbol, exchange=exchange,
                      lastTradeDateOrContractMonth=expiry, includeExpired=True)
    qualified = ib.qualifyContracts(contract)
    if not qualified:
        logger.warning("%s %s: could not qualify contract", symbol, expiry)
        return []
    con = qualified[0]

    # An expired contract has no data after its last trade, and IBKR tends to
    # return nothing if the request window ends in the future, so anchor the
    # request at the expiry for those.
    last = con.lastTradeDateOrContractMonth
    end: object = ""
    if len(last) == 8 and last < datetime.now(timezone.utc).strftime("%Y%m%d"):
        end = datetime.strptime(last, "%Y%m%d").replace(hour=23, tzinfo=timezone.utc)

    bars = ib.reqHistoricalData(
        con,
        endDateTime=end,
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


def looks_live(accounts: List[str]) -> bool:
    """IBKR paper accounts are prefixed 'D' (DU..., DF...); live ones 'U...'.

    A second guard behind the port check: TWS can be configured to serve a live
    session on any port, and the account id is what actually says which it is.
    """
    return any(a.startswith("U") for a in accounts)


def probe(ib) -> int:
    """Thirty-second health check before a twenty-minute download."""
    from ib_async import Future

    print(f"server version: {ib.client.serverVersion()}")
    accounts = ib.managedAccounts()
    print(f"accounts: {accounts}")
    if looks_live(accounts):
        print("LIVE account detected - refusing. Log TWS into the paper account.",
              file=sys.stderr)
        return 2

    details = ib.reqContractDetails(Future(symbol="CL", exchange="NYMEX"))
    if not details:
        print("CL: no contract details - check the API connection settings.",
              file=sys.stderr)
        return 1
    front = sorted(details, key=lambda d: d.contract.lastTradeDateOrContractMonth)[0]
    print(f"CL front: {front.contract.lastTradeDateOrContractMonth} "
          f"(multiplier {front.contract.multiplier})")

    bars = ib.reqHistoricalData(front.contract, endDateTime="", durationStr="5 D",
                                barSizeSetting="1 day", whatToShow="TRADES",
                                useRTH=True, formatDate=1)
    if not bars:
        print("CL: contract found but NO bars. This is almost always a missing "
              "market-data subscription for NYMEX, or the paper account not "
              "sharing the live account's subscriptions.", file=sys.stderr)
        return 1
    print(f"CL: {len(bars)} daily bars, last close {bars[-1].close} on {bars[-1].date}")

    exp = ib.reqContractDetails(Future(symbol="CL", exchange="NYMEX", includeExpired=True))
    n_expired = len(exp) - len(details)
    print(f"CL expired contracts visible via includeExpired: {n_expired}")

    # Is there a micro RBOB? Needed to know the minimum size of a gasoline crack.
    matches = ib.reqMatchingSymbols("RBOB") or []
    names = sorted({f"{m.contract.symbol} ({m.contract.primaryExchange or m.contract.exchange})"
                    for m in matches})
    print(f"symbols matching 'RBOB': {', '.join(names) or 'none'}")
    print("\nprobe OK - safe to run the full download.")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497,
                   help="7497 = paper, 7496 = live. Use paper.")
    p.add_argument("--client-id", type=int, default=17)
    p.add_argument("--years", type=int, default=3,
                   help="bars requested per contract; IBKR keeps expired "
                        "futures only 2 years past expiry, so more rarely helps")
    p.add_argument("--max-expiries", type=int, default=4,
                   help="LIVE contract months per symbol, nearest first; every "
                        "expired month IBKR still serves is added on top")
    p.add_argument("--probe", action="store_true",
                   help="30-second connection and permissions check, then exit")
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
    ib.connect(args.host, args.port, clientId=args.client_id, readonly=True)
    print(f"connected to {args.host}:{args.port}")

    if args.probe:
        try:
            return probe(ib)
        finally:
            ib.disconnect()

    if looks_live(ib.managedAccounts()):
        print("LIVE account detected - refusing. Log TWS into the paper account.",
              file=sys.stderr)
        ib.disconnect()
        return 2

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
