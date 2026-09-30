#!/usr/bin/env python3
"""Download daily bars for every contract of a commodity basket from Databento.

    export DATABENTO_API_KEY=...      # never in code, never in chat
    python -m research.funding_arb.fetch_databento              # cost estimate only
    python -m research.funding_arb.fetch_databento --confirm    # download

Source: CME Globex (GLBX.MDP3), history from 2010-06-06, every listed expiry,
so calendar and inter-commodity spreads can be built from the contracts a
trader would actually hold - not from a rolled continuous series, whose roll
gaps would show up as fake seasonal moves.

Databento bills historical data by volume. The default run is FREE: it asks the
API what the request would cost and stops. Nothing is downloaded, and nothing
is charged, without --confirm. New accounts carry $125 of free credit.

Known limitation: ohlcv-1d bars close at UTC midnight, not at the exchange
settlement, so the close is the last trade of the evening session. On thin
deferred months that trade can be hours older than the other leg's. Good enough
for a screen; finalists should be re-checked on settlement prices
(`statistics` schema).

Reads market data only. No order code, by design (see CLAUDE.md, Boundaries).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from typing import List, Optional, Sequence

from concurrent.futures import ThreadPoolExecutor

from .futures_contracts import MONTH_CODES, SPECS, parse_raw_symbol, write_rows, check_plausible

DATASET = "GLBX.MDP3"
SCHEMA = "ohlcv-1d"
START = "2010-06-06"
#: Refuse to download past this without being told to. The whole basket is
#: expected to cost a small fraction of it; a bigger number means the request
#: is not what we think it is (e.g. spreads or options pulled in by mistake).
DEFAULT_MAX_USD = 25.0


def _client():
    try:
        import databento as db
    except ImportError:
        sys.exit("databento is not installed. Run: pip install databento")
    if not os.environ.get("DATABENTO_API_KEY"):
        sys.exit("DATABENTO_API_KEY is not set. Put it in the environment, not in a file in the repo.")
    return db.Historical()


def outright_symbols(root: str, last_year: int) -> List[str]:
    """Every raw outright symbol a root can have: 'CLF0' ... 'CLZ9', plus the
    two-digit form 'CLF20' ... for 2020 onwards.

    CME writes one year digit for the next ten years and two digits beyond.
    NG switched to two digits for every contract in May 2025: asking only for
    one digit silently ended the NG history there, with no error (measured).

    Asked for by raw symbol rather than by parent ('CL.FUT') because the parent
    also carries every exchange-listed calendar spread and strategy: for CL that
    is 7x the data and 7x the bill (measured: $7.69 vs $1.08, 2010-2026), none
    of it used - spreads are rebuilt from the outright legs. Databento resolves
    raw symbols per date, so 'CLM5' maps to June 2015 in 2015 and June 2025 in
    2025.
    """
    one = [f"{root}{m}{d}" for m in MONTH_CODES for d in range(10)]
    two = [f"{root}{m}{y % 100:02d}" for m in MONTH_CODES for y in range(2020, last_year + 13)]
    return one + two


def _request(root: str, end: str, start: str = START) -> dict:
    return dict(dataset=DATASET, symbols=outright_symbols(root, int(end[:4])),
                stype_in="raw_symbol", schema=SCHEMA, start=start, end=end)


def estimate(client, roots: Sequence[str], end: str, start: str = START) -> List[tuple]:
    # Each metadata call takes 25-45 s server-side; in parallel the basket
    # takes a couple of minutes instead of half an hour.
    def one(root):
        kw = _request(root, end, start)
        return root, client.metadata.get_cost(**kw), client.metadata.get_billable_size(**kw)
    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one, roots))


def download(client, root: str, end: str, out_dir: str, start: str = START,
             suffix: str = "") -> int:
    store = client.timeseries.get_range(**_request(root, end, start))
    df = store.to_df()  # prices as floats, raw symbols mapped (e.g. 'CLM5')
    rows = []
    skipped = 0
    for ts, rec in df.iterrows():
        d = ts.date()
        key = parse_raw_symbol(str(rec["symbol"]), d)
        if key is None or key[0] != root:
            skipped += 1   # anything that is not an outright of this root
            continue
        rows.append((d, key, float(rec["close"]), float(rec["volume"])))
    by_contract = {}
    for d, key, close, _ in rows:
        by_contract.setdefault(key, {})[d] = close
    check_plausible(root, {i: v for i, v in enumerate(c for _, _, c, _ in rows)}, root)
    n = write_rows(os.path.join(out_dir, f"{root}{suffix}.csv.gz"), rows)
    print(f"  {root}: {n} bars over {len(by_contract)} contracts ({skipped} spread/other bars skipped)")
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roots", nargs="+", default=sorted(SPECS), help="product roots (default: whole basket)")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--suffix", default="", help="file name suffix, to add a date range "
                    "to a root already downloaded (the reader merges every file)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "databento"))
    ap.add_argument("--confirm", action="store_true", help="actually download (billed)")
    ap.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    a = ap.parse_args(argv)

    unknown = [r for r in a.roots if r not in SPECS]
    if unknown:
        sys.exit(f"no spec for {unknown}: add them to futures_contracts.SPECS first")
    client = _client()
    est = estimate(client, a.roots, a.end, a.start)
    total = sum(c for _, c, _ in est)
    for root, cost, size in est:
        print(f"  {root:3s} {size / 1e6:8.2f} MB  ${cost:7.2f}")
    print(f"total ${total:.2f} for {len(est)} roots, {a.start} -> {a.end} ({SCHEMA})")
    if not a.confirm:
        print("estimate only - nothing downloaded, nothing billed. Re-run with --confirm.")
        return 0
    if total > a.max_usd:
        sys.exit(f"estimate ${total:.2f} exceeds --max-usd {a.max_usd:.2f}; refusing")
    for root in a.roots:
        download(client, root, a.end, a.out, a.start, a.suffix)
    return 0


if __name__ == "__main__":
    sys.exit(main())
