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

from .futures_contracts import SPECS, parse_raw_symbol, write_rows, check_plausible

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


def estimate(client, roots: Sequence[str], end: str) -> List[tuple]:
    rows = []
    for root in roots:
        kw = dict(dataset=DATASET, symbols=[f"{root}.FUT"], stype_in="parent",
                  schema=SCHEMA, start=START, end=end)
        cost = client.metadata.get_cost(**kw)
        size = client.metadata.get_billable_size(**kw)
        rows.append((root, cost, size))
    return rows


def download(client, root: str, end: str, out_dir: str) -> int:
    store = client.timeseries.get_range(dataset=DATASET, symbols=[f"{root}.FUT"],
                                        stype_in="parent", schema=SCHEMA, start=START, end=end)
    df = store.to_df()  # prices as floats, raw symbols mapped (e.g. 'CLM5')
    rows = []
    skipped = 0
    for ts, rec in df.iterrows():
        d = ts.date()
        key = parse_raw_symbol(str(rec["symbol"]), d)
        if key is None or key[0] != root:
            skipped += 1   # calendar spreads and other strategies listed under the parent
            continue
        rows.append((d, key, float(rec["close"]), float(rec["volume"])))
    by_contract = {}
    for d, key, close, _ in rows:
        by_contract.setdefault(key, {})[d] = close
    check_plausible(root, {i: v for i, v in enumerate(c for _, _, c, _ in rows)}, root)
    n = write_rows(os.path.join(out_dir, f"{root}.csv.gz"), rows)
    print(f"  {root}: {n} bars over {len(by_contract)} contracts ({skipped} spread/other bars skipped)")
    return n


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--roots", nargs="+", default=sorted(SPECS), help="product roots (default: whole basket)")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "data", "databento"))
    ap.add_argument("--confirm", action="store_true", help="actually download (billed)")
    ap.add_argument("--max-usd", type=float, default=DEFAULT_MAX_USD)
    a = ap.parse_args(argv)

    unknown = [r for r in a.roots if r not in SPECS]
    if unknown:
        sys.exit(f"no spec for {unknown}: add them to futures_contracts.SPECS first")
    client = _client()
    est = estimate(client, a.roots, a.end)
    total = sum(c for _, c, _ in est)
    for root, cost, size in est:
        print(f"  {root:3s} {size / 1e6:8.2f} MB  ${cost:7.2f}")
    print(f"total ${total:.2f} for {len(est)} roots, {START} -> {a.end} ({SCHEMA})")
    if not a.confirm:
        print("estimate only - nothing downloaded, nothing billed. Re-run with --confirm.")
        return 0
    if total > a.max_usd:
        sys.exit(f"estimate ${total:.2f} exceeds --max-usd {a.max_usd:.2f}; refusing")
    for root in a.roots:
        download(client, root, a.end, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
