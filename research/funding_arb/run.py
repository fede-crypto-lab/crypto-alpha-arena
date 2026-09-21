"""CLI entry point for the funding-carry research backtest.

    python -m research.funding_arb.run --coins BTC ETH --days 365
    python -m research.funding_arb.run --sweep-entry 0.05,0.10,0.20,0.40

Nothing here places an order or needs an API key; it reads public market data and
prints numbers. That separation is deliberate - the point of this stage is to find
out whether the edge survives costs before any capital, testnet or otherwise, is
pointed at it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import List, Optional

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .costs import from_venues
from .dataset import load_pair
from .metrics import Metrics, format_report, summarize
from .strategy import StrategyParams
from .venues import get_venue

DAY_MS = 86_400_000


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="funding-arb",
        description="Backtest delta-neutral cross-venue funding carry on public data.",
    )
    p.add_argument("--coins", nargs="+", default=["BTC", "ETH"])
    p.add_argument("--venue-a", default="hyperliquid",
                   help="reference leg (hyperliquid, okx, bybit, binance)")
    p.add_argument("--venue-b", default="okx", help="second leg")
    p.add_argument("--days", type=int, default=180, help="lookback window")

    p.add_argument("--notional", type=float, default=10_000.0, help="USD per leg")
    p.add_argument("--leverage", type=float, default=3.0)

    p.add_argument("--entry-apr", type=float, default=0.20)
    p.add_argument("--exit-apr", type=float, default=0.05)
    p.add_argument("--min-hold", type=int, default=48, help="hours")
    p.add_argument("--max-hold", type=int, default=24 * 21, help="hours")
    p.add_argument("--halflife", type=float, default=36.0, help="hours")

    p.add_argument("--no-reverse", action="store_true",
                   help="forbid shorting leg a (required when leg a is spot)")
    p.add_argument("--slippage-bps", type=float, default=2.0,
                   help="per order, on top of the venue taker fee")
    p.add_argument("--sweep-entry", default=None,
                   help="comma-separated entry thresholds to compare, e.g. 0.05,0.1,0.2")
    p.add_argument("--json", default=None, help="write full results to this path")
    p.add_argument("--no-cache", action="store_true", help="bypass the on-disk HTTP cache")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def run_one(coin: str, args, entry_apr: Optional[float] = None):
    venue_a, venue_b = get_venue(args.venue_a), get_venue(args.venue_b)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - args.days * DAY_MS

    pair = load_pair(venue_a, venue_b, coin, start_ms, end_ms)
    costs = from_venues(venue_a, venue_b, slippage_bps=args.slippage_bps)
    params = StrategyParams(
        halflife_hours=args.halflife,
        entry_apr=entry_apr if entry_apr is not None else args.entry_apr,
        exit_apr=min(args.exit_apr, (entry_apr or args.entry_apr) * 0.5),
        min_hold_hours=args.min_hold,
        max_hold_hours=args.max_hold,
        allow_reverse=not args.no_reverse,
    )
    # A spot leg is bought outright: full notional, no liquidation.
    config = BacktestConfig(
        notional=args.notional,
        leverage=args.leverage,
        leverage_a=1.0 if venue_a.is_spot else None,
        leverage_b=1.0 if venue_b.is_spot else None,
    )

    result = run_backtest(pair, costs, params, config)
    return result, summarize(result)


def _as_dict(result: BacktestResult, m: Metrics) -> dict:
    return {
        "coin": result.coin,
        "venue_a": result.venue_a,
        "venue_b": result.venue_b,
        "span_days": round(result.span_hours / 24, 1),
        "params": vars(result.params),
        "costs": {
            "round_trip_bps": result.costs.round_trip_bps,
            "entry_bps": result.costs.entry_bps,
        },
        "metrics": {k: (None if v == float("inf") else v) for k, v in vars(m).items()},
        "trades": [
            {
                "entry_ms": t.entry_ms, "exit_ms": t.exit_ms,
                "long": t.long_venue, "short": t.short_venue,
                "hold_hours": round(t.holding_hours, 1),
                "funding_pnl": round(t.funding_pnl, 2),
                "basis_pnl": round(t.basis_pnl, 2),
                "fees": round(t.fees, 2),
                "net_pnl": round(t.net_pnl, 2),
                "realized_apr": round(t.realized_apr, 4),
                "exit_reason": t.exit_reason,
                "liquidated": t.liquidated,
            }
            for t in result.trades
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.no_cache:
        import shutil
        from .venues import CACHE_DIR
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    payload = []

    if args.sweep_entry:
        thresholds = [float(x) for x in args.sweep_entry.split(",")]
        print(f"\nENTRY THRESHOLD SWEEP  ({args.venue_a} <-> {args.venue_b}, "
              f"{args.days}d, {args.min_hold}h min hold)\n")
        header = f"{'coin':<6}{'entry APR':>11}{'trades':>8}{'win%':>8}{'CI low':>9}" \
                 f"{'expect$':>10}{'APR':>9}{'maxDD%':>9}"
        print(header)
        print("-" * len(header))
        for coin in args.coins:
            for threshold in thresholds:
                try:
                    result, m = run_one(coin, args, entry_apr=threshold)
                except Exception as exc:  # noqa: BLE001 - surface venue issues, keep going
                    print(f"{coin:<6}{threshold:>11.0%}  FAILED: {exc}")
                    continue
                print(f"{coin:<6}{threshold:>11.0%}{m.n_trades:>8}"
                      f"{m.win_rate * 100:>7.1f}%{m.win_rate_lo95 * 100:>8.1f}%"
                      f"{m.expectancy:>10.2f}{m.apr * 100:>8.1f}%"
                      f"{m.max_drawdown_pct * 100:>8.2f}%")
                payload.append(_as_dict(result, m))
        print()
    else:
        for coin in args.coins:
            try:
                result, m = run_one(coin, args)
            except Exception as exc:  # noqa: BLE001
                print(f"{coin}: FAILED - {exc}", file=sys.stderr)
                continue
            print(format_report(result, m))
            payload.append(_as_dict(result, m))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
