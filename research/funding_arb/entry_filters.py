#!/usr/bin/env python3
"""Do technical entry filters improve the seasonal calendar-spread picks?

Plan and pass criterion are in MEMORY.md §11-septies, written before any of
this was run. In short: the picks are the walk-forward best picks of the
calendar-spread scan; features are measured on the spread itself using only
prices from before the entry day; rules are chosen on test years 2018-2022 and
checked ONCE on 2023-2026 with the thresholds frozen.

The two modes keep that order honest - the holdout years are not printed until
a rule is named on the command line:

    python -m research.funding_arb.entry_filters research/funding_arb/data/databento
    python -m research.funding_arb.entry_filters research/funding_arb/data/databento --holdout "vol_regime:low"
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .futures_contracts import read_dir, round_trip_cost
from .seasonal_scan import (
    EXPIRY_BUFFER_DAYS, Cycle, Pick, SpreadDef, _Lookup, calendar_defs, scan, window_pnl, windows,
)

DEV_LAST_YEAR = 2022
FEATURES = ("mom20", "ma_dev", "vol_regime", "level_z", "seasonal_t")
#: Fewer prior prints than this and the 20-day statistics are noise.
MIN_HISTORY = 25
CONFIRM_DAYS = 10


@dataclass
class Trade:
    spread: str
    year: int
    window: Tuple[int, int, int]
    net: float                              # the unfiltered best pick's result
    features: Dict[str, Optional[float]] = field(default_factory=dict)
    confirm_net: Optional[float] = None     # result with momentum-confirmed entry; None = no entry


def _sd(xs: Sequence[float]) -> Optional[float]:
    return statistics.stdev(xs) if len(xs) > 2 and statistics.stdev(xs) > 0 else None


def features_at(values: Dict[date, float], entry: date, direction: int) -> Dict[str, Optional[float]]:
    """Statistics of the spread from prints strictly before `entry`, signed so
    that positive means 'in the trade's favour'."""
    hist = [values[d] for d in sorted(values) if d < entry]
    out: Dict[str, Optional[float]] = {"mom20": None, "ma_dev": None, "vol_regime": None}
    if len(hist) < MIN_HISTORY:
        return out
    diffs = [b - a for a, b in zip(hist[:-1], hist[1:])]
    sd20 = _sd(diffs[-20:])
    if sd20 is None:
        return out
    out["mom20"] = direction * (hist[-1] - hist[-21]) / (sd20 * math.sqrt(20))
    out["ma_dev"] = direction * (hist[-1] - statistics.mean(hist[-20:])) / sd20
    sd_long = _sd(diffs[-120:])
    out["vol_regime"] = sd20 / sd_long if sd_long else None
    return out


def level_z(cycles: List[Cycle], t: int, lookback: int, offset: int, now: float,
            direction: int) -> Optional[float]:
    """Where the spread stands today against the same point of earlier cycles."""
    past = []
    for c in cycles[t - lookback:t]:
        hit = _Lookup(c.values).at_or_after(c.anchor - timedelta(days=offset))
        if hit:
            past.append(hit[1])
    sd = _sd(past)
    return direction * (now - statistics.mean(past)) / sd if sd else None


def seasonal_t(cycles: List[Cycle], t: int, lookback: int, window, cost: float) -> Optional[float]:
    prior = [p - cost for c in cycles[t - lookback:t]
             if (p := window_pnl(c, _Lookup(c.values), window)) is not None]
    sd = _sd(prior)
    return statistics.mean(prior) / (sd / math.sqrt(len(prior))) if sd else None


def confirmed_entry(cycle: Cycle, window, cost: float) -> Optional[float]:
    """Enter on the first day within CONFIRM_DAYS of the seasonal entry date on
    which the 5-print momentum points the trade's way; exit on the original
    exit date. No confirmation in time means no trade."""
    offset, hold, direction = window
    days = sorted(cycle.values)
    lk = _Lookup(cycle.values)
    start = lk.at_or_after(cycle.anchor - timedelta(days=offset))
    if start is None:
        return None
    target_exit = start[0] + timedelta(days=hold)
    for i, d in enumerate(days):
        if d < start[0] or i < 5:
            continue
        if (d - start[0]).days > CONFIRM_DAYS:
            return None
        if direction * (cycle.values[d] - cycle.values[days[i - 5]]) > 0:
            exit_ = lk.at_or_after(target_exit)
            if exit_ is None or exit_[0] <= d or \
                    exit_[0] > cycle.anchor - timedelta(days=EXPIRY_BUFFER_DAYS):
                return None
            return direction * (exit_[1] - cycle.values[d]) - cost
    return None


def collect(data, lookback: int = 8, min_wins: int = 7) -> List[Trade]:
    trades: List[Trade] = []

    def on_pick(sd: SpreadDef, cycles: List[Cycle], t: int, pick: Pick) -> None:
        cycle, (offset, _, direction) = cycles[t], pick.window
        cost = round_trip_cost(sd.roots)
        entry = _Lookup(cycle.values).at_or_after(cycle.anchor - timedelta(days=offset))
        if entry is None:
            return
        f = features_at(cycle.values, entry[0], direction)
        prev = [cycle.values[d] for d in sorted(cycle.values) if d < entry[0]]
        f["level_z"] = level_z(cycles, t, lookback, offset, prev[-1], direction) if prev else None
        f["seasonal_t"] = seasonal_t(cycles, t, lookback, pick.window, cost)
        trades.append(Trade(pick.spread, pick.test_year, pick.window, pick.oos_net, f,
                            confirmed_entry(cycle, pick.window, cost)))

    roots = sorted({k[0] for k in data})
    scan(calendar_defs(roots), data, lookback, min_wins, windows(), on_pick=on_pick)
    return trades


def _stats(xs: Sequence[float]) -> str:
    if not xs:
        return "n=0"
    return (f"n={len(xs):5d} win {sum(x > 0 for x in xs) / len(xs):5.1%} "
            f"mean net {statistics.mean(xs):6.0f} $ median {statistics.median(xs):5.0f} $")


def terciles(trades: Sequence[Trade], feature: str) -> Tuple[float, float]:
    xs = sorted(tr.features[feature] for tr in trades if tr.features.get(feature) is not None)
    return xs[len(xs) // 3], xs[2 * len(xs) // 3]


def bucket(value: Optional[float], cuts: Tuple[float, float]) -> Optional[str]:
    if value is None:
        return None
    return "low" if value < cuts[0] else ("mid" if value < cuts[1] else "high")


def development_report(trades: Sequence[Trade]) -> str:
    dev = [tr for tr in trades if tr.year <= DEV_LAST_YEAR]
    lines = [f"DEVELOPMENT years <= {DEV_LAST_YEAR} only (holdout not shown)",
             f"  unfiltered best picks      {_stats([t.net for t in dev])}"]
    for feat in FEATURES:
        cuts = terciles(dev, feat)
        lines.append(f"  {feat}  (tercile cuts {cuts[0]:.2f} / {cuts[1]:.2f})")
        for b in ("low", "mid", "high"):
            xs = [t.net for t in dev if bucket(t.features.get(feat), cuts) == b]
            lines.append(f"     {b:5s} {_stats(xs)}")
    taken = [t.confirm_net for t in dev if t.confirm_net is not None]
    lines.append(f"  confirm entry (5d momentum)  {_stats(taken)}  "
                 f"({len(taken)}/{len(dev)} entries taken)")
    same = [t.net for t in dev if t.confirm_net is not None]
    lines.append(f"     same trades, seasonal entry {_stats(same)}")
    return "\n".join(lines)


def apply_rule(trades: Sequence[Trade], rule: str, cuts: Dict[str, Tuple[float, float]]) -> List[float]:
    """rule: 'feature:bucket' terms joined by '+', or 'confirm'."""
    if rule == "confirm":
        return [t.confirm_net for t in trades if t.confirm_net is not None]
    out = []
    for t in trades:
        ok = True
        for term in rule.split("+"):
            feat, want = term.split(":")
            got = bucket(t.features.get(feat), cuts[feat])
            ok &= got in want.split("|")
        if ok:
            out.append(t.net)
    return out


def holdout_report(trades: Sequence[Trade], rule: str) -> str:
    dev = [t for t in trades if t.year <= DEV_LAST_YEAR]
    hold = [t for t in trades if t.year > DEV_LAST_YEAR]
    cuts = {f: terciles(dev, f) for f in FEATURES}   # frozen on development years
    lines = [f"HOLDOUT years > {DEV_LAST_YEAR}, rule '{rule}' (thresholds frozen on development)",
             f"  development: unfiltered {_stats([t.net for t in dev])}",
             f"               filtered   {_stats(apply_rule(dev, rule, cuts))}",
             f"  holdout:     unfiltered {_stats([t.net for t in hold])}",
             f"               filtered   {_stats(apply_rule(hold, rule, cuts))}"]
    for y in sorted({t.year for t in hold}):
        yt = [t for t in hold if t.year == y]
        lines.append(f"    {y}: unfiltered {_stats([t.net for t in yt])} | filtered {_stats(apply_rule(yt, rule, cuts))}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("directory")
    ap.add_argument("--holdout", metavar="RULE", help="evaluate this frozen rule on the holdout years, once")
    a = ap.parse_args(argv)
    trades = collect(read_dir(a.directory))
    print(holdout_report(trades, a.holdout) if a.holdout else development_report(trades))
    return 0


if __name__ == "__main__":
    sys.exit(main())
