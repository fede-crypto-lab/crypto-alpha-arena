"""Signal generation for cross-venue funding carry.

The strategy has exactly one belief: **funding spreads are persistent**. A venue
whose perp has been paying 30% annualised more than another venue's for the last
few days is more likely than not to still be doing so tomorrow, because the
imbalance is driven by slow things - who holds the leveraged longs, which venue
retail uses, where the basis desks are. It is not a prediction of price.

So the forecast is the dumbest one that respects that belief: an exponentially
weighted mean of the spread that has *already settled*. No regression, no ML. A
more elaborate forecast on ~1500 hourly observations overfits, and the honest
version makes the backtest's verdict interpretable.

Every value returned here at grid time t is computed from data with timestamp <= t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .dataset import Pair


@dataclass
class StrategyParams:
    """Entry/exit rules, all thresholds expressed as annualised spread."""

    #: Halflife of the EWMA forecast, in hours. Short enough to react to a regime
    #: change, long enough not to trade on a single 8h print.
    halflife_hours: float = 36.0
    #: Minimum forecast spread to open. Must comfortably exceed
    #: `CostModel.breakeven_spread_apr(expected_hold)` or the strategy pays the
    #: exchange to take a view.
    entry_apr: float = 0.20
    #: Close once the forecast decays below this. Hysteresis (exit < entry) stops
    #: the position flickering around a single threshold.
    exit_apr: float = 0.05
    #: Never close before this, even if the forecast decays - a round trip has to
    #: be amortised over enough settlements to pay for itself.
    min_hold_hours: int = 48
    #: Hard cap: stale carry is a liquidation risk with no compensating edge.
    max_hold_hours: int = 24 * 21
    #: Hours to stand down after a close, to avoid re-entering the same noise.
    cooldown_hours: int = 8
    #: Warmup before any trading, so the EWMA is not seeded off two prints.
    warmup_hours: int = 72
    #: Whether the mirror trade (short leg a / long leg b) is allowed. Set False
    #: when leg a is spot: "short venue a" would mean borrowing the coin, which
    #: carries a borrow rate this model does not price.
    allow_reverse: bool = True

    def __post_init__(self) -> None:
        if self.exit_apr >= self.entry_apr:
            raise ValueError("exit_apr must be below entry_apr (hysteresis)")
        if self.halflife_hours <= 0:
            raise ValueError("halflife_hours must be positive")
        if self.min_hold_hours > self.max_hold_hours:
            raise ValueError("min_hold_hours cannot exceed max_hold_hours")


class SpreadForecast:
    """Causal EWMA of the annualised funding spread, precomputed over the grid."""

    def __init__(self, pair: Pair, params: StrategyParams):
        self.params = params
        alpha = 1.0 - 0.5 ** (1.0 / params.halflife_hours)
        self._value: Dict[int, float] = {}
        self._raw: Dict[int, float] = {}

        ewma = None
        for t in pair.grid:
            raw = pair.spread_apr_at(t)
            ewma = raw if ewma is None else ewma + alpha * (raw - ewma)
            self._raw[t] = raw
            self._value[t] = ewma

    def at(self, time_ms: int) -> float:
        return self._value[time_ms]

    def raw_at(self, time_ms: int) -> float:
        return self._raw[time_ms]

    def values(self) -> List[float]:
        return [self._value[t] for t in sorted(self._value)]


def desired_direction(forecast_apr: float, params: StrategyParams) -> int:
    """+1 = short venue b / long venue a. -1 = the mirror. 0 = stay flat.

    Sign convention follows `Pair.spread_apr_at`, which is b minus a: when it is
    positive, b is the expensive leg, so b is the one to be short.
    """
    if forecast_apr >= params.entry_apr:
        return 1
    if forecast_apr <= -params.entry_apr and params.allow_reverse:
        return -1
    return 0


def should_close(forecast_apr: float, direction: int, held_hours: float,
                 params: StrategyParams) -> str:
    """Return a non-empty exit reason, or "" to stay in the position."""
    if held_hours >= params.max_hold_hours:
        return "max_hold"
    if held_hours < params.min_hold_hours:
        return ""
    signed = forecast_apr * direction  # carry we are actually earning
    if signed <= 0:
        return "spread_flip"
    if signed < params.exit_apr:
        return "spread_decay"
    return ""
