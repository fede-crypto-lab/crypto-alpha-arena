"""Cross-sectional funding carry: rank the universe, hold the richest N.

The single-pair engine in `backtest.py` tries to time *one* coin's funding, and
the measured result was that it cannot: the EWMA's entry selectivity came out
below 1.0, so every trade it took was worse than doing nothing.

This module tests the other axis. Instead of asking "is this coin's funding high
*now* versus its own history", it asks "which coins are paying the most *right
now* versus each other". That is a different bet, and a better-posed one: funding
dispersion across coins is enormous (a median near the venue floor against a 90th
percentile an order of magnitude higher), because leveraged retail crowds into
whatever is moving rather than spreading evenly. The premium is concentrated, and
concentration is something a ranking can find even when a time series cannot be
forecast.

Each position stays individually delta-neutral - long spot, short perp, same coin,
same notional - so the portfolio has no net direction. What varies is only *which*
carries are held.

Turnover is the enemy. Every rotation pays a full round trip, so entry and exit
ranks are deliberately far apart: a coin has to fall well out of the top before its
slot is given up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .backtest import BacktestConfig, Trade, _OpenPosition, _bucket_funding, _close
from .costs import CostModel, as_cost_book
from .dataset import HOUR_MS, Pair

logger = logging.getLogger(__name__)


@dataclass
class PortfolioParams:
    """Selection rules for the cross-sectional carry."""

    #: Capital is sized for this many concurrent carries, whether or not they are
    #: all filled - idle slots correctly drag the portfolio APR down.
    max_positions: int = 5
    #: Trailing window used to rank coins, in hours. Ranking on realised funding
    #: keeps the rule strictly causal.
    rank_lookback_hours: int = 168
    #: Open a coin only if it ranks this high or better.
    entry_rank: int = 5
    #: Close only once it falls below this rank. The gap between the two is the
    #: hysteresis that stops the book churning on rank noise.
    exit_rank: int = 12
    #: Absolute floor, on top of the rank: being the best of a bad universe is
    #: not a reason to pay 38bp.
    min_entry_apr: float = 0.15
    min_hold_hours: int = 72
    max_hold_hours: int = 24 * 90
    #: How often the ranking is recomputed. Every hour would be free in the
    #: backtest and ruinous live.
    decision_every_hours: int = 8
    warmup_hours: int = 168
    #: Fraction of the universe, by trailing realised volatility, to refuse at
    #: entry. 0.0 keeps everything; 0.2 drops the most volatile fifth.
    #:
    #: This is a RISK gate, not a timing signal, and the distinction is the whole
    #: point. Price indicators do not forecast funding - over 540 days, MACD
    #: scores a rank IC of 0.025 against next week's funding, momentum -0.051 and
    #: price-vs-SMA200 -0.067, while the trailing funding rate itself scores
    #: 0.498. There is nothing for a trend filter to add to the signal.
    #:
    #: Volatility is different, because it predicts the failure modes rather than
    #: the return. Split the universe into volatility quintiles and the chance of
    #: an adverse run past +99.5% over a 20-day hold - which liquidates the short
    #: leg at 1x - runs 0.00%, 0.34%, 0.68%, 3.04%, 5.76% from calmest to most
    #: volatile. The gate is cross-sectional rather than an absolute threshold so
    #: it keeps meaning when the whole market's volatility shifts.
    exclude_vol_quantile: float = 0.0
    #: Trailing window for that volatility, in hours.
    vol_lookback_hours: int = 168

    def __post_init__(self) -> None:
        if not 0.0 <= self.exclude_vol_quantile < 1.0:
            raise ValueError("exclude_vol_quantile must be in [0, 1)")
        if self.exit_rank < self.entry_rank:
            raise ValueError("exit_rank must be >= entry_rank (hysteresis)")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")


@dataclass
class PortfolioResult:
    config: BacktestConfig
    costs: CostModel
    params: PortfolioParams
    trades: List[Trade] = field(default_factory=list)
    equity: List[Tuple[int, float]] = field(default_factory=list)
    #: Present so `metrics.summarize` can consume this result unchanged; the
    #: per-coin spread series is not meaningful for a whole book.
    spread: List[Tuple[int, float, float]] = field(default_factory=list)
    liquidations: int = 0
    #: Fraction of available slot-hours that actually held a position.
    utilisation: float = 0.0
    coins_traded: List[str] = field(default_factory=list)

    @property
    def capital(self) -> float:
        """Capital for the whole book: one slot's margin times the slot count."""
        return self.config.capital * self.params.max_positions

    @property
    def span_hours(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        return (self.equity[-1][0] - self.equity[0][0]) / HOUR_MS


def trailing_apr(pair: Pair, time_ms: int, window_hours: int) -> Optional[float]:
    """Mean annualised funding actually settled in (t - window, t].

    Realised settlements only, so the ranking cannot see a rate before the venue
    charged it. Returns None when the coin has no history in the window, which
    keeps a newly listed perp out of the ranking instead of scoring it as zero.
    """
    start = time_ms - window_hours * HOUR_MS
    points = pair.b.funding_between(start, time_ms)
    if not points:
        return None
    from .dataset import annualize
    return sum(annualize(p.rate, pair.b.interval_hours) for p in points) / len(points)


def run_portfolio(
    universe: Dict[str, Pair],
    costs: CostModel,
    params: PortfolioParams,
    config: BacktestConfig,
) -> PortfolioResult:
    """Replay the whole universe, holding up to `max_positions` carries at once."""
    if not universe:
        raise ValueError("empty universe")

    book = as_cost_book(costs)
    result = PortfolioResult(config=config, costs=book.default, params=params)
    funding_buckets = {
        coin: (_bucket_funding(pair.a), _bucket_funding(pair.b))
        for coin, pair in universe.items()
    }
    bounds = {coin: (pair.grid[0], pair.grid[-1]) for coin, pair in universe.items()}

    grid_start = min(lo for lo, _ in bounds.values())
    grid_end = max(hi for _, hi in bounds.values())
    grid = list(range(grid_start, grid_end + 1, HOUR_MS))

    q = config.notional
    realized = 0.0
    positions: Dict[str, _OpenPosition] = {}
    slot_hours = filled_hours = 0
    warmup_until = grid_start + params.warmup_hours * HOUR_MS

    for t in grid:
        # 1. Accrue funding on every open carry.
        for coin, pos in positions.items():
            if pos.entry_ms < t:
                fa, fb = funding_buckets[coin]
                rate_a, rate_b = fa.get(t, 0.0), fb.get(t, 0.0)
                long_rate = rate_a if pos.direction == 1 else rate_b
                short_rate = rate_b if pos.direction == 1 else rate_a
                pos.accrued_funding += q * (short_rate - long_rate)

        # 2. Mark the book and force-close anything that breached its margin.
        unrealized = 0.0
        for coin in list(positions):
            pos = positions[coin]
            pair = universe[coin]
            legs = _leg_pnl(pair, pos, t, q)
            if legs is None:
                continue
            long_leg_pnl, short_leg_pnl = legs
            unrealized += long_leg_pnl + short_leg_pnl + pos.accrued_funding

            lev_long = config.leverage_a if pos.direction == 1 else config.leverage_b
            lev_short = config.leverage_b if pos.direction == 1 else config.leverage_a
            maint = config.maintenance_margin_frac * q
            if (long_leg_pnl <= -(q / lev_long - maint)
                    or short_leg_pnl <= -(q / lev_short - maint)):
                cc = book.for_coin(coin)
                fees = (cc.entry_cost(q) + cc.exit_cost(q)
                        + q * config.liquidation_penalty_bps / 10_000.0)
                trade = _close(pair, pos, t, q, long_leg_pnl + short_leg_pnl,
                               fees, "liquidation", liquidated=True)
                result.trades.append(trade)
                result.liquidations += 1
                realized += trade.net_pnl
                unrealized -= long_leg_pnl + short_leg_pnl + pos.accrued_funding
                del positions[coin]
                logger.warning("%s: leg liquidated at %s", coin, t)

        # 3. Rebalance, on the decision clock only.
        is_decision = ((t - grid_start) // HOUR_MS) % params.decision_every_hours == 0
        if is_decision and t >= warmup_until:
            ranked = _rank(universe, bounds, t, params)
            rank_of = {coin: i + 1 for i, (coin, _) in enumerate(ranked)}

            for coin in list(positions):
                pos = positions[coin]
                held = (t - pos.entry_ms) / HOUR_MS
                rank = rank_of.get(coin, 10**6)
                reason = ""
                if held >= params.max_hold_hours:
                    reason = "max_hold"
                elif held >= params.min_hold_hours and rank > params.exit_rank:
                    reason = "rank_decay"
                if not reason:
                    continue
                legs = _leg_pnl(universe[coin], pos, t, q)
                if legs is None:
                    continue
                basis = legs[0] + legs[1]
                cc = book.for_coin(coin)
                fees = cc.entry_cost(q) + cc.exit_cost(q)
                trade = _close(universe[coin], pos, t, q, basis, fees, reason)
                result.trades.append(trade)
                realized += trade.net_pnl
                unrealized -= basis + pos.accrued_funding
                del positions[coin]

            for coin, apr in ranked[:params.entry_rank]:
                if len(positions) >= params.max_positions:
                    break
                if coin in positions or apr < params.min_entry_apr:
                    continue
                pair = universe[coin]
                marks = _entry_marks(pair, t, direction=1)
                if marks is None:
                    continue
                positions[coin] = _OpenPosition(
                    direction=1,
                    entry_ms=t,
                    entry_mark_long=marks[0],
                    entry_mark_short=marks[1],
                    forecast_at_entry=apr,
                )
                if coin not in result.coins_traded:
                    result.coins_traded.append(coin)

        slot_hours += params.max_positions
        filled_hours += len(positions)
        result.equity.append((t, result.capital + realized + unrealized))

    # Close the book at the end of the sample so nothing escapes the statistics.
    t = grid[-1]
    for coin, pos in positions.items():
        legs = _leg_pnl(universe[coin], pos, t, q)
        if legs is None:
            continue
        cc = book.for_coin(coin)
        fees = cc.entry_cost(q) + cc.exit_cost(q)
        result.trades.append(
            _close(universe[coin], pos, t, q, legs[0] + legs[1], fees, "end_of_sample")
        )

    result.utilisation = filled_hours / slot_hours if slot_hours else 0.0
    return result


def _rank(universe, bounds, t, params) -> List[Tuple[str, float]]:
    """Coins ordered by trailing realised funding, richest first.

    The volatility gate is applied here rather than at exit: a position already
    open is left alone, because closing it costs a full round trip and the move
    the gate is worried about may already have happened.
    """
    scored = []
    vols = {}
    for coin, pair in universe.items():
        lo, hi = bounds[coin]
        if not lo <= t <= hi:
            continue
        apr = trailing_apr(pair, t, params.rank_lookback_hours)
        if apr is None:
            continue
        scored.append((coin, apr))
        if params.exclude_vol_quantile > 0:
            vol = pair.b.realized_volatility(t, params.vol_lookback_hours)
            if vol is not None:
                vols[coin] = vol

    if params.exclude_vol_quantile > 0 and len(vols) >= 5:
        ordered = sorted(vols, key=lambda c: vols[c])
        keep = max(1, int(round(len(ordered) * (1 - params.exclude_vol_quantile))))
        allowed = set(ordered[:keep])
        # A coin with no volatility reading is dropped rather than waved through:
        # an unmeasurable risk is not the same as an absent one.
        scored = [(c, apr) for c, apr in scored if c in allowed]

    scored.sort(key=lambda kv: -kv[1])
    return scored


def _leg_pnl(pair: Pair, pos: _OpenPosition, t: int, q: float):
    long_series = pair.a if pos.direction == 1 else pair.b
    short_series = pair.b if pos.direction == 1 else pair.a
    try:
        long_ret = long_series.mark_at(t) / pos.entry_mark_long - 1.0
        short_ret = short_series.mark_at(t) / pos.entry_mark_short - 1.0
    except KeyError:
        return None
    return q * long_ret, -q * short_ret


def _entry_marks(pair: Pair, t: int, direction: int):
    long_series = pair.a if direction == 1 else pair.b
    short_series = pair.b if direction == 1 else pair.a
    try:
        return long_series.mark_at(t), short_series.mark_at(t)
    except KeyError:
        return None
