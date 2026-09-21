"""Event-driven backtest for a delta-neutral cross-venue funding carry.

The position is always a pair: long the cheap-funding venue, short the expensive
one, same coin, same notional. What is being harvested is the *difference* in
funding, not direction.

Three things this engine refuses to pretend:

* **Funding is replayed as discrete events**, at the timestamps the venues
  actually settled them, not as a resampled hourly average. A carry held across
  three OKX settlements is paid three times, no more.
* **The hedge is imperfect.** The two legs are marked on their own venues' prices,
  so the residual (basis PnL) lands in the results as its own line item. If a
  "market-neutral" strategy's P&L is mostly basis, it is not market-neutral.
* **Margin is siloed per venue.** The losing leg can be liquidated while the
  winning leg sits in profit on the other exchange. This is the way real funding
  carry books actually blow up, so it is modelled explicitly rather than assumed
  away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .costs import CostModel
from .dataset import HOUR_MS, Pair, VenueSeries
from .strategy import SpreadForecast, StrategyParams, desired_direction, should_close

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    #: Notional per leg, in USD. Gross exposure is twice this.
    notional: float = 10_000.0
    #: Default isolated leverage, applied to whichever leg does not override it.
    leverage: float = 3.0
    #: Per-leg leverage overrides. A spot leg must be 1.0: the coin is bought
    #: outright, so it ties up full notional and cannot be force-closed. Applying
    #: perp leverage to it both understates the capital required and invents
    #: liquidations that could not happen.
    leverage_a: Optional[float] = None
    leverage_b: Optional[float] = None
    #: Venue maintenance margin, as a fraction of notional.
    maintenance_margin_frac: float = 0.005
    #: Extra cost incurred when a leg is force-closed.
    liquidation_penalty_bps: float = 50.0

    def __post_init__(self) -> None:
        if self.leverage_a is None:
            self.leverage_a = self.leverage
        if self.leverage_b is None:
            self.leverage_b = self.leverage
        if min(self.leverage_a, self.leverage_b) <= 0:
            raise ValueError("leverage must be positive")

    @property
    def capital(self) -> float:
        """Cash that has to sit on the two venues to run one leg-pair.

        Each leg posts its own margin - there is no netting across exchanges,
        which is precisely why the losing leg can liquidate on its own.
        """
        return self.notional / self.leverage_a + self.notional / self.leverage_b


@dataclass
class Trade:
    coin: str
    direction: int          # +1 = short venue b / long venue a
    long_venue: str
    short_venue: str
    entry_ms: int
    exit_ms: int
    notional: float
    funding_pnl: float      # net funding collected over the hold
    basis_pnl: float        # residual from the two legs' prices diverging
    fees: float             # entry + exit costs, plus any liquidation penalty
    exit_reason: str
    forecast_at_entry: float
    liquidated: bool = False

    @property
    def net_pnl(self) -> float:
        return self.funding_pnl + self.basis_pnl - self.fees

    @property
    def holding_hours(self) -> float:
        return (self.exit_ms - self.entry_ms) / HOUR_MS

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    @property
    def realized_apr(self) -> float:
        """Return on the capital this one trade tied up, annualised."""
        hours = self.holding_hours
        if hours <= 0:
            return 0.0
        return self.net_pnl / self.notional * (24 * 365 / hours)


@dataclass
class _OpenPosition:
    direction: int
    entry_ms: int
    entry_mark_long: float
    entry_mark_short: float
    forecast_at_entry: float
    accrued_funding: float = 0.0


@dataclass
class BacktestResult:
    coin: str
    venue_a: str
    venue_b: str
    config: BacktestConfig
    costs: CostModel
    params: StrategyParams
    trades: List[Trade] = field(default_factory=list)
    #: (time_ms, equity) sampled hourly, mark-to-market.
    equity: List[Tuple[int, float]] = field(default_factory=list)
    #: (time_ms, raw annualised spread, smoothed forecast) for inspection/plots.
    spread: List[Tuple[int, float, float]] = field(default_factory=list)
    liquidations: int = 0

    @property
    def capital(self) -> float:
        return self.config.capital

    @property
    def span_hours(self) -> float:
        if len(self.equity) < 2:
            return 0.0
        return (self.equity[-1][0] - self.equity[0][0]) / HOUR_MS


def _bucket_funding(series: VenueSeries) -> Dict[int, float]:
    """Sum each venue's settlements into the hour they landed in.

    Venue timestamps sit a few milliseconds either side of the hour, so they are
    snapped to the nearest hour rather than floored - flooring a 12:00:00.032
    print into the 12:00 bucket works, but flooring 11:59:59.900 would push a
    settlement an hour late.
    """
    out: Dict[int, float] = {}
    for point in series.funding:
        hour = int(round(point.time_ms / HOUR_MS)) * HOUR_MS
        out[hour] = out.get(hour, 0.0) + point.rate
    return out


def run_backtest(
    pair: Pair,
    costs: CostModel,
    params: StrategyParams,
    config: BacktestConfig,
) -> BacktestResult:
    """Replay the pair hour by hour and return trades plus the equity curve."""
    forecast = SpreadForecast(pair, params)
    funding_a = _bucket_funding(pair.a)
    funding_b = _bucket_funding(pair.b)

    result = BacktestResult(
        coin=pair.coin, venue_a=pair.a.name, venue_b=pair.b.name,
        config=config, costs=costs, params=params,
    )

    q = config.notional
    realized = 0.0
    position: Optional[_OpenPosition] = None
    cooldown_until = 0
    warmup_until = pair.grid[0] + params.warmup_hours * HOUR_MS

    for t in pair.grid:
        fc = forecast.at(t)
        result.spread.append((t, forecast.raw_at(t), fc))

        # 1. Credit funding settled this hour, but only for a position that was
        #    already open before it - entering *on* a settlement collects nothing.
        if position is not None and position.entry_ms < t:
            rate_a, rate_b = funding_a.get(t, 0.0), funding_b.get(t, 0.0)
            long_rate = rate_a if position.direction == 1 else rate_b
            short_rate = rate_b if position.direction == 1 else rate_a
            # The long leg pays funding when the rate is positive; the short receives.
            position.accrued_funding += q * (short_rate - long_rate)

        # 2. Mark both legs on their own venue's price.
        unrealized = 0.0
        if position is not None:
            long_series = pair.a if position.direction == 1 else pair.b
            short_series = pair.b if position.direction == 1 else pair.a
            long_ret = long_series.mark_at(t) / position.entry_mark_long - 1.0
            short_ret = short_series.mark_at(t) / position.entry_mark_short - 1.0
            long_leg_pnl = q * long_ret
            short_leg_pnl = -q * short_ret
            unrealized = long_leg_pnl + short_leg_pnl + position.accrued_funding

            # 3. Isolated-margin liquidation check, per leg, on that leg's own
            #    venue leverage - a spot leg at 1x effectively cannot be hit.
            lev_long = config.leverage_a if position.direction == 1 else config.leverage_b
            lev_short = config.leverage_b if position.direction == 1 else config.leverage_a
            maint = config.maintenance_margin_frac * q
            hit = (
                long_leg_pnl <= -(q / lev_long - maint)
                or short_leg_pnl <= -(q / lev_short - maint)
            )
            held_hours = (t - position.entry_ms) / HOUR_MS

            if hit:
                # Both legs are closed together here, which is the *benign*
                # reading of a liquidation: the hedge dies but nothing is left
                # dangling. Reality is worse - the exchange closes the losing leg
                # and hands you a naked position on the other venue, directional
                # until something notices. Treat the liquidation count, not the
                # P&L of a liquidated trade, as the signal to act on.
                fees = (
                    costs.entry_cost(q) + costs.exit_cost(q)
                    + q * config.liquidation_penalty_bps / 10_000.0
                )
                trade = _close(pair, position, t, q, long_leg_pnl + short_leg_pnl,
                               fees, "liquidation", liquidated=True)
                result.trades.append(trade)
                result.liquidations += 1
                realized += trade.net_pnl
                position = None
                cooldown_until = t + params.cooldown_hours * HOUR_MS
                unrealized = 0.0
                logger.warning("%s: leg liquidated at %s after %.0fh",
                               pair.coin, t, held_hours)
            else:
                # 4. Normal exit.
                reason = should_close(fc, position.direction, held_hours, params)
                if reason:
                    fees = costs.entry_cost(q) + costs.exit_cost(q)
                    trade = _close(pair, position, t, q, long_leg_pnl + short_leg_pnl,
                                   fees, reason)
                    result.trades.append(trade)
                    realized += trade.net_pnl
                    position = None
                    cooldown_until = t + params.cooldown_hours * HOUR_MS
                    unrealized = 0.0

        # 5. Entry.
        if position is None and t >= warmup_until and t >= cooldown_until:
            direction = desired_direction(fc, params)
            if direction != 0:
                long_series = pair.a if direction == 1 else pair.b
                short_series = pair.b if direction == 1 else pair.a
                position = _OpenPosition(
                    direction=direction,
                    entry_ms=t,
                    entry_mark_long=long_series.mark_at(t),
                    entry_mark_short=short_series.mark_at(t),
                    forecast_at_entry=fc,
                )

        result.equity.append((t, config.capital + realized + unrealized))

    # Close anything still open at the end of the sample, so the final trade is
    # not silently excluded from the win rate.
    if position is not None:
        t = pair.grid[-1]
        long_series = pair.a if position.direction == 1 else pair.b
        short_series = pair.b if position.direction == 1 else pair.a
        leg_pnl = (
            q * (long_series.mark_at(t) / position.entry_mark_long - 1.0)
            - q * (short_series.mark_at(t) / position.entry_mark_short - 1.0)
        )
        fees = costs.entry_cost(q) + costs.exit_cost(q)
        result.trades.append(_close(pair, position, t, q, leg_pnl, fees, "end_of_sample"))

    return result


def _close(pair: Pair, position: _OpenPosition, exit_ms: int, notional: float,
           basis_pnl: float, fees: float, reason: str, liquidated: bool = False) -> Trade:
    long_name = pair.a.name if position.direction == 1 else pair.b.name
    short_name = pair.b.name if position.direction == 1 else pair.a.name
    return Trade(
        coin=pair.coin,
        direction=position.direction,
        long_venue=long_name,
        short_venue=short_name,
        entry_ms=position.entry_ms,
        exit_ms=exit_ms,
        notional=notional,
        funding_pnl=position.accrued_funding,
        basis_pnl=basis_pnl,
        fees=fees,
        exit_reason=reason,
        forecast_at_entry=position.forecast_at_entry,
        liquidated=liquidated,
    )
