"""Performance metrics, including an honest read on the win rate.

The brief for this research was "a success rate consistently above 50%". Taken
literally that is a weak requirement - take-profit at 0.1% against a stop at 5%
clears it easily and still loses money - so two things are reported side by side:

* **Expectancy**, which is what actually determines whether the strategy makes
  money: `win_rate * avg_win - loss_rate * avg_loss`, net of all fees.
* **A Wilson confidence interval on the win rate**, because a 70% win rate over
  9 trades is indistinguishable from a coin flip. "Consistently above 50%" is a
  claim about the interval's *lower bound*, not the point estimate, and this is
  the field to read before believing any of the others.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import List, Sequence

from .backtest import BacktestResult, Trade


HOURS_PER_YEAR = 24 * 365
Z_95 = 1.959963984540054


@dataclass
class Metrics:
    n_trades: int
    n_wins: int
    win_rate: float
    win_rate_lo95: float        # Wilson lower bound - the number that matters
    win_rate_hi95: float
    beats_coinflip: bool        # lower bound strictly above 50%
    avg_win: float
    avg_loss: float             # positive magnitude
    expectancy: float           # per trade, USD, net of fees
    profit_factor: float
    total_pnl: float
    return_on_capital: float
    apr: float
    sharpe: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_hold_hours: float
    liquidations: int
    # Attribution: does the P&L come from the carry, or from an unhedged view?
    funding_pnl: float
    basis_pnl: float
    fees_paid: float
    # Does the signal actually select anything?
    mean_spread_apr: float
    mean_spread_apr_at_entry: float
    breakeven_spread_apr: float


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> tuple:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because trade counts here are small
    (tens, not thousands) and the naive interval misbehaves badly near 0 and 1.
    """
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def max_drawdown(equity: Sequence[float]) -> tuple:
    """Largest peak-to-trough fall, as (absolute, fraction of peak)."""
    peak = float("-inf")
    worst_abs, worst_pct = 0.0, 0.0
    for value in equity:
        peak = max(peak, value)
        drop = peak - value
        if drop > worst_abs:
            worst_abs = drop
        if peak > 0 and drop / peak > worst_pct:
            worst_pct = drop / peak
    return worst_abs, worst_pct


def annualized_sharpe(equity: Sequence[float]) -> float:
    """Sharpe of hourly mark-to-market returns, annualised, excess of zero.

    Treat with care: carry equity is flat for long stretches and then steps, which
    understates volatility between trades and flatters this number relative to a
    strategy that is continuously exposed.
    """
    if len(equity) < 3:
        return 0.0
    rets = [
        (b - a) / a
        for a, b in zip(equity, equity[1:])
        if a != 0
    ]
    if len(rets) < 2:
        return 0.0
    sd = statistics.pstdev(rets)
    if sd == 0:
        return 0.0
    return statistics.fmean(rets) / sd * math.sqrt(HOURS_PER_YEAR)


def summarize(result: BacktestResult) -> Metrics:
    trades: List[Trade] = result.trades
    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    n = len(trades)

    lo, hi = wilson_interval(len(wins), n)
    equity_values = [e for _, e in result.equity]
    dd_abs, dd_pct = max_drawdown(equity_values)

    total = sum(t.net_pnl for t in trades)
    capital = result.capital
    span_years = result.span_hours / HOURS_PER_YEAR if result.span_hours else 0.0

    gross_win = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)

    avg_hold = statistics.fmean([t.holding_hours for t in trades]) if trades else 0.0
    entry_spreads = [abs(t.forecast_at_entry) for t in trades]
    all_spreads = [abs(s) for _, s, _ in result.spread]

    return Metrics(
        n_trades=n,
        n_wins=len(wins),
        win_rate=len(wins) / n if n else 0.0,
        win_rate_lo95=lo,
        win_rate_hi95=hi,
        beats_coinflip=n > 0 and lo > 0.5,
        avg_win=statistics.fmean([t.net_pnl for t in wins]) if wins else 0.0,
        avg_loss=abs(statistics.fmean([t.net_pnl for t in losses])) if losses else 0.0,
        expectancy=total / n if n else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        total_pnl=total,
        return_on_capital=total / capital if capital else 0.0,
        apr=(total / capital / span_years) if capital and span_years else 0.0,
        sharpe=annualized_sharpe(equity_values),
        max_drawdown=dd_abs,
        max_drawdown_pct=dd_pct,
        avg_hold_hours=avg_hold,
        liquidations=result.liquidations,
        funding_pnl=sum(t.funding_pnl for t in trades),
        basis_pnl=sum(t.basis_pnl for t in trades),
        fees_paid=sum(t.fees for t in trades),
        mean_spread_apr=statistics.fmean(all_spreads) if all_spreads else 0.0,
        mean_spread_apr_at_entry=statistics.fmean(entry_spreads) if entry_spreads else 0.0,
        breakeven_spread_apr=result.costs.breakeven_spread_apr(avg_hold) if avg_hold else float("inf"),
    )


def format_report(result: BacktestResult, m: Metrics) -> str:
    """Human-readable summary, written to be skimmed top-down for red flags."""
    pct = lambda x: f"{x * 100:.2f}%"
    usd = lambda x: f"${x:,.2f}"

    verdict = (
        "win rate is significantly above 50%"
        if m.beats_coinflip
        else "win rate is NOT statistically distinguishable from a coin flip"
    )
    edge = "POSITIVE" if m.expectancy > 0 else "NEGATIVE"

    carry_share = (
        m.funding_pnl / (abs(m.funding_pnl) + abs(m.basis_pnl))
        if (abs(m.funding_pnl) + abs(m.basis_pnl)) > 0 else 0.0
    )

    return "\n".join([
        f"{'=' * 68}",
        f" {result.coin}  long/short across {result.venue_a} <-> {result.venue_b}",
        f" sample: {result.span_hours / 24:.0f} days"
        f"   notional/leg: {usd(result.config.notional)}"
        f"   capital: {usd(result.capital)}",
        f"{'=' * 68}",
        "",
        " TRADE STATISTICS",
        f"   trades                 {m.n_trades}",
        f"   win rate               {pct(m.win_rate)}"
        f"   [95% CI {pct(m.win_rate_lo95)} - {pct(m.win_rate_hi95)}]",
        f"   -> {verdict}",
        f"   avg win                {usd(m.avg_win)}",
        f"   avg loss               {usd(m.avg_loss)}",
        f"   expectancy/trade       {usd(m.expectancy)}   ({edge} edge)",
        f"   profit factor          {m.profit_factor:.2f}",
        f"   avg holding period     {m.avg_hold_hours:.0f}h ({m.avg_hold_hours / 24:.1f} days)",
        "",
        " P&L ATTRIBUTION",
        f"   funding collected      {usd(m.funding_pnl)}",
        f"   basis / hedge residual {usd(m.basis_pnl)}",
        f"   fees + slippage        {usd(-m.fees_paid)}",
        f"   net                    {usd(m.total_pnl)}",
        f"   carry share of gross   {pct(carry_share)}"
        f"   {'(genuinely carry-driven)' if carry_share > 0.6 else '(WARNING: mostly unhedged price risk)'}",
        "",
        " RETURNS",
        f"   return on capital      {pct(m.return_on_capital)}",
        f"   APR                    {pct(m.apr)}",
        f"   Sharpe (hourly, ann.)  {m.sharpe:.2f}",
        f"   max drawdown           {usd(m.max_drawdown)} ({pct(m.max_drawdown_pct)})",
        f"   liquidated legs        {m.liquidations}",
        "",
        " SIGNAL QUALITY",
        f"   mean |spread| overall  {pct(m.mean_spread_apr)} APR",
        f"   mean |spread| at entry {pct(m.mean_spread_apr_at_entry)} APR",
        f"   breakeven spread       {pct(m.breakeven_spread_apr)} APR"
        f" (at {m.avg_hold_hours:.0f}h holds, {result.costs.round_trip_bps:.1f}bp round trip)",
        f"   entry selectivity      "
        f"{m.mean_spread_apr_at_entry / m.mean_spread_apr:.2f}x"
        if m.mean_spread_apr > 0 else "   entry selectivity      n/a",
        f"{'=' * 68}",
    ])


def format_portfolio_report(result, m: Metrics) -> str:
    """Summary for a cross-sectional book.

    Two lines matter more than the headline return. **Utilisation** says how much
    of the capital was actually earning: a book that sits 70% empty is not a 20%
    APR strategy, it is a 6% one that occasionally looks good. **Turnover** says
    how much of the gross carry went back to the exchange - on a strategy whose
    whole thesis is "do not churn", it is the number that kills it first.
    """
    pct = lambda x: f"{x * 100:.2f}%"
    usd = lambda x: f"${x:,.2f}"

    gross = m.funding_pnl + m.basis_pnl
    fee_share = m.fees_paid / gross if gross > 0 else float("inf")
    rotations_per_year = (
        m.n_trades / (result.span_hours / HOURS_PER_YEAR) if result.span_hours else 0.0
    )

    return "\n".join([
        "=" * 68,
        f" CROSS-SECTIONAL CARRY   {len(result.coins_traded)} coin negoziate,"
        f" {result.params.max_positions} slot",
        f" sample: {result.span_hours / 24:.0f} days"
        f"   notional/leg: {usd(result.config.notional)}"
        f"   capital: {usd(result.capital)}",
        "=" * 68,
        "",
        " BOOK",
        f"   rotations              {m.n_trades}  ({rotations_per_year:.0f}/yr)",
        f"   utilisation            {pct(result.utilisation)} of slot-hours filled",
        f"   avg holding period     {m.avg_hold_hours:.0f}h"
        f" ({m.avg_hold_hours / 24:.1f} days)",
        f"   coins                  {', '.join(result.coins_traded[:14])}"
        + (" ..." if len(result.coins_traded) > 14 else ""),
        "",
        " P&L ATTRIBUTION",
        f"   funding collected      {usd(m.funding_pnl)}",
        f"   basis / hedge residual {usd(m.basis_pnl)}",
        f"   fees + slippage        {usd(-m.fees_paid)}"
        f"   ({pct(fee_share)} of gross carry)" if gross > 0 else
        f"   fees + slippage        {usd(-m.fees_paid)}",
        f"   net                    {usd(m.total_pnl)}",
        "",
        " RETURNS",
        f"   win rate               {pct(m.win_rate)}"
        f"   [95% CI {pct(m.win_rate_lo95)} - {pct(m.win_rate_hi95)}]"
        f"   {'SIGNIFICANT' if m.beats_coinflip else 'not significant'}",
        f"   expectancy/rotation    {usd(m.expectancy)}",
        f"   return on capital      {pct(m.return_on_capital)}",
        f"   APR                    {pct(m.apr)}",
        f"   Sharpe (hourly, ann.)  {m.sharpe:.2f}",
        f"   max drawdown           {usd(m.max_drawdown)} ({pct(m.max_drawdown_pct)})",
        f"   liquidated legs        {m.liquidations}",
        "=" * 68,
    ])
