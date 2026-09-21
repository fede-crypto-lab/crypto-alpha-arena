"""Tests for the funding-carry research engine.

These target the places where a backtest lies to you rather than the places where
it crashes: interval normalisation, lookahead, and funding cash-flow accounting.
A bug in any of the three produces a plausible-looking equity curve that does not
exist, which is strictly worse than an exception.
"""

from __future__ import annotations

import math

import pytest

from research.funding_arb.backtest import BacktestConfig, run_backtest
from research.funding_arb.costs import CostModel
from research.funding_arb.dataset import HOUR_MS, Pair, VenueSeries, annualize
from research.funding_arb.metrics import max_drawdown, summarize, wilson_interval
from research.funding_arb.strategy import (
    SpreadForecast,
    StrategyParams,
    desired_direction,
    should_close,
)
from research.funding_arb.venues import FundingPoint, Mark, Venue

T0 = 1_700_000_000_000 // HOUR_MS * HOUR_MS  # an arbitrary exact hour


class FakeVenue(Venue):
    """A venue whose history is handed to it, so tests never touch the network."""

    def __init__(self, name, interval_hours, taker_fee_bps, funding, marks):
        self.name = name
        self.funding_interval_hours = interval_hours
        self.taker_fee_bps = taker_fee_bps
        self._funding = funding
        self._marks = marks

    def symbol(self, coin):
        return coin

    def fetch_funding(self, coin, start_ms, end_ms):
        return self._funding

    def fetch_marks(self, coin, start_ms, end_ms):
        return self._marks


def build_pair(rate_a, rate_b, hours=400, interval_a=1.0, interval_b=8.0,
               prices_a=None, prices_b=None, fee_a=4.5, fee_b=5.0):
    """A synthetic pair with constant funding on each leg and given price paths."""
    grid = [T0 + i * HOUR_MS for i in range(hours)]
    prices_a = prices_a or [100.0] * hours
    prices_b = prices_b or [100.0] * hours

    fund_a = [FundingPoint(t, rate_a) for t in grid
              if (t - T0) % int(interval_a * HOUR_MS) == 0]
    fund_b = [FundingPoint(t, rate_b) for t in grid
              if (t - T0) % int(interval_b * HOUR_MS) == 0]

    marks_a = [Mark(t, p) for t, p in zip(grid, prices_a)]
    marks_b = [Mark(t, p) for t, p in zip(grid, prices_b)]

    a = VenueSeries(FakeVenue("a", interval_a, fee_a, fund_a, marks_a),
                    fund_a, marks_a, interval_a)
    b = VenueSeries(FakeVenue("b", interval_b, fee_b, fund_b, marks_b),
                    fund_b, marks_b, interval_b)
    return Pair(coin="TEST", a=a, b=b, grid=grid)


# ---------------------------------------------------------------- normalisation

def test_annualize_scales_by_settlements_per_year():
    # 1bp every 8h is three settlements a day: 3 * 365 * 1bp = 10.95% a year.
    assert annualize(0.0001, 8.0) == pytest.approx(0.1095)
    # The same 1bp charged hourly is eight times as much.
    assert annualize(0.0001, 1.0) == pytest.approx(0.876)


def test_identical_raw_rates_on_different_intervals_are_not_a_spread():
    """The trap: 1bp/1h and 1bp/8h look equal raw but differ 8x annualised."""
    pair = build_pair(rate_a=0.0001, rate_b=0.0001, interval_a=1.0, interval_b=8.0)
    spread = pair.spread_apr_at(pair.grid[50])
    assert spread == pytest.approx(annualize(0.0001, 8.0) - annualize(0.0001, 1.0))
    assert spread < -0.7  # leg a is vastly the more expensive one


def test_interval_is_detected_from_data_not_the_declared_constant():
    points = [FundingPoint(T0 + i * 4 * HOUR_MS, 0.0001) for i in range(10)]
    venue = FakeVenue("v", interval_hours=8.0, taker_fee_bps=5.0, funding=points, marks=[])
    assert venue.detect_interval_hours(points) == pytest.approx(4.0)


# -------------------------------------------------------------------- causality

def test_annualized_at_never_reads_the_future():
    pair = build_pair(rate_a=0.0, rate_b=0.0, hours=100)
    pair.b.funding = [FundingPoint(T0, 0.0), FundingPoint(T0 + 50 * HOUR_MS, 0.01)]
    # One millisecond before the settlement the new rate must be invisible.
    assert pair.b.annualized_at(T0 + 50 * HOUR_MS - 1) == 0.0
    assert pair.b.annualized_at(T0 + 50 * HOUR_MS) == pytest.approx(annualize(0.01, 8.0))


def test_forecast_only_uses_settled_prints():
    pair = build_pair(rate_a=0.0, rate_b=0.0, hours=200)
    pair.b.funding = [FundingPoint(T0, 0.0), FundingPoint(T0 + 100 * HOUR_MS, 0.01)]
    forecast = SpreadForecast(pair, StrategyParams(halflife_hours=10.0))
    assert forecast.at(T0 + 99 * HOUR_MS) == pytest.approx(0.0)
    assert forecast.at(T0 + 150 * HOUR_MS) > 0.0


def test_funding_between_excludes_entry_includes_exit():
    """Entering exactly on a settlement must not collect that settlement."""
    pair = build_pair(rate_a=0.0001, rate_b=0.0002, hours=50, interval_a=1.0, interval_b=1.0)
    entry, exit_ = T0 + 10 * HOUR_MS, T0 + 13 * HOUR_MS
    collected = pair.a.funding_between(entry, exit_)
    assert [p.time_ms for p in collected] == [
        T0 + 11 * HOUR_MS, T0 + 12 * HOUR_MS, T0 + 13 * HOUR_MS
    ]


# ------------------------------------------------------------------------ costs

def test_round_trip_counts_four_taker_legs():
    costs = CostModel(long_taker_bps=4.5, short_taker_bps=5.0, slippage_bps=2.0)
    assert costs.entry_bps == pytest.approx(13.5)
    assert costs.round_trip_bps == pytest.approx(27.0)
    assert costs.round_trip_cost(10_000) == pytest.approx(27.0)


def test_breakeven_falls_with_holding_period():
    costs = CostModel(long_taker_bps=5.0, short_taker_bps=5.0, slippage_bps=0.0)
    day, fortnight = costs.breakeven_spread_apr(24), costs.breakeven_spread_apr(24 * 14)
    assert day == pytest.approx(0.002 * 365)
    assert fortnight == pytest.approx(day / 14)
    assert costs.breakeven_spread_apr(0) == float("inf")


# --------------------------------------------------------------------- strategy

def test_direction_follows_the_expensive_leg():
    params = StrategyParams(entry_apr=0.10, exit_apr=0.02)
    assert desired_direction(0.15, params) == 1     # b expensive -> short b
    assert desired_direction(-0.15, params) == -1
    assert desired_direction(0.05, params) == 0


def test_reverse_can_be_forbidden_for_a_spot_leg():
    params = StrategyParams(entry_apr=0.10, exit_apr=0.02, allow_reverse=False)
    assert desired_direction(-0.50, params) == 0


def test_min_hold_overrides_decay_but_not_the_max_hold_cap():
    params = StrategyParams(entry_apr=0.10, exit_apr=0.02,
                            min_hold_hours=48, max_hold_hours=200)
    assert should_close(0.0, direction=1, held_hours=10, params=params) == ""
    assert should_close(0.0, direction=1, held_hours=60, params=params) == "spread_flip"
    assert should_close(0.5, direction=1, held_hours=250, params=params) == "max_hold"


def test_hysteresis_is_enforced():
    with pytest.raises(ValueError):
        StrategyParams(entry_apr=0.10, exit_apr=0.10)


# --------------------------------------------------------------------- backtest

def test_funding_accrual_matches_hand_computation():
    """Flat identical prices: net P&L must be exactly funding minus fees."""
    rate_b = 0.0004  # 4bp per 8h on leg b == ~43.8% APR
    pair = build_pair(rate_a=0.0, rate_b=rate_b, hours=600)
    costs = CostModel(long_taker_bps=5.0, short_taker_bps=5.0, slippage_bps=0.0)
    params = StrategyParams(entry_apr=0.20, exit_apr=0.05, min_hold_hours=48,
                            max_hold_hours=240, warmup_hours=72)
    config = BacktestConfig(notional=10_000.0, leverage=3.0)

    result = run_backtest(pair, costs, params, config)
    assert result.trades, "a 43% APR spread should have been traded"

    trade = result.trades[0]
    settlements = len([p for p in pair.b.funding
                       if trade.entry_ms < p.time_ms <= trade.exit_ms])
    assert trade.funding_pnl == pytest.approx(settlements * 10_000 * rate_b)
    assert trade.basis_pnl == pytest.approx(0.0, abs=1e-9)
    assert trade.fees == pytest.approx(20.0)  # 10bp entry + 10bp exit on 10k
    assert trade.direction == 1 and trade.short_venue == "b"


def test_perfectly_correlated_legs_leave_no_basis_pnl():
    path = [100.0 * (1.0 + 0.4 * math.sin(i / 20.0)) for i in range(600)]
    pair = build_pair(rate_a=0.0, rate_b=0.0004, hours=600,
                      prices_a=path, prices_b=list(path))
    result = run_backtest(
        pair,
        CostModel(long_taker_bps=5.0, short_taker_bps=5.0, slippage_bps=0.0),
        StrategyParams(entry_apr=0.20, exit_apr=0.05, min_hold_hours=48, max_hold_hours=240),
        BacktestConfig(notional=10_000.0, leverage=3.0),
    )
    assert result.trades
    assert all(abs(t.basis_pnl) < 1e-6 for t in result.trades)


def test_diverging_legs_liquidate_the_losing_side():
    """Leg b runs away from leg a; at 3x the short leg must be force-closed."""
    flat = [100.0] * 600
    runaway = [100.0 * (1 + 0.0015 * i) for i in range(600)]  # +90% over the sample
    pair = build_pair(rate_a=0.0, rate_b=0.0004, hours=600,
                      prices_a=flat, prices_b=runaway)
    result = run_backtest(
        pair,
        CostModel(long_taker_bps=5.0, short_taker_bps=5.0, slippage_bps=0.0),
        StrategyParams(entry_apr=0.20, exit_apr=0.05, min_hold_hours=48, max_hold_hours=2400),
        BacktestConfig(notional=10_000.0, leverage=3.0, maintenance_margin_frac=0.005),
    )
    assert result.liquidations >= 1
    assert any(t.liquidated and t.net_pnl < 0 for t in result.trades)


def test_no_trades_when_the_spread_never_clears_the_threshold():
    pair = build_pair(rate_a=0.0, rate_b=0.000001, hours=600)
    result = run_backtest(
        pair,
        CostModel(long_taker_bps=5.0, short_taker_bps=5.0),
        StrategyParams(entry_apr=0.20, exit_apr=0.05),
        BacktestConfig(),
    )
    assert result.trades == []
    assert summarize(result).n_trades == 0


def test_open_position_is_closed_at_end_of_sample():
    pair = build_pair(rate_a=0.0, rate_b=0.0004, hours=300)
    result = run_backtest(
        pair,
        CostModel(long_taker_bps=5.0, short_taker_bps=5.0),
        StrategyParams(entry_apr=0.20, exit_apr=0.05, min_hold_hours=48, max_hold_hours=10_000),
        BacktestConfig(),
    )
    assert result.trades and result.trades[-1].exit_reason == "end_of_sample"


def test_capital_is_margin_on_both_venues():
    assert BacktestConfig(notional=10_000.0, leverage=3.0).capital == pytest.approx(6_666.667)
    assert BacktestConfig(notional=10_000.0, leverage=1.0).capital == pytest.approx(20_000.0)


def test_a_spot_leg_ties_up_full_notional():
    """Cash-and-carry capital is notional (spot) + margin (perp), not 2x margin."""
    config = BacktestConfig(notional=10_000.0, leverage=3.0, leverage_a=1.0)
    assert config.capital == pytest.approx(10_000.0 + 3_333.333)


def test_an_unlevered_spot_leg_is_not_liquidated_by_a_crash():
    """The same crash that liquidates a 3x leg must leave a 1x spot leg alone.

    Modelling the spot side at perp leverage invents liquidations that cannot
    happen - the coin is owned outright - and simultaneously understates the
    capital the strategy needs.
    """
    crash = [100.0 * (1 - 0.0008 * i) for i in range(600)]  # -48% over the sample
    flat = [100.0] * 600

    def carry(**leverage):
        pair = build_pair(rate_a=0.0, rate_b=0.0004, hours=600,
                          prices_a=crash, prices_b=flat)
        return run_backtest(
            pair,
            CostModel(long_taker_bps=5.0, short_taker_bps=5.0, slippage_bps=0.0),
            StrategyParams(entry_apr=0.20, exit_apr=0.05, min_hold_hours=48,
                           max_hold_hours=2400, allow_reverse=False),
            BacktestConfig(notional=10_000.0, **leverage),
        )

    assert carry(leverage=3.0).liquidations >= 1
    assert carry(leverage=3.0, leverage_a=1.0).liquidations == 0


# ---------------------------------------------------------------------- metrics

def test_wilson_lower_bound_rejects_a_small_sample():
    """7 wins out of 10 is not evidence of an edge; 700 out of 1000 is."""
    lo_small, _ = wilson_interval(7, 10)
    lo_large, _ = wilson_interval(700, 1000)
    assert lo_small < 0.5
    assert lo_large > 0.5


def test_wilson_handles_the_degenerate_cases():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    lo, hi = wilson_interval(5, 5)
    assert hi == pytest.approx(1.0) and 0.0 < lo < 1.0


def test_max_drawdown_measures_peak_to_trough():
    abs_dd, pct_dd = max_drawdown([100.0, 120.0, 90.0, 110.0])
    assert abs_dd == pytest.approx(30.0)
    assert pct_dd == pytest.approx(0.25)


def test_a_high_win_rate_can_still_be_a_losing_strategy():
    """The reason expectancy, not win rate, is the headline metric."""
    pair = build_pair(rate_a=0.0, rate_b=0.0004, hours=600)
    result = run_backtest(
        pair,
        CostModel(long_taker_bps=5.0, short_taker_bps=5.0),
        StrategyParams(entry_apr=0.20, exit_apr=0.05, min_hold_hours=48, max_hold_hours=240),
        BacktestConfig(),
    )
    m = summarize(result)
    # The sample is profitable overall and wins most of its trades, yet with a
    # handful of trades the Wilson bound still sits below 50% - so the claim
    # "consistently above 50%" is not supportable from this evidence, however
    # good the point estimate looks.
    assert m.total_pnl > 0
    assert m.win_rate > 0.5
    assert m.win_rate_lo95 < 0.5
    assert not m.beats_coinflip


# ------------------------------------------------------- cross-sectional book

from research.funding_arb.backtest import BacktestConfig as _Cfg  # noqa: E402
from research.funding_arb.portfolio import (  # noqa: E402
    PortfolioParams,
    run_portfolio,
    trailing_apr,
)


def build_carry(coin: str, perp_rate: float, hours: int = 900, price: float = 100.0):
    """A spot/perp pair: leg a is cash (zero funding), leg b is the perp."""
    pair = build_pair(rate_a=0.0, rate_b=perp_rate, hours=hours,
                      interval_a=1.0, interval_b=8.0,
                      prices_a=[price] * hours, prices_b=[price] * hours,
                      fee_a=10.0, fee_b=5.0)
    pair.coin = coin
    return pair


def test_trailing_apr_is_causal_and_absent_without_history():
    """The ranking must not score a coin on funding the venue has not charged yet."""
    pair = build_carry("X", 0.0004)

    # A window that ends before the series begins has nothing to rank on - the
    # coin stays out of the ranking instead of scoring as zero.
    assert trailing_apr(pair, pair.grid[0] - HOUR_MS, window_hours=168) is None

    # Raise the rate partway through; before the change the trailing mean must
    # still show only the old rate.
    cutover = pair.grid[500]
    pair.b.funding = [
        FundingPoint(p.time_ms, 0.0004 if p.time_ms <= cutover else 0.0040)
        for p in pair.b.funding
    ]
    assert trailing_apr(pair, cutover, 168) == pytest.approx(annualize(0.0004, 8.0))
    # Well after the change, the new rate dominates.
    assert trailing_apr(pair, cutover + 168 * HOUR_MS, 168) == pytest.approx(
        annualize(0.0040, 8.0)
    )


def test_book_never_exceeds_its_slot_count():
    universe = {
        c: build_carry(c, r)
        for c, r in [("A", 0.0009), ("B", 0.0008), ("C", 0.0007), ("D", 0.0006)]
    }
    result = run_portfolio(
        universe,
        CostModel(long_taker_bps=10.0, short_taker_bps=5.0, slippage_bps=0.0),
        PortfolioParams(max_positions=2, entry_rank=4, exit_rank=4,
                        min_entry_apr=0.10, min_hold_hours=48, warmup_hours=200),
        _Cfg(notional=10_000.0, leverage=3.0, leverage_a=1.0),
    )
    assert len(result.coins_traded) <= 4
    assert 0.0 < result.utilisation <= 1.0
    # Capital is sized for the slots, filled or not.
    assert result.capital == pytest.approx(2 * (10_000.0 + 10_000.0 / 3))


def test_ranking_prefers_the_richest_funding():
    universe = {
        "RICH": build_carry("RICH", 0.0009),
        "POOR": build_carry("POOR", 0.00005),
    }
    result = run_portfolio(
        universe,
        CostModel(long_taker_bps=10.0, short_taker_bps=5.0, slippage_bps=0.0),
        PortfolioParams(max_positions=1, entry_rank=1, exit_rank=2,
                        min_entry_apr=0.10, min_hold_hours=48, warmup_hours=200),
        _Cfg(notional=10_000.0, leverage=3.0, leverage_a=1.0),
    )
    assert result.coins_traded == ["RICH"]


def test_the_absolute_floor_overrides_a_good_rank():
    """Being the best of a bad universe is not a reason to pay a round trip."""
    universe = {c: build_carry(c, 0.000002) for c in ("A", "B", "C")}
    result = run_portfolio(
        universe,
        CostModel(long_taker_bps=10.0, short_taker_bps=5.0),
        PortfolioParams(max_positions=2, entry_rank=2, exit_rank=3,
                        min_entry_apr=0.15, warmup_hours=200),
        _Cfg(notional=10_000.0, leverage=3.0, leverage_a=1.0),
    )
    assert result.trades == []
    assert result.utilisation == 0.0


def test_hysteresis_is_required_between_entry_and_exit_rank():
    with pytest.raises(ValueError):
        PortfolioParams(entry_rank=8, exit_rank=3)


def test_book_pnl_is_funding_minus_fees_when_prices_are_flat():
    universe = {"A": build_carry("A", 0.0009)}
    costs = CostModel(long_taker_bps=10.0, short_taker_bps=5.0, slippage_bps=0.0)
    result = run_portfolio(
        universe, costs,
        PortfolioParams(max_positions=1, entry_rank=1, exit_rank=2,
                        min_entry_apr=0.10, min_hold_hours=48,
                        max_hold_hours=10_000, warmup_hours=200),
        _Cfg(notional=10_000.0, leverage=3.0, leverage_a=1.0),
    )
    assert result.trades
    trade = result.trades[0]
    settlements = len([p for p in universe["A"].b.funding
                       if trade.entry_ms < p.time_ms <= trade.exit_ms])
    assert trade.funding_pnl == pytest.approx(settlements * 10_000 * 0.0009)
    assert trade.basis_pnl == pytest.approx(0.0, abs=1e-9)
    assert result.liquidations == 0


# ------------------------------------------------------- rank persistence

from research.funding_arb.persistence import measure, spearman  # noqa: E402


def test_spearman_recovers_a_monotonic_relationship():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_ignores_the_scale_of_outliers():
    """Funding has a hard floor and a long right tail, so ranks beat levels."""
    assert spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 10_000]) == pytest.approx(1.0)


def test_spearman_is_defined_on_degenerate_input():
    assert spearman([1, 2], [1, 2]) == 0.0          # too few points
    assert spearman([1, 1, 1], [5, 5, 5]) == 0.0    # no variance


def test_persistence_detects_a_stable_ranking():
    """Coins with permanently different funding must show near-perfect rho."""
    panel = {
        f"C{i}": [FundingPoint(T0 + h * HOUR_MS, 0.00001 * i) for h in range(24 * 80)]
        for i in range(1, 26)
    }
    result = measure(panel, T0, T0 + 80 * 24 * HOUR_MS, window_days=7,
                     interval_hours=1.0)
    assert result is not None
    assert result.mean_rho > 0.95
    assert result.top_quintile_apr > result.median_apr > result.bottom_quintile_apr
    assert result.spread_apr > 0


def test_persistence_reports_no_edge_on_a_shuffled_ranking():
    """The control: funding that alternates sign period to period must not rank."""
    import random
    rng = random.Random(7)
    panel = {}
    for i in range(25):
        pts = []
        for h in range(24 * 80):
            # Re-draw every 7 days, so the trailing window never predicts the next.
            if h % (24 * 7) == 0:
                rate = rng.uniform(-0.0001, 0.0003)
            pts.append(FundingPoint(T0 + h * HOUR_MS, rate))
        panel[f"C{i}"] = pts
    result = measure(panel, T0, T0 + 80 * 24 * HOUR_MS, window_days=7,
                     interval_hours=1.0)
    assert result is not None
    assert abs(result.mean_rho) < 0.5


def test_persistence_needs_enough_coins():
    panel = {f"C{i}": [FundingPoint(T0 + h * HOUR_MS, 0.0001) for h in range(24 * 60)]
             for i in range(3)}
    assert measure(panel, T0, T0 + 60 * 24 * HOUR_MS, 7, min_coins=20) is None
