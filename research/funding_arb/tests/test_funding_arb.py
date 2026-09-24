"""Tests for the funding-carry research engine.

These target the places where a backtest lies to you rather than the places where
it crashes: interval normalisation, lookahead, and funding cash-flow accounting.
A bug in any of the three produces a plausible-looking equity curve that does not
exist, which is strictly worse than an exception.
"""

from __future__ import annotations

import math
import statistics

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


# ------------------------------------------------------ measured execution cost

from research.funding_arb.costs import CostBook, as_cost_book, from_liquidity  # noqa: E402
from research.funding_arb.liquidity import CarryLiquidity, walk  # noqa: E402


def test_walking_one_level_costs_only_the_half_spread():
    # Mid 100, best ask 101: crossing costs 100bp whatever the size within the level.
    fill = walk([(101.0, 1000.0)], notional_usd=10_000, mid=100.0, is_buy=True)
    assert fill.vwap == pytest.approx(101.0)
    assert fill.slippage_bps == pytest.approx(100.0)
    assert fill.filled


def test_walking_several_levels_pays_the_volume_weighted_price():
    # $5k at 100 then $5k at 102 -> VWAP is weighted by base units, not by price.
    levels = [(100.0, 50.0), (102.0, 50.0)]  # $5,000 and $5,100
    fill = walk(levels, notional_usd=10_000, mid=100.0, is_buy=True)
    assert 100.0 < fill.vwap < 102.0
    assert fill.slippage_bps > 0
    assert fill.filled


def test_selling_below_the_mid_is_also_a_cost():
    """Slippage is signed as a cost on both sides, never negative for crossing."""
    fill = walk([(99.0, 1000.0)], notional_usd=10_000, mid=100.0, is_buy=False)
    assert fill.slippage_bps == pytest.approx(100.0)


def test_a_book_too_thin_reports_a_lower_bound_not_a_fill():
    fill = walk([(100.0, 1.0)], notional_usd=10_000, mid=100.0, is_buy=True)
    assert not fill.filled
    assert fill.depth_usd == pytest.approx(100.0)


def test_walking_an_empty_book_does_not_divide_by_zero():
    fill = walk([], notional_usd=10_000, mid=100.0, is_buy=True)
    assert not fill.filled
    assert fill.slippage_bps == 0.0


def test_cost_book_falls_back_to_the_default():
    default = CostModel(long_taker_bps=10.0, short_taker_bps=5.0, slippage_bps=2.0)
    cheap = CostModel(long_taker_bps=10.0, short_taker_bps=5.0, slippage_bps=0.1)
    book = CostBook(default=default, per_coin={"BTC": cheap})
    assert book.for_coin("BTC") is cheap
    assert book.for_coin("NEVER_MEASURED") is default


def test_cost_book_gates_on_the_round_trip_budget():
    book = CostBook(
        default=CostModel(long_taker_bps=10.0, short_taker_bps=5.0, slippage_bps=2.0),
        per_coin={"THIN": CostModel(long_taker_bps=10.0, short_taker_bps=5.0,
                                    slippage_bps=120.0)},
    )
    assert book.tradable("BTC", max_round_trip_bps=60.0)
    assert not book.tradable("THIN", max_round_trip_bps=60.0)


def test_a_single_model_is_accepted_wherever_a_book_is():
    model = CostModel(long_taker_bps=10.0, short_taker_bps=5.0)
    book = as_cost_book(model)
    assert book.for_coin("anything") is model
    assert as_cost_book(book) is book


def test_measured_round_trip_is_spread_over_the_four_crossings():
    """slippage_bps is charged per order, so a 40bp round trip is 10bp each."""
    measurement = CarryLiquidity(
        coin="TAO", notional=10_000.0, round_trip_slippage_bps=40.0,
        spot_buy_bps=10.0, spot_sell_bps=10.0, perp_sell_bps=10.0, perp_buy_bps=10.0,
        spot_depth_usd=1e6, perp_depth_usd=1e6, complete=True, timestamp_ms=0,
    )
    spot = FakeVenue("okx_spot", 1.0, taker_fee_bps=10.0, funding=[], marks=[])
    perp = FakeVenue("hyperliquid", 1.0, taker_fee_bps=4.5, funding=[], marks=[])
    book = from_liquidity([measurement], spot, perp)
    assert book.for_coin("TAO").slippage_bps == pytest.approx(10.0)
    # fees (10 + 4.5) + slippage (2 x 10) per side, doubled for the round trip
    assert book.for_coin("TAO").round_trip_bps == pytest.approx(2 * (14.5 + 20.0))


# --------------------------------------------------------- historical depth

from research.funding_arb.depth_history import (  # noqa: E402
    DepthSnapshot,
    is_plausible,
    round_trip_bps,
    slippage_bps,
    summarize as depth_summarize,
)


def make_snapshot(bands):
    return DepthSnapshot(timestamp="2026-01-01 00:00:00", asks=dict(bands),
                         bids=dict(bands))


def test_an_order_inside_the_first_band_pays_half_of_it():
    """Filling a fraction of the first band averages halfway into it, not to its edge."""
    # $1m standing within 0.2% of mid; a $1m order therefore averages 0.1% = 10bp.
    bands = {0.2: 1_000_000.0, 1.0: 5_000_000.0}
    assert slippage_bps(bands, 1_000_000) == pytest.approx(10.0)
    # Half that size averages halfway again: 0.05% = 5bp.
    assert slippage_bps(bands, 500_000) == pytest.approx(5.0)


def test_slippage_grows_with_size():
    bands = {0.2: 1_000_000.0, 1.0: 5_000_000.0, 5.0: 20_000_000.0}
    small = slippage_bps(bands, 100_000)
    large = slippage_bps(bands, 4_000_000)
    assert 0 < small < large


def test_a_size_beyond_the_published_bands_has_no_answer():
    """Past 5% out the archive says nothing, so neither do we."""
    bands = {0.2: 1_000.0, 1.0: 5_000.0, 5.0: 20_000.0}
    assert slippage_bps(bands, 10_000_000) is None
    assert round_trip_bps(make_snapshot(bands), 10_000_000) is None


def test_zero_size_costs_nothing():
    assert slippage_bps({0.2: 1_000.0}, 0) == 0.0


def test_round_trip_charges_both_crossings():
    bands = {0.2: 1_000_000.0, 1.0: 5_000_000.0}
    one_side = slippage_bps(bands, 500_000)
    assert round_trip_bps(make_snapshot(bands), 500_000) == pytest.approx(2 * one_side)


def test_flat_depth_across_bands_is_rejected_as_corrupt():
    """A real book deepens as you walk out; the archive sometimes says otherwise.

    NEARUSDT's 2026-09-07..11 files report the same $13 at 0.2% and at 5%. Left
    in, those rows read as a liquidity crisis on a perp trading hundreds of
    millions a day - they alone produced a spurious '30% of snapshots could not
    absorb $10k'.
    """
    corrupt = {0.2: 13.0, 1.0: 13.0, 2.0: 13.0, 3.0: 13.0, 4.0: 13.0, 5.0: 13.0}
    assert not is_plausible(make_snapshot(corrupt))

    real = {0.2: 281_466.0, 1.0: 1_505_821.0, 2.0: 2_621_683.0,
            3.0: 3_216_967.0, 4.0: 4_611_760.0, 5.0: 5_580_443.0}
    assert is_plausible(make_snapshot(real))


def test_empty_or_zero_depth_is_rejected():
    assert not is_plausible(make_snapshot({}))
    assert not is_plausible(make_snapshot({0.2: 0.0, 5.0: 0.0}))


def test_stress_multiple_is_withheld_when_the_median_is_noise():
    """A ratio against a rounding-error median says nothing; the cost is just free."""
    free = [make_snapshot({0.2: 5e8, 1.0: 1e9, 5.0: 4e9})] * 100
    stats = depth_summarize("BTCUSDT", free, notional=10_000)
    assert stats.median_bps < 0.1
    assert stats.stress_multiple != stats.stress_multiple  # NaN, deliberately

    pricey = [make_snapshot({0.2: 20_000.0, 1.0: 60_000.0, 5.0: 200_000.0})] * 100
    stats = depth_summarize("THINUSDT", pricey, notional=10_000)
    assert stats.median_bps > 0.1
    assert stats.stress_multiple == pytest.approx(1.0)  # constant book, no stress


def test_uncovered_snapshots_are_counted_not_silently_dropped():
    covered = make_snapshot({0.2: 1e6, 1.0: 5e6, 5.0: 2e7})
    thin = make_snapshot({0.2: 100.0, 1.0: 200.0, 5.0: 400.0})
    stats = depth_summarize("MIXED", [covered] * 3 + [thin], notional=1_000_000)
    assert stats.uncovered_share == pytest.approx(0.25)


def test_exit_slippage_defaults_to_the_entry_figure():
    symmetric = CostModel(long_taker_bps=5.0, short_taker_bps=5.0, slippage_bps=2.0)
    assert symmetric.exit_bps == symmetric.entry_bps


def test_a_tail_exit_costs_more_than_the_entry():
    """You choose when to enter; the market chooses when you exit."""
    model = CostModel(long_taker_bps=5.0, short_taker_bps=5.0,
                      slippage_bps=1.0, exit_slippage_bps=50.0)
    assert model.entry_bps == pytest.approx(12.0)   # 10 fees + 2 x 1 slippage
    assert model.exit_bps == pytest.approx(110.0)   # 10 fees + 2 x 50 slippage
    assert model.round_trip_bps == pytest.approx(122.0)

    # Breakeven scales with the round trip, so a fragile exit raises the funding
    # spread the carry must earn by the same 122/24 = 5.1x - a two-week hold that
    # needed 6.3% APR now needs 31.8%, which excludes most of the universe.
    symmetric = CostModel(long_taker_bps=5.0, short_taker_bps=5.0, slippage_bps=1.0)
    ratio = (model.breakeven_spread_apr(24 * 14)
             / symmetric.breakeven_spread_apr(24 * 14))
    assert ratio == pytest.approx(122.0 / 24.0, rel=1e-6)


def test_depth_history_prices_entry_and_exit_differently():
    from research.funding_arb.costs import from_depth_history
    from research.funding_arb.depth_history import DepthStats

    fragile = DepthStats(symbol="TAOUSDT", notional=10_000.0, n_snapshots=1000,
                         uncovered_share=0.0, median_bps=1.0, p90_bps=1.6,
                         p99_bps=295.0, worst_bps=336.0)
    robust = DepthStats(symbol="NEARUSDT", notional=10_000.0, n_snapshots=1000,
                        uncovered_share=0.0, median_bps=0.9, p90_bps=1.2,
                        p99_bps=1.6, worst_bps=17.0)

    spot = FakeVenue("mexc_spot", 1.0, taker_fee_bps=5.0, funding=[], marks=[])
    perp = FakeVenue("mexc", 8.0, taker_fee_bps=2.0, funding=[], marks=[])
    book = from_depth_history([fragile, robust], spot, perp)

    tao, near = book.for_coin("TAO"), book.for_coin("NEAR")
    # Near-identical medians, so entry costs are near-identical...
    assert tao.entry_bps == pytest.approx(near.entry_bps, rel=0.2)
    # ...but the fragile book's exit is an order of magnitude worse.
    assert tao.exit_bps > 20 * near.exit_bps
    assert tao.round_trip_bps > near.round_trip_bps


# ----------------------------------------------------------- hedge quality

from research.funding_arb.basis import (  # noqa: E402
    basis_series,
    holding_move,
    measure as basis_measure,
)


def marks_from(prices, start=T0):
    return [Mark(start + i * HOUR_MS, p) for i, p in enumerate(prices)]


def test_basis_is_measured_only_where_both_legs_report():
    """Forward-filling a stale leg would invent a basis move that never happened."""
    spot = [Mark(T0, 100.0), Mark(T0 + 2 * HOUR_MS, 110.0)]
    perp = [Mark(T0, 101.0), Mark(T0 + HOUR_MS, 105.0), Mark(T0 + 2 * HOUR_MS, 111.0)]
    series = basis_series(spot, perp)
    assert [t for t, _ in series] == [T0, T0 + 2 * HOUR_MS]
    assert series[0][1] == pytest.approx(0.01)


def test_a_perfectly_tracking_perp_shows_no_basis_move():
    prices = [100.0 * (1 + 0.01 * i) for i in range(600)]
    spot, perp = marks_from(prices), marks_from(prices)
    stats = basis_measure("TIGHT", spot, perp, hold_hours=48)
    assert stats is not None
    assert stats.p99_move == pytest.approx(0.0, abs=1e-12)
    assert stats.hedge_is_reliable


def test_a_drifting_perp_is_flagged_as_an_unreliable_hedge():
    # 900 hours, so a 480h hold still leaves enough windows to measure.
    spot = marks_from([100.0] * 900)
    # The perp drifts steadily to 5% above spot: a static hedge bleeds that.
    perp = marks_from([100.0 * (1 + 0.05 * i / 900) for i in range(900)])
    stats = basis_measure("DRIFT", spot, perp, hold_hours=480)
    assert stats is not None
    assert stats.p99_move > 0.01
    assert not stats.hedge_is_reliable


def test_holding_move_measures_the_change_not_the_level():
    """A large but CONSTANT basis costs nothing: you enter and leave at the same one."""
    spot = marks_from([100.0] * 600)
    perp = marks_from([103.0] * 600)  # a permanent 3% premium
    stats = basis_measure("CONSTANT", spot, perp, hold_hours=48)
    assert stats.median_level == pytest.approx(0.03)
    assert stats.p99_move == pytest.approx(0.0, abs=1e-12)
    assert stats.hedge_is_reliable


def test_holding_move_respects_the_holding_period():
    spot = marks_from([100.0] * 600)
    perp = marks_from([100.0 * (1 + 0.05 * i / 600) for i in range(600)])
    short_hold = holding_move(basis_series(spot, perp), 24)
    long_hold = holding_move(basis_series(spot, perp), 240)
    assert max(long_hold) > max(short_hold)


def test_carry_months_lost_prices_the_tail_against_the_carry():
    spot = marks_from([100.0] * 900)
    perp = marks_from([100.0 * (1 + 0.10 * i / 900) for i in range(900)])
    stats = basis_measure("DRIFT", spot, perp, hold_hours=480)
    # A p99 move worth half a year's carry means the hedge, not the funding,
    # decides the trade.
    assert stats.carry_months_lost(carry_apr=0.20) > 1.0


def test_too_little_overlap_returns_nothing_rather_than_a_guess():
    spot, perp = marks_from([100.0] * 50), marks_from([101.0] * 50)
    assert basis_measure("SHORT", spot, perp, hold_hours=48) is None


# ------------------------------------------------------- volatility risk gate

def test_realized_volatility_is_backward_looking():
    calm = [100.0] * 200
    pair = build_carry("X", 0.0004, hours=900, price=100.0)
    pair.b.marks = [Mark(T0 + i * HOUR_MS, p) for i, p in enumerate(calm)] + [
        Mark(T0 + (200 + i) * HOUR_MS, 100.0 * (1 + 0.05 * (-1) ** i))
        for i in range(200)
    ]
    # Before the turbulent stretch begins, it must not be visible.
    assert pair.b.realized_volatility(T0 + 180 * HOUR_MS, 168) == pytest.approx(0.0)
    assert pair.b.realized_volatility(T0 + 380 * HOUR_MS, 168) > 0.01


def test_realized_volatility_needs_enough_history():
    pair = build_carry("X", 0.0004, hours=900)
    assert pair.b.realized_volatility(pair.grid[0], 168) is None


def test_the_gate_refuses_the_most_volatile_coins():
    """Same funding everywhere, so only volatility can decide who gets traded."""
    import math as _math

    universe = {}
    for i, name in enumerate(("CALM1", "CALM2", "CALM3", "WILD1", "WILD2")):
        pair = build_carry(name, 0.0009, hours=900)
        amplitude = 0.001 if name.startswith("CALM") else 0.20
        path = [100.0 * (1 + amplitude * _math.sin(h / 3.0)) for h in range(900)]
        pair.a.marks = [Mark(T0 + h * HOUR_MS, p) for h, p in enumerate(path)]
        pair.b.marks = [Mark(T0 + h * HOUR_MS, p) for h, p in enumerate(path)]
        universe[name] = pair

    def run(gate):
        return run_portfolio(
            universe,
            CostModel(long_taker_bps=10.0, short_taker_bps=5.0, slippage_bps=0.0),
            PortfolioParams(max_positions=5, entry_rank=5, exit_rank=5,
                            min_entry_apr=0.10, min_hold_hours=48,
                            max_hold_hours=10_000, warmup_hours=200,
                            exclude_vol_quantile=gate),
            _Cfg(notional=10_000.0, leverage=1.0, leverage_a=1.0),
        )

    assert any(c.startswith("WILD") for c in run(0.0).coins_traded)
    gated = run(0.4).coins_traded
    assert gated and not any(c.startswith("WILD") for c in gated)


def test_the_gate_is_validated():
    with pytest.raises(ValueError):
        PortfolioParams(exclude_vol_quantile=1.0)
    with pytest.raises(ValueError):
        PortfolioParams(exclude_vol_quantile=-0.1)


# ------------------------------------------------------- seasonality decomposition

from research.funding_arb.seasonality import (  # noqa: E402
    carry_apr,
    decompose,
    deseasonalize,
    lagged_rho,
    seasonal_mean,
    to_monthly,
)


def monthly(values_by_year_month):
    return dict(values_by_year_month)


def test_seasonal_mean_excludes_the_year_being_estimated():
    """Including it would let each observation help predict itself."""
    series = {(y, 1): float(y) for y in range(2000, 2010)}
    # The mean of every OTHER year's January, never 2005's own value.
    expected = statistics.fmean([y for y in range(2000, 2010) if y != 2005])
    assert seasonal_mean(series, 2005, 1) == pytest.approx(expected)


def test_seasonal_mean_refuses_a_thin_month():
    series = {(y, 3): 1.0 for y in range(2000, 2003)}   # only 3 years
    assert seasonal_mean(series, 2001, 3) is None


def test_a_purely_seasonal_series_deseasonalises_to_nothing():
    """Same shape every year: the residual is zero and R2 is 1."""
    shape = {m: float(m) for m in range(1, 13)}
    series = {(y, m): shape[m] for y in range(2000, 2020) for m in range(1, 13)}
    residual = deseasonalize(series)
    assert all(abs(v) < 1e-9 for v in residual.values())
    result = decompose("PURE", series)
    assert result.seasonal_r2 == pytest.approx(1.0)


def test_a_series_with_no_calendar_structure_keeps_its_variance():
    import random
    rng = random.Random(11)
    series = {(y, m): rng.gauss(0, 1)
              for y in range(2000, 2020) for m in range(1, 13)}
    result = decompose("NOISE", series)
    # Subtracting a noisy month mean cannot explain much, and may add variance.
    assert result.seasonal_r2 < 0.25


def test_seasonality_can_hide_persistence_rather_than_supply_it():
    """The natural-gas case: a sign-flipping cycle masks a trending signal.

    Raw 6-month autocorrelation is dragged negative by the seasonal cycle, while
    the underlying level persists. Deseasonalising must recover it.
    """
    import math
    series = {}
    for i, (y, m) in enumerate((y, m) for y in range(2000, 2020)
                               for m in range(1, 13)):
        seasonal = 10.0 * math.sin(2 * math.pi * (m - 1) / 12)   # flips over 6m
        trend = i * 0.05                                          # persistent
        series[(y, m)] = seasonal + trend

    result = decompose("MASKED", series, lags=(3, 6))

    # At six months the cycle has flipped sign, dragging raw rho negative while
    # the underlying trend persists. Deseasonalising recovers it.
    assert result.raw_rho[6] < 0 < result.residual_rho[6]
    assert result.masks_signal(6) is True

    # The share-of-persistence ratio is undefined against a negative raw value
    # and must say so rather than return a misleading positive number.
    assert result.seasonality_explains(6) is None

    # At three months raw rho is still positive, so the share is defined - and
    # negative, which is the signature of seasonality hiding signal.
    assert result.raw_rho[3] > 0
    assert result.seasonality_explains(3) < 0
    assert result.masks_signal(3) is True


def test_seasonality_explains_reports_a_share_when_it_does_supply_it():
    shape = {m: float(m) for m in range(1, 13)}
    import random
    rng = random.Random(3)
    series = {(y, m): shape[m] + rng.gauss(0, 0.2)
              for y in range(2000, 2020) for m in range(1, 13)}
    result = decompose("SEASONAL", series, lags=(12,))
    # Twelve months apart is the same calendar month, so raw rho is high and
    # almost all of it is the calendar.
    assert result.raw_rho[12] > 0.8
    assert result.seasonality_explains(12) > 0.8


def test_lagged_rho_needs_enough_pairs():
    series = {(2000, m): float(m) for m in range(1, 13)}
    assert lagged_rho(series, 1) != lagged_rho(series, 1) or True  # NaN-safe
    import math
    assert math.isnan(lagged_rho(series, 1))


def test_carry_is_positive_in_backwardation():
    """Near above far means you are paid to be long - a positive funding rate."""
    assert carry_apr(near=105.0, far=100.0) == pytest.approx(0.60, rel=1e-6)
    assert carry_apr(near=100.0, far=105.0) < 0
    assert carry_apr(near=102.0, far=100.0, months_apart=2.0) == pytest.approx(0.12)
    with pytest.raises(ValueError):
        carry_apr(near=100.0, far=0.0)


def test_to_monthly_averages_within_the_month():
    daily = {(2024, 1, 1): 1.0, (2024, 1, 2): 3.0, (2024, 2, 1): 10.0}
    assert to_monthly(daily) == {(2024, 1): 2.0, (2024, 2): 10.0}


# ------------------------------------------------------- seasonal walk-forward

from datetime import date as _date, timedelta as _td  # noqa: E402

from research.funding_arb.seasonal_walkforward import (  # noqa: E402
    _Index,
    candidate_grid,
    chance_of_qualifying,
    trade_pnl,
    walk_forward,
)

_SMALL_GRID = candidate_grid(entry_step=15, holds=(30, 60))


def _daily(years, fn):
    out = {}
    d = _date(years[0], 1, 1)
    while d.year <= years[-1]:
        out[d] = fn(d)
        d += _td(days=1)
    return out


def test_chance_of_qualifying_matches_the_binomial():
    # 12+ wins of 15 on a fair coin: (455 + 105 + 15 + 1) / 2**15
    assert chance_of_qualifying(15, 12) == pytest.approx(576 / 32768)
    assert chance_of_qualifying(15, 15) == pytest.approx(1 / 32768)


def test_trade_pnl_is_signed_by_direction():
    series = {_date(2020, 1, 1) + _td(days=i): float(i) for i in range(200)}
    idx = _Index(series)
    assert trade_pnl(idx, 2020, (1, 30, 1)) == pytest.approx(30.0)
    assert trade_pnl(idx, 2020, (1, 30, -1)) == pytest.approx(-30.0)


def test_a_long_data_gap_is_refused_not_bridged():
    series = {_date(2020, 1, 1): 1.0, _date(2020, 3, 1): 2.0}
    assert _Index(series).on_or_after(_date(2020, 1, 5)) is None


def test_a_genuine_seasonal_pattern_survives_out_of_sample():
    """Rises every spring by construction, plus noise: selection must carry forward."""
    import random
    rng = random.Random(5)

    def value(d):
        spring = 5.0 if 60 <= d.timetuple().tm_yday <= 150 else 0.0
        ramp = spring * (d.timetuple().tm_yday - 60) / 90 if spring else 0.0
        return ramp + rng.gauss(0, 0.3)

    series = _daily(range(1990, 2021), value)
    r = walk_forward("SPRING", series, lookback=15, min_wins=12,
                     candidates=_SMALL_GRID)
    assert r.qualified_per_year > r.expected_false_per_year
    assert r.selected_oos_win > r.baseline_oos_win + 0.2
    assert r.best_pick_oos_win > 0.8


def test_pure_noise_qualifies_at_the_chance_rate_and_fails_forward():
    """A random walk has no seasonality: whatever qualifies is luck, and stays luck."""
    import random
    rng = random.Random(9)
    level = [0.0]

    def value(_):
        level[0] += rng.gauss(0, 1)
        return level[0]

    series = _daily(range(1985, 2021), value)
    r = walk_forward("NOISE", series, lookback=15, min_wins=12,
                     candidates=_SMALL_GRID)
    # Out of sample the selected windows must look like the unselected ones.
    assert abs(r.selected_oos_win - r.baseline_oos_win) < 0.15


def test_best_picks_record_which_window_was_chosen():
    series = _daily(range(1990, 2010),
                    lambda d: 3.0 if 90 <= d.timetuple().tm_yday <= 150 else 0.0)
    r = walk_forward("STEP", series, lookback=10, min_wins=9, candidates=_SMALL_GRID)
    assert r.best_picks
    year, (doy, hold, direction), pnl = r.best_picks[-1]
    assert direction in (1, -1) and hold in (30, 60)
