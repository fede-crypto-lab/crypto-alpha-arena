"""Transaction-cost model.

This is the module that decides whether the strategy exists. A cross-venue funding
carry opens and closes *four* taker orders (long leg in, short leg in, long leg out,
short leg out). At retail taker tiers that is roughly 20bp of notional round-trip
before slippage - and a typical 8h funding spread is 1-3bp. The edge is therefore
not in the rate, it is in holding the position for enough settlements to clear a
fixed cost. Everything downstream depends on getting this number honest, so the
defaults here are deliberately pessimistic rather than promotional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


@dataclass(frozen=True)
class CostModel:
    """Per-notional costs of running one leg-pair.

    All figures are basis points (1bp = 0.01%) of the *per-leg* notional. A trade
    with notional Q on each side pays these on Q, twice per leg-pair round trip.
    """

    long_taker_bps: float
    short_taker_bps: float
    #: Market-impact + spread crossing beyond the quoted fee, per order, on the
    #: way IN. Entry is discretionary - a carry can wait for a calm book.
    slippage_bps: float = 2.0
    #: Same on the way OUT, defaulting to the entry figure. It should usually be
    #: worse, and often much worse: an exit is triggered by funding decaying or
    #: flipping, which happens when the crowded side is being liquidated - that
    #: is, precisely when depth has gone. Measured on Binance's archives, TAO and
    #: ENA carry 99th-percentile costs 200-300x their median while NEAR and WLD
    #: sit at 1.8x on near-identical medians, so a symmetric cost model does not
    #: merely understate the exit, it fails to distinguish fragile books from
    #: robust ones at all.
    exit_slippage_bps: Optional[float] = None
    #: Fraction of notional lost when a leg has to be re-hedged after a partial fill,
    #: amortised per round trip. Set to 0 to model perfect simultaneous execution.
    execution_slip_bps: float = 0.0

    @property
    def _exit_slippage(self) -> float:
        return self.slippage_bps if self.exit_slippage_bps is None else self.exit_slippage_bps

    @property
    def entry_bps(self) -> float:
        """Cost of opening both legs, in bp of one leg's notional."""
        return (
            self.long_taker_bps + self.short_taker_bps
            + 2 * self.slippage_bps
            + self.execution_slip_bps
        )

    @property
    def exit_bps(self) -> float:
        """Cost of closing both legs, in bp of one leg's notional."""
        return (
            self.long_taker_bps + self.short_taker_bps
            + 2 * self._exit_slippage
            + self.execution_slip_bps
        )

    @property
    def round_trip_bps(self) -> float:
        return self.entry_bps + self.exit_bps

    def round_trip_cost(self, notional: float) -> float:
        return notional * self.round_trip_bps / 10_000.0

    def entry_cost(self, notional: float) -> float:
        return notional * self.entry_bps / 10_000.0

    def exit_cost(self, notional: float) -> float:
        return notional * self.exit_bps / 10_000.0

    def breakeven_spread_apr(self, holding_hours: float) -> float:
        """Annualised funding spread needed to cover a round trip held this long.

        This is the number that should drive the entry threshold. Holding a
        20bp round trip for 24h needs a ~73% APR spread to break even; holding the
        same trade for 14 days needs ~5.2%. Nothing about the strategy makes sense
        until you look at this curve.
        """
        if holding_hours <= 0:
            return float("inf")
        return (self.round_trip_bps / 10_000.0) * (24 * 365 / holding_hours)


def from_venues(long_venue, short_venue, **kwargs) -> CostModel:
    """Build a CostModel from two `Venue` instances' published taker tiers."""
    return CostModel(
        long_taker_bps=long_venue.taker_fee_bps,
        short_taker_bps=short_venue.taker_fee_bps,
        **kwargs,
    )


@dataclass
class CostBook:
    """Per-coin costs, so a backtest stops pretending every coin executes alike.

    A flat slippage figure is the single most misleading input in a
    cross-sectional carry. Measured on live books, the round trip runs from under
    1bp on BTC to several hundred on a thin perp - and the ranking, left alone,
    selects precisely the thin ones, because a rate is high partly *because*
    nobody will take the other side. Feeding one average into the book therefore
    understates cost exactly where the positions are.
    """

    default: CostModel
    per_coin: Dict[str, CostModel] = field(default_factory=dict)

    def for_coin(self, coin: str) -> CostModel:
        return self.per_coin.get(coin, self.default)

    def tradable(self, coin: str, max_round_trip_bps: float) -> bool:
        """Whether this coin's measured cost is inside the budget."""
        return self.for_coin(coin).round_trip_bps <= max_round_trip_bps


def as_cost_book(costs) -> CostBook:
    """Accept either a single model or a book, so callers can pass either."""
    return costs if isinstance(costs, CostBook) else CostBook(default=costs)


def from_liquidity(measurements, long_venue, short_venue) -> CostBook:
    """Build a per-coin book from `liquidity.scan` output.

    The measured round trip covers all four crossings, so it is divided by four
    to land in `slippage_bps`, which the model applies per order. Coins whose
    book could not absorb the size are kept with the cost they did show, which is
    a lower bound - the caller should be excluding them on `complete`, not
    trusting the number.
    """
    default = CostModel(
        long_taker_bps=long_venue.taker_fee_bps,
        short_taker_bps=short_venue.taker_fee_bps,
    )
    per_coin = {
        m.coin: CostModel(
            long_taker_bps=long_venue.taker_fee_bps,
            short_taker_bps=short_venue.taker_fee_bps,
            slippage_bps=m.round_trip_slippage_bps / 4.0,
        )
        for m in measurements
    }
    return CostBook(default=default, per_coin=per_coin)


def from_depth_history(stats: Sequence, long_venue, short_venue,
                       exit_quantile: str = "p99") -> CostBook:
    """Build a book that prices entry at the median and exit at the tail.

    Takes `depth_history.DepthStats`. The asymmetry is the whole point: a carry
    is opened when its operator chooses and closed when the market decides, and
    the market decides during the moments when the book is thinnest. Pricing both
    sides at the median assumes an exit that can be scheduled, which is the one
    thing this strategy cannot do.

    The archives are Binance's, so the *level* belongs to Binance. Their value
    here is the ratio between a typical moment and a bad one, which travels
    across venues far better than the absolute figure does.
    """
    default = CostModel(
        long_taker_bps=long_venue.taker_fee_bps,
        short_taker_bps=short_venue.taker_fee_bps,
    )
    per_coin = {}
    for row in stats:
        coin = row.symbol.replace("USDT", "")
        tail = getattr(row, f"{exit_quantile}_bps")
        per_coin[coin] = CostModel(
            long_taker_bps=long_venue.taker_fee_bps,
            short_taker_bps=short_venue.taker_fee_bps,
            # The archive figure covers a round trip on one leg (in and out), so
            # halving it gives the cost of a single crossing.
            slippage_bps=row.median_bps / 2.0,
            exit_slippage_bps=tail / 2.0,
        )
    return CostBook(default=default, per_coin=per_coin)
