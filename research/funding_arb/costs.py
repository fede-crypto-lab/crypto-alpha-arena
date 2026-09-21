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

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """Per-notional costs of running one leg-pair.

    All figures are basis points (1bp = 0.01%) of the *per-leg* notional. A trade
    with notional Q on each side pays these on Q, twice per leg-pair round trip.
    """

    long_taker_bps: float
    short_taker_bps: float
    #: Market-impact + spread crossing beyond the quoted fee, per order.
    slippage_bps: float = 2.0
    #: Fraction of notional lost when a leg has to be re-hedged after a partial fill,
    #: amortised per round trip. Set to 0 to model perfect simultaneous execution.
    execution_slip_bps: float = 0.0

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
        return self.entry_bps

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
