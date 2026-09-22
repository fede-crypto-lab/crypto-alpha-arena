"""Separate a commodity's carry from its seasonal component.

Commodity carry and commodity seasonality are largely the same number: the slope
of the gas curve in November *is* winter seasonality. So "does carry persist" and
"is this just seasonality" are not two questions, they are one, and the way to
answer it is a decomposition rather than a search.

The obvious approach - scan for calendar patterns that repeat - does not work and
the arithmetic says why. A seasonal search across ~50 commodities, calendar and
inter-commodity spreads, and entry/exit windows spans roughly 120 million
combinations; against 30 years of history, pure chance alone yields some 86,000
spreads that win 24 years out of 30 and about 509 that win 27. "Won 27 of the last
30 years" is what random data looks like at that width, and the published
out-of-sample evidence agrees (arXiv 2609.12227 finds no seasonal model beating an
equal-weight long benchmark once costs and multiple comparisons are accounted for).

What this module does instead is subtract the seasonal mean and ask what is left.
Measured on EIA energy curves the answer was not the expected one: at a one to
three month horizon, deseasonalising does not weaken carry persistence, it
*strengthens* it - natural gas goes from -0.102 to +0.338 at three months, because
the seasonal cycle flips sign over that span and masks the underlying signal.
Seasonality turned out to be the noise hiding the edge rather than the edge.

The seasonal mean is always computed leave-one-out: the estimate for a given year
excludes that year. Including it would let each observation predict itself, which
manufactures exactly the persistence the test is meant to detect.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: (year, month) -> value
MonthlySeries = Dict[Tuple[int, int], float]

#: Below this many other years, a calendar month's mean is too noisy to subtract
#: and the observation is dropped rather than adjusted with a guess.
MIN_YEARS_PER_MONTH = 5


@dataclass
class Decomposition:
    name: str
    n_observations: int
    #: Share of the carry's variance explained by calendar month alone.
    seasonal_r2: float
    #: Rank autocorrelation of the raw series, by lag in months.
    raw_rho: Dict[int, float]
    #: Same for the series with its seasonal mean removed.
    residual_rho: Dict[int, float]

    def seasonality_explains(self, lag: int) -> Optional[float]:
        """Fraction of the raw persistence at `lag` that was seasonal.

        Near 1 means the apparent predictability was the calendar; negative means
        the seasonal component was *hiding* signal rather than supplying it.

        Undefined, and returns None, when the raw correlation is not positive:
        "what share of the persistence was seasonal" presupposes persistence to
        apportion. Left as a ratio it misreports badly - a raw of -0.65 against a
        residual of +0.99 yields +2.5, which reads as "seasonality explained more
        than all of it" when the truth is the opposite. Use `masks_signal` for
        that case.
        """
        raw, res = self.raw_rho.get(lag), self.residual_rho.get(lag)
        if raw is None or res is None or raw <= 0:
            return None
        return 1.0 - res / raw

    def masks_signal(self, lag: int) -> Optional[bool]:
        """True when removing seasonality *improves* persistence at this lag.

        The direct comparison, valid whatever the sign of the raw correlation.
        """
        raw, res = self.raw_rho.get(lag), self.residual_rho.get(lag)
        if raw is None or res is None or raw != raw or res != res:
            return None
        return res > raw


def seasonal_mean(series: MonthlySeries, year: int, month: int,
                  min_years: int = MIN_YEARS_PER_MONTH) -> Optional[float]:
    """Mean value for that calendar month across all OTHER years."""
    others = [v for (y, m), v in series.items() if m == month and y != year]
    return statistics.fmean(others) if len(others) >= min_years else None


def deseasonalize(series: MonthlySeries,
                  min_years: int = MIN_YEARS_PER_MONTH) -> MonthlySeries:
    """Subtract each month's leave-one-out seasonal mean."""
    out: MonthlySeries = {}
    for (year, month), value in series.items():
        mean = seasonal_mean(series, year, month, min_years)
        if mean is not None:
            out[(year, month)] = value - mean
    return out


def lagged_rho(series: MonthlySeries, lag_months: int,
               min_pairs: int = 40) -> float:
    """Spearman correlation between the series and itself `lag` months later."""
    from .persistence import spearman

    now, later = [], []
    for (year, month) in sorted(series):
        shifted_year = year + (month + lag_months - 1) // 12
        shifted_month = (month + lag_months - 1) % 12 + 1
        target = series.get((shifted_year, shifted_month))
        if target is not None:
            now.append(series[(year, month)])
            later.append(target)
    if len(now) < min_pairs:
        return float("nan")
    return spearman(now, later)


def decompose(name: str, series: MonthlySeries,
              lags: Tuple[int, ...] = (1, 3, 6, 12)) -> Optional[Decomposition]:
    residual = deseasonalize(series)
    if len(residual) < 60:
        return None

    aligned = {k: series[k] for k in residual}
    total_var = statistics.pvariance(list(aligned.values()))
    residual_var = statistics.pvariance(list(residual.values()))
    r2 = 1.0 - residual_var / total_var if total_var > 0 else 0.0

    return Decomposition(
        name=name,
        n_observations=len(residual),
        seasonal_r2=r2,
        raw_rho={L: lagged_rho(aligned, L) for L in lags},
        residual_rho={L: lagged_rho(residual, L) for L in lags},
    )


def to_monthly(daily: Dict[Tuple[int, int, int], float]) -> MonthlySeries:
    """Average a daily series into (year, month) buckets."""
    buckets: Dict[Tuple[int, int], List[float]] = {}
    for (year, month, _), value in daily.items():
        buckets.setdefault((year, month), []).append(value)
    return {k: statistics.fmean(v) for k, v in buckets.items()}


def carry_apr(near: float, far: float, months_apart: float = 1.0) -> float:
    """Annualised curve slope. Positive = backwardation = paid to be long.

    This is the commodity analogue of a positive funding rate, and the analogy is
    close enough that `persistence.py` consumes it unchanged.
    """
    if far <= 0 or months_apart <= 0:
        raise ValueError("far price and months_apart must be positive")
    return (near / far - 1.0) * (12.0 / months_apart)


def format_decompositions(rows: List[Decomposition],
                          lags: Tuple[int, ...] = (1, 3, 6, 12)) -> str:
    if not rows:
        return "nothing to decompose"

    header = "series".ljust(12) + "obs".rjust(6) + "seasonal R2".rjust(13)
    for lag in lags:
        header += (f"{lag}m raw").rjust(10) + (f"{lag}m resid").rjust(11)
    lines = ["", header, "-" * len(header)]
    for row in rows:
        line = row.name.ljust(12) + str(row.n_observations).rjust(6)
        line += f"{row.seasonal_r2 * 100:.1f}%".rjust(13)
        for lag in lags:
            line += f"{row.raw_rho.get(lag, float('nan')):.3f}".rjust(10)
            line += f"{row.residual_rho.get(lag, float('nan')):.3f}".rjust(11)
        lines.append(line)
    lines += [
        "-" * len(header),
        " seasonal R2 is the share of carry variance explained by calendar month.",
        " 'resid' subtracts that month's mean computed leave-one-out, so nothing",
        " predicts itself. Where resid EXCEEDS raw, the seasonal cycle was masking",
        " the signal rather than providing it.",
        " Caution: a 1-month lag on a monthly-averaged, slow-moving series is",
        " partly mechanical. The 3 and 6 month columns carry the information.",
    ]
    return "\n".join(lines)
