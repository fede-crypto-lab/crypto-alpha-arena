# CLAUDE.md — research/funding_arb

Research module for funding-rate carry. Read this before touching anything here.
It applies to this directory and overrides the repository root `CLAUDE.md` where
the two disagree.

## Read these first, in this order

1. **`MEMORY.md`** — every established number, every dead end, every bug. Check it
   before measuring anything: most questions are already answered there, and
   re-measuring burns hours to reproduce a figure that is already written down.
2. **`NEXT_SESSION.md`** — the current task brief.
3. `README.md` — the long-form analysis (Italian). Sections 3, 6 and 10 are the
   ones that carry the argument.

Docs are in Italian because the user is; code, comments and commit messages are
in English. Keep that split.

## What this module is, and is not

It is a **research framework**, not a trading bot. Nothing here places an order,
holds a key, or talks to an authenticated endpoint. Every data source is public.
That separation is deliberate and must survive: if a task needs order placement,
it belongs in a new module, not in this one.

## The rules that produced the results

**Falsify before you build.** Every strategy here was given a cheap test that
could kill it before any backtest was written. The cross-sectional carry survived
because `persistence.py` was written first and returned ρ≈0.65; the time-series
version died because its entry selectivity came back at 0.83. If a new idea has
no cheap falsification test, find one before writing the engine.

**Nothing reads the future.** At grid time `t`, only data with timestamp `<= t` is
visible. Forward-filling the last known value is legal; interpolating between two
prints is not, because it leaks the next one backwards. Two tests enforce this
(`test_annualized_at_never_reads_the_future`,
`test_forecast_only_uses_settled_prints`). Do not weaken them.

**Costs are measured, not assumed.** `liquidity.py` walks live books,
`depth_history.py` reads Binance's archives. A flat slippage number is a last
resort and must be labelled as one wherever a result is reported.

**Normalise funding intervals.** Hyperliquid settles hourly, most CEXs every 8h,
and some MEXC symbols every 4h. A 1bp/1h rate and a 1bp/8h rate are the same raw
number and differ 8× annualised. Everything used for a signal goes through
`annualize()`, and the interval is **detected from the data**, never taken from
the declared constant.

**Report the Wilson lower bound, not the win rate.** These strategies open tens of
trades a year, so a 100% win rate over 14 trades is not evidence. `beats_coinflip`
is the field to read. A high win rate with a wide interval is a warning.

**Attribute the P&L.** Every result splits into funding / basis / fees. If the
carry share of gross drops below ~60%, the position is not market-neutral and the
result is a directional bet wearing a costume.

## Invariants worth re-checking after any engine change

- Recorded `basis_pnl` on a closed trade equals what the entry and exit marks
  imply, to the cent. This was verified on 16/16 trades when a −$1,259 figure
  looked like a bug and turned out to be real. If it ever fails, the engine is
  wrong, not the market.
- Capital is `notional/leverage_a + notional/leverage_b`. A spot leg is 1× and
  cannot be liquidated.
- Funding accrues on `(entry, exit]` — entering exactly on a settlement collects
  nothing.

## Bugs that produced plausible numbers instead of errors

Four of the seven bugs in `MEMORY.md` §10 returned believable output rather than
raising. That is why tests here assert invariants and relationships, not just
absence of exceptions. When adding a test, ask what a *silently wrong* version of
the code would return, and assert against that.

Specific traps that are now covered and must stay covered: tied ranks in
`spearman`, order-book truncation, the `"-5.00"` percentage field, corrupt archive
days with flat depth bands, and per-host request pacing (without it MEXC drops
coins from the universe mid-load and the run stops being reproducible).

## Commands

```bash
pytest research/funding_arb/tests/ -q            # 68 tests, all must pass

python -m research.funding_arb.run --persistence --days 1095    # falsification test
python -m research.funding_arb.run --portfolio --days 540 \
    --venue-a mexc_spot --venue-b mexc --no-reverse \
    --max-positions 5 --entry-rank 5 --exit-rank 20 \
    --min-hold 480 --leverage 1 --slippage-bps 5
python -m research.funding_arb.run --liquidity --notional 10000  # live book cost
python -m research.funding_arb.run --depth-history --depth-days 30
python -m research.funding_arb.run --basis --days 540 --venue-a mexc_spot --venue-b mexc
```

HTTP responses cache to `.cache/` (gitignored). `--no-cache` clears it. A full
universe load is thousands of paced requests — run it in the background and expect
tens of minutes on a cold cache.

## Venue notes that will otherwise cost an hour

- **MEXC** needs a browser User-Agent (`BROWSER_UA`) or its CDN returns an HTML
  "Access Denied" page, and it throttles intermittently even when accepted. Both
  are handled as transient. Do not remove `_HOST_MIN_INTERVAL`.
- **Hyperliquid** serves 3 years of funding but caps candles at 5,000 bars
  (208 days). That cap, not the funding, is what limits a long HL backtest.
- **Bybit and Binance REST** are geo-blocked from cloud containers; their adapters
  are correct and work from an unrestricted host.
- Binance's *data dumps* (`data.binance.vision`) are reachable even when its REST
  API is not.

## House style

Match the surrounding code: dataclasses, type hints, no external dependencies
beyond the standard library in the research modules (`pytest` and `pyflakes` for
development only). Comments explain *why a number or a guard exists*, not what the
line does. Run `python -m pyflakes research/funding_arb/*.py` before committing.

## Boundaries

Do not place orders, connect to a live brokerage account, or write code that
could. Paper/simulated endpoints only, and only when the task asks for them. When
a task would cross that line, stop and say so.
