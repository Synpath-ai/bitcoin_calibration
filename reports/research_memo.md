# Research Memo: Bitcoin Prediction Market Calibration & Execution-Aware Backtest

This memo reports only what was measured on the actual downloaded sample. See
`data_quality_report.md` for the full data-quality breakdown and `README.md` for
methodology. All figures below are pulled from real runs of the pipeline against
real, live-fetched Polymarket/Binance data (`data/processed/*`), not simulated or
invented.

## Sample

- **507** resolved daily "Bitcoin Up or Down" markets discovered and structurally
  validated (of 527 discovered; 20 excluded — see data quality report), spanning
  **2025-03-14 to 2026-08-25** (~17.5 months).
- **2,520** fixed-horizon observations (24h/12h/6h/3h/1h before resolution) across
  **506** of those markets (one market's price series was too stale/short-lived to
  contribute observations at any horizon).
- Realized Up frequency across the 506 markets: **52.6%** — close to, but not
  exactly, a fair coin flip.

## 1. Is the raw Polymarket probability calibrated?

Yes, closely, at every horizon we could measure — with the expected pattern that the
market gets *more informative*, not more or less calibrated, as resolution approaches:

| Horizon | N obs / markets | Brier (95% CI) | Calib. slope / intercept |
|---|---|---|---|
| 24h | 499 / 499 | 0.2496 [0.248, 0.251] | 0.93 / 0.08 |
| 12h | 505 / 505 | 0.1868 [0.170, 0.203] | 0.99 / 0.02 |
| 6h  | 506 / 506 | 0.1423 [0.127, 0.159] | 1.06 / 0.02 |
| 3h  | 504 / 504 | 0.1112 [0.095, 0.127] | 1.17 / -0.02 |
| 1h  | 506 / 506 | 0.0653 [0.053, 0.079] | 1.12 / 0.06 |

At 24h out, the market's Brier score (0.2496) is statistically indistinguishable
from a constant-50% forecast (0.2500) — 24 hours before resolution there is
essentially no exploitable information in the price yet, which is exactly what
you'd expect for a same-day BTC coin-flip. By 1h out, Brier drops to 0.065 and
calibration slope stays close to 1 (perfect = 1.0) at every horizon inside 12h.
**We find no strong, consistent evidence that the raw price is systematically
mis-calibrated** — slopes hover near 1 and intercepts near 0 throughout, well
within bootstrap CIs. The one horizon with a slope furthest from 1 (3h, slope
1.17) is not a stable trend across neighboring horizons and should not be read as
a systematic effect without more data.

## 2. Does calibration error vary by regime?

The dashboard's Mispricing Analysis tab computes calibration gaps by probability
bucket, BTC momentum sign, BTC volatility tercile, and market-volume quartile, each
with market-level bootstrap CIs and a `statistically_significant` flag requiring
≥20 independent markets AND a CI excluding zero. **We did not find a bucket that
clears that bar with a economically large, consistently-signed gap across
horizons** — most bucket-level gaps are within their own confidence intervals of
zero once market-level (not row-level) clustering is accounted for. This is a
genuine negative result worth taking seriously, not a shortfall of the analysis:
with ~500 independent markets split across 10 probability buckets and 5 horizons,
many cells simply don't have the sample size to detect anything but a large effect.
**News-intensity and social-sentiment regime breakdowns could not be computed** —
GDELT was unreachable for the historical backfill in this run (0% coverage; see
Known Limitations) and no social-media source was available. This is a real gap in
the analysis, not a claim that those factors don't matter.

## 3. Does any apparent mispricing survive execution costs?

> ⚠️ **The P&L figures in this section are superseded by Parts 5 and 6.** They were
> produced with an execution model that scaled the spread proportionally
> (real books quote a flat ~1c) and sized positions off market volume rather
> than book depth. Part 5 corrects both; Part 6 then re-fits the depth
> constants to 1,782 measured order books. Read Part 6 for current figures.

We backtested three strategies — raw-probability, calibration-only (Platt scaling),
and the full-feature model — walk-forward, under optimistic/base/conservative
execution-cost scenarios, with `estimated_execution`-tier fills throughout (see
Known Limitations — no historical order book/trades exist to do better than this).

| Strategy | Scenario | Signals | Fills | Net P&L | Hit rate | Max DD | Sharpe (per-trade) |
|---|---|---|---|---|---|---|---|
| raw_probability | all | 0 | 0 | $0 | — | — | — |
| calibration_only | base | 174 | 174 | **-$7,086** | 72.4% | 11.9% | -0.03 |
| full_feature | base | 407 | 407 | **+$35,003** | 56.3% | 22.0% | 0.05 |

(base scenario shown; optimistic/conservative move net P&L by roughly ±10%, see
`backtest_summary.csv` / dashboard — the qualitative ranking across strategies does
not change across scenarios.)

Three things stand out, and we report them honestly rather than picking the
flattering one:

- **The raw-probability strategy trades against its own price by construction** and
  correctly never finds a positive edge (0 signals) — a useful sanity check that the
  cost model and edge logic aren't leaking information.
- **The calibration-only strategy loses money despite a 72% hit rate.** It mostly
  buys already-extreme-probability contracts where a small residual mispricing
  exists but the payoff per win is small and a rare loss wipes out several wins'
  worth of edge — a textbook case of hit rate being the wrong metric to look at in
  isolation for an asymmetric payoff.
- **The full-feature model's positive P&L is concentrated at short horizons, not
  the 24h horizon that generates 75% of its trade volume.** Broken down by horizon
  (base scenario): the 24h bucket (304/407 trades) nets **-$7,419** with a 50.3% hit
  rate — indistinguishable from noise, consistent with our Part 1 finding that the
  market carries no information at 24h. The 1h/3h/6h/12h buckets (103/407 trades)
  contribute essentially all of the positive total. **This means the walk-forward
  full-feature model is generating a large number of low-quality 24h signals that
  happen not to lose much money in this particular 17.5-month sample, and a smaller
  number of higher-quality short-horizon signals that plausibly reflect real
  information (BTC's own short-term momentum/volatility).** We would not present
  the aggregate +$35k number as evidence of a durable, tradable 24h edge; the
  short-horizon result is more credible but rests on ~103 trades, most of them
  in the 12h bucket which the same model calibrates worse than raw probability
  does at that horizon (see walk-forward table) — treat it as suggestive, not
  conclusive.
- The full-feature strategy also fires on **86% of eligible test markets** at the
  default `min_model_edge=0.03` — a signal rate that high for a claimed edge is
  itself a flag for likely overconfidence in the model's probability estimates
  rather than genuinely abundant mispricing, and raises the max drawdown into the
  low-20%s, close to the 25% configured stop.

## 4. Replication check: does the edge hold up in a larger sample?

> ⚠️ **The P&L figures in this section are superseded by Parts 5 and 6.** They were
> produced with an execution model that scaled the spread proportionally
> (real books quote a flat ~1c) and sized positions off market volume rather
> than book depth. Part 5 corrects both; Part 6 then re-fits the depth
> constants to 1,782 measured order books. Read Part 6 for current figures.

The daily-market backtest's positive result rested on a small number of short-
horizon trades (~103, see Part 3), which is too little to be confident in. To test
whether it replicates, we reran the identical pipeline — same feature set, same
walk-forward logic, same execution-cost model, same risk engine — on Polymarket's
**hourly** BTC Up/Down market (`btc-up-or-down-hourly`, same Up/Down-on-a-Binance-
candle mechanic, 24x the frequency of the daily one). We pulled the most recent
**1,999 resolved hourly markets** (2026-06-02 to 2026-08-26, ~86 days — this is a
*different, more recent, and shorter* window than the daily sample's Mar 2025–Aug
2026, not the same period at higher frequency; see caveat below), giving **9,974
observations** across 5 minutes-before-resolution horizons (45/30/15/5/2 min,
scaled down from the daily 24/12/6/3/1h since this market only runs ~1 hour).

**Calibration**: the raw price is well-calibrated at every horizon (slopes
0.79–1.19, all within a bootstrap CI of 1.0) and, unlike the daily 24h horizon,
even the *longest* horizon here (45min before resolution — the closest analogue
to daily's 24h) carries real information: Brier 0.215 vs. the 0.250 constant-50%
baseline, compared to daily 24h's 0.2496 (statistically indistinguishable from
uninformative).

**Walk-forward + backtest** (base scenario, full-feature strategy): **1,195
fills**, net P&L **+$157,957** on $100k initial capital, 59.9% hit rate, max
drawdown 20.1%. Crucially, the pattern that undermined the daily result does
**not** repeat here — the trade volume is NOT dominated by an uninformative
horizon:

| Horizon | Trades | Net P&L | Hit rate |
|---|---|---|---|
| 45 min | 810 (68%) | +$58,710 | 63.7% |
| 30 min | 245 | +$26,458 | 52.2% |
| 15 min | 100 | +$50,744 | 53.0% |
| 5 min | 29 | +$9,587 | 41.4% |
| 2 min | 11 | +$12,459 | 63.6% |

The largest bucket (45min, 68% of trades) is now the *best*-performing one, not
the worst — consistent with the calibration finding that this horizon is
genuinely informative for the hourly market, unlike the daily market's 24h
horizon. This is a meaningfully different and more encouraging picture than the
daily-only result, **conditional on the caveats below** — we are not calling this
a confirmed, durable edge, but the earlier concern ("positive P&L was an artifact
of noisy 24h-horizon trades") specifically does not hold up here.

The calibration-only (Platt-scaling) strategy again **loses money** on this
larger sample too (-$15,624 base scenario, 40.0% hit rate) — reinforcing Part 3's
point that a high hit rate / simple recalibration is not by itself evidence of a
tradable edge; the full-feature model's use of BTC's own momentum/volatility
features is doing real work here, not just re-centering the market price.

**Caveats specific to this replication, read together with Part 3/5 above:**

- **Not the same time period.** The hourly sample is a recent 86-day window, the
  daily sample spans 17.5 months — a genuine difference in BTC regime, not a
  pure sample-size increase. We could not pull the hourly series' full history
  (its offset-based pagination caps at ~2,100 events; the true history goes back
  to May 2025) within this project's time budget; a full backfill via Gamma's
  keyset pagination endpoint (`/events/keyset`) is a natural next step.
- **More acute serial correlation.** Hourly BTC candles on the same day are far
  more autocorrelated with each other than daily closes are, so "1,999 markets"
  overstates independence more than "507 daily markets" did — market-level
  bootstrap CIs (one resample unit per market) are a weaker correction here.
- **Same execution-cost-model caveat as the daily backtest** — no historical
  order book/trades for the hourly market either; every fill is `estimated_execution`.
- Full detail (per-scenario tables, horizon breakdowns, calibration diagrams) is
  in the dashboard's "6 · Extended Sample (Hourly)" tab.

## 5. Revised execution model: absolute spread floor + order-book capacity

Two flaws in the original cost model were found by checking it against live
order books, and both have been fixed. The headline P&L falls substantially.

**(a) The spread was modelled proportionally; real books quote a flat ~1 cent.**
The original half-spread scaled with the contract price, so a 5c contract was
charged ~0.03c one-way against the ~0.5c actually quoted -- roughly 17x light.
Polymarket's own UI reports "Spread: 1c" on these books irrespective of level.
An absolute floor of $0.005 per side now applies, and it is deliberately *not*
relaxed by the optimistic scenario: you cannot trade inside the quoted spread.

**(b) Position size was capped off market *volume*, not book *depth*.** Volume
is a flow over a market's whole life; depth is the stock resting on the book at
the instant you trade. Live sampling found only a few hundred dollars within a
cent of the touch, while the volume-based cap permitted ~$2,000 positions --
i.e. the backtest's typical trade was several times the entire visible
near-touch liquidity. The book is now modelled explicitly (a touch tranche plus
a further tranche per cent walked), fills are priced at the walked VWAP, and an
order that cannot be filled within 3 cents is recorded as **unfilled** rather
than silently filled at a good price.

| full_feature_strategy | before | after | change |
|---|---|---|---|
| Daily, base | $35,003 (407 fills, ~$1,822 avg) | **$25,469** (294 fills, ~$749 avg) | −27% |
| Hourly, base | $157,957 (1,195 fills, ~$2,128 avg) | **$36,575** (734 fills, ~$748 avg) | −77% |
| Hourly, optimistic | $183,192 | $68,623 | |
| Hourly, conservative | $117,106 | $19,848 | |

**What survives:** the edge is still positive under every scenario, and the
per-trade return is actually *higher* than before (daily 11.6%, hourly 6.7% per
trade) because the surviving trades are the ones small enough to execute at or
near the touch. Fill counts fall by ~30-40% as the thinnest opportunities become
genuinely unfillable.

**What changes:** this is now unambiguously a **small-capacity** strategy. At
~$750 per position and 25 concurrent slots it absorbs roughly $19k of working
capital, not the $100k the original run assumed. The scenario range also widens
sharply (hourly $20k-$69k), which is the honest reflection of the fact that the
depth parameters are anchored to only a handful of live samples.

**Still unresolved:** the depth constants are the weakest input in the whole
model. Polymarket publishes no historical order books, so they were set from
live spot checks -- several of them on freshly seeded, zero-volume markets
rather than the 45/30/15/5/2-minutes-to-resolution points where the strategy
actually trades. The live paper trader now logs executable depth on every
10-minute check-in; once that accumulates, these constants should be re-fit to
measured depth at trade-relevant horizons and the backtest re-run. Until then,
treat the scenario range, not the base case, as the honest answer.

## 6. Depth constants re-fitted to measured order books

Part 5's depth parameters were guesses anchored to a handful of live spot checks.
The live paper trader has since logged **1,782 real order-book snapshots**
(2026-08-28 to 2026-09-08), so they are now fitted to measurement.

**Depth collapses as resolution approaches** — the opposite of convenient, since
that is where the model has the most information:

| Time to resolution | n | Median depth within 1c |
|---|---|---|
| >3h | 762 | $2,217 |
| 1-3h | 78 | $424 |
| 30-60min | 468 | $263 |
| 15-30min | 311 | $220 |
| 5-15min | 152 | **$123** |

**The two families differ by ~10x** (daily median $1,979, hourly $241), because
they trade in completely different windows. Constants are therefore now
per-family, fitted over each one's actual trading window:

| | measured touch depth (p25 / median / p75) | per-cent | previously assumed |
|---|---|---|---|
| Daily (1-24h) | $587 / **$2,050** / $5,251 | $1,501 | $300 / $150 — ~8x too conservative |
| Hourly (2-45min) | $93 / **$233** / $481 | $135 | $300 / $150 — slightly optimistic |

The existing optimistic/conservative multipliers (2x / 0.5x) bracket roughly the
measured p25-p75 range, so the scenario band is now empirically grounded too.

**Re-run results (`full_feature_strategy`):**

| | optimistic | base | conservative |
|---|---|---|---|
| Daily | $33,754 | **$52,899** | $42,639 |
| Hourly | $62,616 | **$33,306** | $15,183 |

**The daily result is non-monotonic, and that is a real effect rather than a bug.**
Looser cost assumptions produce *more* trades (388 vs 324) of *worse* quality
(4.76% vs 8.00% return per dollar staked). The mechanism is a strategy design
flaw the sensitivity analysis exposed: cheaper assumed execution lets marginal
signals clear the edge bar at the *earliest* horizon, and because the backtest
opens at most one position per market and takes the first qualifying horizon, it
locks in a 24h entry (276 of 388 trades under optimistic) and forgoes the far
more informative 1h signal it would otherwise have waited for. The 24h bucket
loses money in every scenario. Preferring later horizons, rather than the first
that qualifies, is the obvious fix and is not yet implemented.

**Caveat unchanged in kind, reduced in degree:** these constants are measured
rather than assumed, but over an 11-day window in one BTC regime, and the p25-p75
spread is wide (daily $587-$5,251). The scenario range remains the honest answer.

## 7. Live forward test — the first genuinely out-of-sample evidence

Everything above is a backtest. The live paper trader has now been running for
11 days on markets that did not exist when any model was fitted, entering at the
**real best ask it observed** (`actual_book`), not a modelled fill. Settling
those recommendations against actual resolutions (`scripts/18_settle_paper_trades.py`)
gives the first out-of-sample read on whether the edge is real.

The trader re-evaluates each market every 10 minutes, so the raw log contains
713 recommendations covering only 82 distinct markets. Counting each line as a
trade would multiply one bet into dozens; the settlement keeps the first
qualifying signal per market, matching the backtest's one-position-per-market
rule.

| | settled trades | staked | realized P&L | hit rate |
|---|---|---|---|---|
| Daily | 6 | $6,057 | **-$2,228 (-36.8%)** | 33.3% |
| Hourly | 74 | $57,959 | **+$911 (+1.6%)** | 52.7% |
| **Combined** | **80** | **$64,017** | **-$1,317 (-2.1%)** | 51.2% |

**The headline is negative, but it is not evidence against the edge.** Signals
claimed +6.2% average edge and the stake-weighted realization is -2.1%. That gap
looks damning until it is decomposed, and an earlier draft of this section
overstated it:

- **Noise dominates.** Single-trade return dispersion is **126%** (binary
  contracts pay 1 or 0), against a claimed edge of ~6%. At n=74 the standard
  error of the mean is 14.7%. **Even if the true edge were exactly +5%, there is
  a ~37% chance of observing ≤0 over 74 trades.** This sample cannot separate
  "no edge" from "the backtested edge, unlucky fortnight".
- **The daily −36.8% is six trades**, five of which entered at the 12-24h
  horizon — precisely where the design flaw in Part 6 pushes entries. It carries
  no statistical weight.
- **Like-for-like, live is not worse.** Comparing the longest (least informative)
  horizon in each: backtest 45min bucket +2.3%/trade at 61.6% hit; live 30-60min
  **+6.1%/trade** at 56.8% hit. Per trade, live is running *ahead* at the
  comparable horizon. The negative total comes from stake-weighting plus the six
  daily trades.
- **Model overconfidence is visible but not established.** Hourly: predicted win
  rate 57.0%, realized 52.7% — a −4.3% gap inside a ±11.4% CI.

**But the sample cannot settle the question either way.** For the hourly family
(n=74) the mean per-trade return is +7.8% with a 95% CI of **[-21.5%, +37.1%]**,
p=0.60 — statistically indistinguishable from zero. At the observed dispersion,
detecting a true 5% edge would take roughly **2,455 trades, about a year** of
running at this cadence. The daily family's 6 trades are worth nothing
statistically and should not be read as evidence of anything.

**What this does establish:**

- The pipeline works end to end on live data: discovery, book reads, model
  inference, sizing and risk checks all execute unattended, and entries are
  taken at prices actually quoted.
- The realized total is negative, but at ~126% per-trade dispersion and n=80 it
  is statistically indistinguishable from the backtested edge, from zero, and
  from a modest negative edge alike. No directional conclusion is available yet.
- The honest summary of this project, as of now, is: *a well-calibrated market,
  a backtest that suggests a modest edge, and a forward test that has not
  confirmed it.* Anyone reading the backtest numbers without this section would
  overstate what has been demonstrated.

## 8. Out-of-sample calibration: the model is not more accurate than the price

Part 7's 80 settled trades were too few to conclude anything, and they were also
produced by a live trader that (until 2026-09-08) was not running the backtest's
strategy — it evaluated every 10 minutes rather than at fixed horizons, kept no
portfolio state, and applied no depth cap. Those divergences invalidate the P&L
comparison.

They do **not** invalidate the probability estimates. Every check-in recorded the
model's probability and the market's price for a market that later resolved, and
neither depends on how the trader sized or timed anything. That gives **1,858
scored estimates across 187 resolved markets** — roughly 23x the evidence in the
trade log — to answer the question the entire project rests on: out of sample, is
the model more accurate than the market price?

Hourly family (1,057 observations, 179 markets), Brier bootstrapped by market:

| | Brier (95% CI) | Calibration slope | Intercept |
|---|---|---|---|
| Market price | **0.1689** [0.1500, 0.1899] | 1.03 | +0.05 |
| Model | 0.1732 [0.1522, 0.1947] | 0.92 | **+0.29** |

Paired per market: model Brier − market Brier = **+0.0037** (p=0.324).

**The model is not beating the price.** The difference is not statistically
significant, but the point estimate is on the wrong side, and the calibration
intercept is telling: the market's +0.05 is near-perfectly centred, while the
model's **+0.29** indicates a systematic bias in its probability estimates that
the market does not share.

This matters more than the P&L result. A trading rule can only extract value if
the probability feeding it is better than the price it trades against. On the
only genuinely out-of-sample evidence available, it is not — which is consistent
with Part 1's finding that the raw market price was already well calibrated, and
suggests the backtested edge is more likely an artifact of the historical fit
than a property of the market.

The daily family has only 7 resolved markets in the live window and is not
scored.

## 9. Replaying the aligned strategy over the recorded live data

The live trader's own entries used the wrong timing, but it checked in every 10
minutes and recorded the model probability, the book and the volume each time.
The check-ins that landed on a backtest horizon therefore contain what the
aligned strategy would have seen — 634 of them across 159 markets — so the rule
can be replayed without waiting to accumulate new data
(`scripts/20_replay_aligned_strategy.py`).

This is a replay, not a fresh live test: prices, depths and probabilities are all
as observed, and none of these markets existed when the model was fitted, but the
decision to trade is applied after the fact.

| | trades | return on stake | hit rate |
|---|---|---|---|
| Backtest (hourly, base) | 711 | +7.35% | 60.1% |
| **Aligned replay (hourly)** | 121 | **+5.68%** | 52.1% |
| Actual live, unaligned | 74 | +1.57% | 52.7% |

Aligning the entry rule moves the result from roughly flat to positive, and the
combined replay across both families is +8.66% on $24,692 staked. **That headline
is fragile, and the reason matters:** the stake-weighted return is +5.68% while
the *equal-weighted* mean per trade is **−7.4%**. The positive total rests on a
few large winning positions rather than on the typical trade, which loses money.
Note this is the opposite asymmetry to Part 7, where the equal-weighted mean was
positive and the stake-weighted return negative — with samples this size the two
weightings can disagree in either direction, which is itself a sign that neither
is pinning anything down.

Statistically it settles nothing: n=121, p=0.394, 95% CI **[−24.5%, +9.7%]**.

The more informative number is the hit rate. The replay wins 52.1% of the time
(95% CI [43.2%, 60.8%]) against the backtest's 60.1%. The backtest figure sits
just inside the interval, so it is not contradicted — but the point estimate is
well below it, and that is consistent with Part 8, where the model's
probabilities were not more accurate than the market's price. Two independent
cuts of the live data point the same way: the edge, if present, is smaller than
the backtest implies.

## 10. Re-scoring the replay with a properly walk-forward-trained model

Part 9's replay used `calibrated_fair_value` as it was actually logged, which
Part 10's root-cause finding (the refresh-training-data commit) traces to a model
trained on a frozen 2026-08-26 snapshot -- 2 to 12 days stale across the replay
window, and skewed toward predicting Down (mean 0.478 vs the market's 0.517).

This section asks the sharper question: with the same real market states (book,
price, volume -- none of that depends on the model) but a PROPERLY walk-forward-
trained model at each point in time (only markets resolved before the test
market opened, refit whenever that pool changes, exactly matching
src/walkforward.py), what would the aligned strategy have entered?
(`scripts/21_rescore_live_with_fresh_model.py`.) BTC features are recomputed
point-in-time from the (now date-complete) OHLCV history via the same
`observation_builder._btc_features` the training pipeline uses, so this
introduces no new kind of look-ahead beyond what the rest of the project
already carries.

| | trades | staked | P&L (stake-wtd) | P&L (equal-wtd mean) | hit rate |
|---|---|---|---|---|---|
| Part 9 (stale model) | 121 | $22,765 | +5.68% | **-7.4%** | 52.1% |
| **Part 10 (fresh model)** | 92 | $19,302 | **+23.74%** | **+6.6%** | **65.2%** |

Same underlying market snapshots, same entry rule, same sizing rule -- only the
probability feeding the decision differs. Three things changed together:

- **The two weightings now agree in sign.** Parts 7 and 9 each had stake-weighted
  and equal-weighted returns pointing opposite ways -- a sign a few large
  positions were driving the total, not typical trade quality. Here both are
  positive.
- **The NO skew is far less extreme.** 78/92 trades are still buy_no, but that is
  a directional tilt rather than the near-total 121/121 of Part 9 -- consistent
  with the fresh model's mean probability (0.493) sitting close to the market's
  (0.510) rather than well below it.
- **Hit rate now brackets the backtest's.** 65.2% (95% CI [55.1%, 74.4%])
  comfortably contains the backtest's 60.1% -- the closest alignment between any
  live-derived figure and the backtest anywhere in this memo.

**It is still not statistically significant**: p=0.460, and the sample shrank to
92 (some snapshots fell below the 30-market minimum training pool and were
dropped -- verified this was not concentrated in any one part of the window, all
7 days are represented in both the old and new samples). The daily family (5
trades, all buy_yes) is too small to read at all.

**Where this leaves the project:** the single largest quantifiable problem
uncovered in this memo -- 12 days of frozen training data -- is fixed, and every
indicator moved in the same encouraging direction when it was corrected. That is
a meaningfully better position than Part 9 left off at, but "encouraging" is the
right word, not "confirmed": n=92 at ~100%+ per-trade dispersion cannot yet
separate this from noise. The refresh pipeline (`refresh_training_data.yml`) now
keeps training data current going forward, so genuinely fresh live check-ins
will accumulate real (not reconstructed) evidence from here.

## Discussion: what would change our confidence

- **Repeated-market dependence.** Every observation is ultimately a draw from "is
  BTC's price up over roughly one day," repeated ~500 times over one BTC market
  regime (mostly a 2025-2026 window). All CIs here are market-level bootstrapped to
  respect that, but 500 correlated daily draws of one asset's short-term direction
  is a much smaller effective sample than 2,520 rows suggests.
- **Selection bias.** We only include markets whose resolution rules and outcome
  mapping we could verify; 20 markets (16 anomalously long-duration, 4 re-listed
  under an altered slug) were excluded. This is unlikely to bias calibration
  materially given the small excluded count, but it is not zero.
- **Historical order-book limitations.** Every historical fill is a *modeled*
  execution price, not an observed one — see README. The backtest numbers above
  should be read as "under this transparent, documented cost assumption," not as a
  claim of achieved historical fills.
- **Estimated vs. executable prices.** The live paper trader, by contrast, prices
  off the real live order book (`actual_book` tier) — a meaningfully higher-fidelity
  input than anything available historically. As of 2026-09-08 the recorded fill
  is a real VWAP walked against the recorded depth bands
  (`walk_real_book()` in `src/paper_trader.py`), not the flat best_ask a prior
  version of the live trader recorded — validated offline first (Part 10's
  92-trade rescore: stake-weighted return +23.74% → +21.89%, 8/92 trades lost
  their edge once walked) before being ported into the production entry path.
- **Data leakage controls.** Walk-forward training strictly uses only markets
  resolved before the test market opened (checked directly per test market, not
  assumed monotonic — see `tests/test_walkforward.py` and `test_no_lookahead`-style
  checks in `test_observation_builder.py`). We did not find evidence of look-ahead
  leakage in testing.
- **Does miscalibration survive costs?** Given Part 2's negative finding (no
  strong, statistically robust regime-level miscalibration detected), the honest
  answer is: **we don't have strong evidence of a systematic mispricing to begin
  with**, so the backtest's positive full-feature P&L is better read as the model
  finding *some* real short-horizon signal in BTC's own momentum/volatility rather
  than as confirmation of a specific, named Polymarket mispricing pattern. That
  distinction matters for how much confidence to place in the result going forward.
  Part 4's hourly-market replication (~4x the markets, ~12x the trades) supports
  this "real signal, not a 24h-horizon artifact" reading — the edge shows up
  broadly across horizons there, including the longest/highest-volume one — but
  it comes from a different, more recent time window, so it is corroborating
  evidence, not independent confirmation on the same period.
