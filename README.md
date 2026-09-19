# Bitcoin Prediction Market Calibration & Execution-Aware Backtest

A research (not production-trading) project analyzing Polymarket's recurring daily
**"Bitcoin Up or Down"** markets: are their implied probabilities calibrated against
realized outcomes, does any miscalibration vary by regime, and does it survive
realistic execution costs? Includes a walk-forward-trained model, an execution-aware
backtest, a live (paper-only) trader, and a 5-tab Streamlit dashboard with a
global Daily/Hourly market-family toggle.

**This is a research and paper-trading system. It never submits real orders and never
requests wallet credentials.**

## Research question

1. Were Polymarket's historical implied YES probabilities for daily BTC Up/Down
   markets calibrated against realized outcomes, at horizons of 24h/12h/6h/3h/1h
   before resolution?
2. Does calibration error vary systematically by probability level, time-to-resolution,
   BTC momentum/volatility, news intensity, sentiment, or Polymarket liquidity/volume?
3. Does any apparent mispricing survive conservative execution costs (spread, slippage,
   fees, participation and position limits)?
4. What does the current walk-forward-trained model imply for the latest active market?

## Data sources

| Source | Used for | Notes |
|---|---|---|
| `gamma-api.polymarket.com` | Market discovery, metadata, resolution, current book via linked CLOB tokens | Public, no key. `series_id` (resolved dynamically from the `btc-up-or-down-daily` series slug) is the only reliable event filter — the `series=<slug>` query param is silently ignored by the API (see `src/polymarket_client.py`). |
| `clob.polymarket.com` | Historical price series (`/prices-history`), live order book (`/book`) | Public reads, no key. `/trades` requires an API key we don't have; historical order-book snapshots are not exposed at all. |
| Binance public REST (`api.binance.com/api/v3/klines`) | BTC/USDT spot OHLCV, 5-minute bars | No key required. |
| GDELT DOC 2.0 API | News intensity (article counts, source counts) | No key required, but rate-limited to ~1 req/5s per source IP; from this project's sandbox that limit was **inconsistently enforced** — sometimes reachable, sometimes a persistent 429 (see Known Limitations). |
| — | Social/media attention | No credentialed source available; adapter implemented, defaults to unavailable (see Known Limitations). |

Every raw API response used to build the dataset is cached under `data/raw/` for
reproducibility (re-running the pipeline replays from cache instead of re-hitting
the network).

## Market inclusion rules

Markets are discovered via Gamma's `btc-up-or-down-daily` series (id resolved
dynamically, never hard-coded) and then **independently re-validated** against:

- Title matches `bitcoin up or down on ...`
- Slug matches `bitcoin-up-or-down-on-<month>-<day>[-<year>][-noon]`
- Description contains the expected Binance/close/candle resolution language
- Start/end timestamps present, parseable, and end > start
- Duration between 20h and 96h (daily markets run ~1-4 days end-to-end; anything
  outside that range is treated as a data anomaly, not a normal daily market)
- Outcomes are exactly `["Up", "Down"]` with two CLOB token ids and a condition id
- Final `outcomePrices` resolve unambiguously to Up (≥0.99/≤0.01) or Down

A market failing any check is **excluded** with a recorded reason (see
`reports/data_quality_report.md` and the dashboard's Data Quality tab) — never
silently dropped. As of this run: **507 markets accepted**, **20 excluded**
(16 `duration_out_of_range`, 4 `slug_pattern_mismatch` — both real anomalies:
re-listed/duplicate markets and markets left open ~4x longer than normal).

## Feature definitions

For each accepted, resolved market, an observation is built at
`resolution_time - {24,12,6,3,1}h`, using **only** data at-or-before that timestamp
(last-observation-carried-forward with an explicit staleness tolerance — 20 min for
Polymarket price, 15 min for BTC price; if nothing is within tolerance, the
observation is dropped and the reason recorded, never imputed).

- **`yes_probability`**: last Polymarket "Up" token price at/before the observation timestamp.
- **BTC features** (`btc_return_15m/1h/6h`, `btc_realized_vol_6h`, `btc_spot_volume_1h`):
  computed from Binance 5-minute BTC/USDT candles strictly at-or-before the observation
  timestamp. Realized vol = stdev of trailing 6h of 5-minute log returns.
- **`market_volume` / `market_liquidity`**: Gamma's reported market volume/liquidity.
  **Limitation**: `liquidityNum` is structurally 0 for every market in this dataset —
  it reflects AMM-pool liquidity, which these order-book (CLOB) markets don't use.
  Total volume is substituted as the depth proxy everywhere liquidity would otherwise
  be used (spread model, position sizing caps) — see `src/execution.py` and `src/risk.py`.
- **News features** (`article_count_1h/6h/24h`, `news_intensity_change`,
  `n_distinct_sources_24h`, `avg_tone`): from GDELT DOC 2.0, `mode=artlist`, restricted
  to articles seen at/before the observation timestamp. `avg_tone` is left NaN even when
  GDELT is reachable — `artlist` mode doesn't return per-article tone (would need
  `tonechart` mode or GKG, out of scope here). Every row carries `news_data_available`;
  see Known Limitations for why this is 0% in the committed dataset.
- **Social features**: adapter implemented (`src/social_client.py`), returns
  `social_data_available=False` unless a user supplies a CSV of real timestamped
  activity — no credentialed API was available, and Reddit's unauthenticated JSON
  endpoints only expose *current* listings, not a genuine historical series, so using
  them would be indistinguishable from fabrication. **Clearly not the same thing as
  the GDELT news-media proxy** — the two are separate modules/columns by design.
- **Data-quality flags**: `price_staleness_sec`, `btc_staleness_sec`,
  `news_data_available`, `social_data_available` travel with every observation.

## Calibration methodology

Per horizon: Brier score, log loss, Expected Calibration Error (10 fixed 0.1-wide
bins), a logistic calibration-in-the-large fit (`y ~ a + b·logit(p)`; slope=1/intercept=0
is perfect), a fixed-bin reliability table, and an adaptive quantile-bin reliability
table (for horizons where fixed bins are sparse). Every point estimate reports a 95%
CI from **1,000 block-bootstrap resamples at the market level** — an entire market's
observations are resampled together, never individual rows — because a market's five
horizons are correlated draws of the same coin-flip, not independent observations.
`calibration_gap = realized_frequency - avg_implied_probability`; positive means YES
was historically underpriced, negative means overpriced. A gap is only flagged
`statistically_significant` in the dashboard when the bucket has ≥20 independent
markets AND a 95% CI excluding 0.

## Walk-forward design

Markets are sorted by `start_date`. For each test market, the training pool is
**every market whose `end_date` is strictly before this market's `start_date`** —
checked directly per test market via a `searchsorted` on end dates, not assumed
monotonic, because Polymarket runs 2-3 daily BTC markets concurrently (a new market
opens before the prior one resolves). Markets below `min_train_markets=30` are
recorded as `insufficient_history` and skipped, never predicted. All preprocessing
(feature scaling) and model fitting happen inside `model.fit(train_df)`, called only
on that window's training data — see `src/models.py` / `src/walkforward.py` and
`tests/test_walkforward.py` for the explicit leakage checks.

Models compared: constant 50%, raw Polymarket probability, Platt-scaled (logistic)
calibration, isotonic calibration, and an L2-regularized logistic regression on
`[logit(raw_prob), horizon_hours, btc_return_15m/1h/6h, btc_realized_vol_6h,
btc_spot_volume_1h, market_volume, market_liquidity]` (news/social omitted from this
feature set — 0% available in this run, see Known Limitations).

## Execution assumptions

Polymarket's public API does not expose historical order-book snapshots, and
`clob.polymarket.com/trades` requires an API key we don't have — confirmed by direct
testing (see `src/execution.py`). **Every historical backtest fill in this project is
therefore labeled `estimated_execution`** (tier 3 of the documented hierarchy:
`actual_book` > `trade_proxy` > `estimated_execution`) — a transparent spread+slippage
model as a function of market depth (volume, substituting for the unpopulated
liquidity field), evaluated under optimistic/base/conservative scenarios (0.5x/1x/2x
cost multipliers). The live paper trader, by contrast, uses the **real** live order
book (`actual_book` tier) since that endpoint does work for currently-open markets:
a candidate signal is detected at the flat touch price, then the intended stake is
walked against the real recorded depth-within-band figures (`walk_real_book()` in
`src/paper_trader.py`) to get a true VWAP fill price, and the edge is re-checked
at that VWAP — a trade whose edge only existed at the touch and disappears once
the real depth is walked is recorded as **unfilled**, not opened. This was
validated offline first (`scripts/21_rescore_live_with_fresh_model.py`, rescoring
92 historical live check-ins: stake-weighted return fell from +23.74% to +21.89%
and 8 trades lost their edge and were dropped) before being applied to the
production entry path. YES and NO are modeled as **separate long-only contracts** — no short-selling / no
costless "sell YES" is assumed, per project scope. Fees default to 0 bps (Polymarket
charges no standard taker fee on these markets) but are fully configurable for
sensitivity analysis.

## Risk constraints

Configurable in `src/risk.py` (`RiskConfig`): max risk per market, max total capital
deployed, max participation rate, max trade size, max simultaneous positions, min
market volume, min model edge, min hours-to-resolution, fractional-Kelly cap, and a
max-drawdown stop. Default position size = `min(0.10 × Kelly size, liquidity-based cap,
participation cap, per-market risk cap, remaining capital, max trade size)`.

## Extended sample: hourly BTC Up/Down markets (replication check)

The daily market family only has 507 resolved markets total (the series launched
March 2025), which left the backtest's positive result resting on a small
(~103-trade) short-horizon subsample. To test whether that edge replicates with
more data, `scripts/11-17` rerun the identical pipeline against Polymarket's
**hourly** BTC Up/Down market (`btc-up-or-down-hourly`) — same mechanic, 24x the
frequency. This required a separate discovery module
(`src/market_matching_hourly.py`) because that series' `endDate` field is not the
true resolution time and its slug format has changed multiple times; both are
handled explicitly (with tests) rather than assumed. Every dashboard tab has a
"Market family: Daily / Hourly extended sample" toggle at the top of the page
(including the live paper trader, which finds and evaluates whichever family's
currently-open market) — see `reports/research_memo.md` Part 4 for the full
writeup — short version: on the most recent ~2,000 resolved hourly markets,
the edge shows up more broadly across horizons (not concentrated in the least-
informative one, unlike the daily result), which is corroborating evidence, but
it comes from a different, more recent time window than the daily sample so it
isn't a strict independent replication on the same period.

## Known limitations

- **GDELT reachability was inconsistent from this project's sandbox.** Early in this
  session, repeated manual requests (spaced 20-45s apart, well over GDELT's stated
  5s minimum) all returned HTTP 429 — consistent with a shared/NAT'd egress IP whose
  quota other tenants had already exhausted. Later in the session, individual
  requests succeeded (used by the live paper trader). Given the 507-market dataset
  needs one query per market minimum and requests were failing/slow unpredictably, a
  full historical backfill was not completed in the committed dataset — **every
  historical observation has `news_data_available=False` and NaN news features**,
  never fabricated or imputed. The adapter itself (`src/news_client.py`) is real,
  rate-limit-compliant code that works when GDELT is reachable, as the live paper
  trader demonstrates. A partial backfill attempt is cached under `data/raw/news/`.
- **`market_liquidity` is 0 for 100% of observations** (Gamma AMM-liquidity field,
  unused by CLOB order-book markets). Total volume substitutes as the depth proxy
  throughout — see Feature definitions above.
- **No historical order book or trades.** All historical execution is a documented
  cost *model*, not observed executable prices — see Execution assumptions.
- **No social-media data** (see Feature definitions) — the module is a real,
  functioning adapter that simply has no data to report in this run.
- **Repeated-market / correlated-risk exposure.** Every observation ultimately
  resolves against the same kind of BTC coin-flip; even with market-level bootstrap
  and one-position-per-market backtest logic, 507 markets over ~17.5 months is a
  fairly small, serially-dependent (BTC regime) sample — see `reports/research_memo.md`
  for an honest discussion of how much weight the headline backtest numbers deserve.
- **Backtest signal volume is high relative to the sample** (e.g. the full-feature
  strategy fills on ~86% of eligible test markets at `min_model_edge=0.03`), which is
  a flag for possible overconfidence/overfitting in that model rather than a
  demonstrated durable edge — discussed in the research memo.

## How to run

```bash
git clone https://github.com/Synpath-ai/bitcoin_calibration && cd bitcoin_calibration
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Full pipeline (real network calls, cached to data/raw/ on first run):
.venv/bin/python scripts/01_fetch_markets.py
.venv/bin/python scripts/02_fetch_btc.py
.venv/bin/python scripts/03_fetch_market_prices.py
.venv/bin/python scripts/04_build_observations.py          # add USE_LIVE_GDELT=1 to retry news
.venv/bin/python scripts/05_calibration.py
.venv/bin/python scripts/06_walkforward.py
.venv/bin/python scripts/07_backtest.py
.venv/bin/python scripts/08_live_paper_trade.py             # one-off live check-in (real data, paper only)
.venv/bin/python scripts/10_generate_reports.py             # regenerate data_quality_report.md from cached data

# Extended sample: hourly BTC Up/Down markets (replication check, ~1,999 markets):
.venv/bin/python scripts/11_fetch_hourly_markets.py
.venv/bin/python scripts/12_fetch_hourly_btc.py
.venv/bin/python scripts/13_fetch_hourly_prices.py
.venv/bin/python scripts/14_build_hourly_observations.py
.venv/bin/python scripts/15_hourly_calibration.py
.venv/bin/python scripts/16_hourly_walkforward.py
.venv/bin/python scripts/17_hourly_backtest.py

# Tests (fixtures only, no network):
.venv/bin/python -m pytest tests/ -q

# Dashboard:
.venv/bin/streamlit run dashboard/app.py
```

## Continuous live paper trading (no local machine required)

`.github/workflows/live_paper_trade.yml` runs both market families' live
check-ins on GitHub's runners and commits the results back to
`paper_trading/*.jsonl`, so the paper trader keeps accumulating a forward
(out-of-sample) track record without anything running locally.

**The timer is deliberately external to GitHub.** The workflow exposes only
`workflow_dispatch`; an external cron service (cron-job.org) POSTs to the REST
dispatch endpoint every 10 minutes:

```
POST https://api.github.com/repos/Synpath-ai/bitcoin_calibration/actions/workflows/live_paper_trade.yml/dispatches
Accept: application/vnd.github+json
Authorization: Bearer <fine-grained PAT — this repo only, Actions: Read and write>
X-GitHub-Api-Version: 2022-11-28

{"ref":"main","inputs":{"family":"both"}}
```

A successful dispatch returns **HTTP 204** with no body.

*Why not `on: schedule`?* It was tried first and does not work reliably here.
GitHub's docs warn the schedule event "can be delayed during periods of high
loads" and that "some queued jobs may be dropped" — in practice a 10-minute
cron produced only **6 runs over ~16 hours**, at intervals from 30 minutes to
6 hours. This was not a misconfiguration (10 min is well inside the documented
5-minute minimum; Actions was enabled at repo *and* org level; billing was
within the included allowance). `workflow_dispatch` fired reliably every time,
so only the timer moved off GitHub — the work still runs on GitHub's runners.

## Order-book depth and the absolute spread floor

Two execution assumptions were corrected after checking the model against live
order books. Both made the original backtest materially too optimistic.

**Spread is a flat ~1 cent, not proportional.** The first cost model scaled the
half-spread with the contract price, charging a 5c contract ~0.03c one-way
against the ~0.5c real books actually quote (Polymarket's own UI reports
"Spread: 1c" regardless of level). `ExecutionConfig.min_half_spread_abs` now
imposes a $0.005 per-side floor, and it is **not** relaxed by the optimistic
scenario — you cannot trade inside the quoted spread.

**Capacity is set by book depth, not by market volume.** Volume is a flow over a
market's whole life; depth is the stock resting on the book at the moment you
trade. Live sampling found only a few hundred dollars within a cent of the
touch, while the volume-based participation cap permitted ~$2,000 positions —
so the original backtest's typical trade was several times the entire visible
near-touch liquidity. The book is now modelled explicitly (`touch_depth_usd`
fillable at the touch plus `depth_per_cent_usd` per additional cent walked),
fills are priced at the walked VWAP, and orders that cannot fill within
`max_walk_cents` are recorded as **unfilled** rather than silently filled.

Effect on the headline result (`full_feature_strategy`, base scenario): daily
$35,003 → **$25,469**, hourly $157,957 → **$36,575**, with average position size
falling from ~$2,000 to ~$750. The edge remains positive in every scenario and
per-trade returns actually improve, but this is now clearly a **small-capacity**
strategy — roughly $19k of working capital, not $100k. See
`reports/research_memo.md` Part 5.

**The depth constants are the weakest input in the model.** Polymarket publishes
no historical order books, so they are anchored to a handful of live spot checks
— several taken on freshly seeded, zero-volume markets rather than the
minutes-to-resolution points where the strategy actually trades. The live paper
trader therefore records executable depth (`buy_depth_1c/2c/5c`,
`sell_depth_1c`, NO-leg equivalents, touch and spread) into
`paper_trading/mark_to_market*.jsonl` on every 10-minute check-in. Once enough
accumulates at trade-relevant horizons, these constants should be re-fit to
measured depth and the backtest re-run. Until then the scenario range, not the
base case, is the honest answer.

## Repository layout

```
src/                  core library (client wrappers, matching, features, models, backtest, risk, execution)
scripts/              numbered, idempotent pipeline stages (cache-aware — safe to re-run)
data/raw/              cached raw API responses (reproducibility)
data/processed/        derived datasets/artifacts consumed by the dashboard
dashboard/app.py       5-tab Streamlit app; every tab has a Daily/Hourly market-family toggle
paper_trading/         append-only logs from the live paper trader
tests/                 pytest suite, fixtures-only (no live network calls)
fixtures/              saved sample API payloads used by tests
reports/               research_memo.md, data_quality_report.md
```
