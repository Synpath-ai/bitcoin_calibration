"""
Re-score the 634 horizon-aligned live check-ins with a properly walk-forward-
trained model, instead of the stale one that was actually running -- AND price
entries at a real-book-walked VWAP instead of assuming the full stake fills at
the best ask.

Part 9's replay used `calibrated_fair_value` as it was logged at the time --
produced by a model trained on observations.parquet as it stood on 2026-08-26/27
and never refreshed (see the refresh_training_data workflow and its commit
message for how that was found and fixed). That means every entry decision in
that replay was made by a model that, as real time moved from Aug 28 to Sep 8,
grew from 2 to 12 days stale and drifted from mean-predicting 0.478 (vs the
market's 0.517) to a market-price-centred 0.506 (vs 0.502) once refreshed. Part
10 fixed just the model and left entries priced at best_ask; this also fixes the
entry price.

This script asks: with the SAME market states (real book, real price, real
volume -- none of that depends on the model) but a PROPERLY walk-forward-trained
model at each point in time, AND a fill price that reflects walking the real
recorded depth bands rather than assuming the whole stake clears at the top of
book, what would the aligned strategy have entered?

Point-in-time correctness:
  - BTC features are recomputed from the (now-extended) 5m/1m OHLCV parquets
    using observation_builder._btc_features, which only ever looks at candles
    at-or-before the timestamp it is asked about -- the same function the
    historical training pipeline uses, so this does not introduce a different
    kind of look-ahead than the rest of the project already tolerates.
  - The model trained for each snapshot uses ONLY markets whose end_date is
    strictly before the snapshot's own market's start_date, matching
    src/walkforward.py's rule exactly. Snapshots are grouped by that training
    pool (it only changes when a market resolves) so this refits a few dozen
    times, not 634.
  - `yes_probability` fed to the model is the market mid AT THAT TIMESTAMP
    (best_bid/best_ask as logged), not today's price.
  - Entry price is walked against the REAL recorded depth-within-band figures
    (buy_depth_1c/2c/5c, no_buy_depth_1c/2c) via walk_real_book(), not the
    modelled linear book execution.py uses for the historical backtest. Each
    band's contribution is priced at its midpoint since only the cumulative
    notional per band was logged, not each individual price level.

What this does NOT fix: market_volume/market_liquidity in the mark_to_market
log are the values AS OF WHEN THE MARKET WAS LOGGED (near its end, in most
rows), not as of the snapshot timestamp -- Polymarket's API does not expose a
historical volume series, which is the same limitation the historical
observation-builder documents. This affects the sizing cap, not the edge
calculation. And the NO leg only ever had 1c/2c bands logged (no_buy_depth_5c
does not exist in the data), so its walk stops at 2c -- a real asymmetry in
precision between the two legs, not a bug.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import models as mdl
from src import observation_builder as ob
from src import paper_trader as pt
from src import risk as rk

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
PAPER = Path(__file__).resolve().parent.parent / "paper_trading"


# walk_real_book now lives in src/paper_trader.py -- this script was where it
# was first written and validated (see that function's docstring for the
# before/after numbers), and it has since been ported into the production live
# entry path itself. Kept as a local alias so every call site below is
# unchanged, but there is now exactly one implementation, not two that could
# silently drift apart. Note this replay stops the NO leg's walk at 2c
# (no_buy_depth_5c is never logged, unlike production which sees the full
# symmetric depth straight from summarize_book) -- a precision asymmetry in
# this historical replay only, not in production.
walk_real_book = pt.walk_real_book


def load_snapshots_needing_rescoring() -> pd.DataFrame:
    rows = []
    for p in PAPER.glob("mark_to_market*.jsonl"):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("status") == "no_active_market" or not d.get("best_ask"):
                continue
            h = d.get("hours_to_resolution")
            if h is None:
                continue
            horizon = pt.nearest_backtest_horizon(d["family"], h)
            if horizon is None:
                continue
            rows.append({
                "timestamp": d["timestamp"], "slug": d["slug"], "family": d["family"],
                "horizon": horizon, "hours_to_res": h,
                "best_bid": d.get("best_bid"), "best_ask": d["best_ask"],
                "buy_depth_1c": d.get("buy_depth_1c") or 0.0,
                "buy_depth_2c": d.get("buy_depth_2c") or 0.0,
                "buy_depth_5c": d.get("buy_depth_5c") or 0.0,
                "no_buy_depth_1c": d.get("no_buy_depth_1c") or 0.0,
                "no_buy_depth_2c": d.get("no_buy_depth_2c") or 0.0,
                "volume": d.get("market_volume") or 0.0,
            })
    return pd.DataFrame(rows)


def load_btc(family: str) -> pd.DataFrame:
    name = "btc_5m.parquet" if family == "daily" else "btc_1m_hourly_window.parquet"
    df = pd.read_parquet(PROCESSED / name)
    df["_epoch"] = df["timestamp"].astype("datetime64[ns, UTC]").astype("int64") / 1e9
    return df.sort_values("_epoch").reset_index(drop=True)


def load_market_windows(family: str) -> pd.DataFrame:
    name = "markets_accepted.json" if family == "daily" else "hourly_markets_accepted.json"
    m = pd.DataFrame(json.loads((PROCESSED / name).read_text()))[["slug", "start_date", "end_date"]]
    m["start_date"] = pd.to_datetime(m["start_date"], utc=True, format="ISO8601")
    m["end_date"] = pd.to_datetime(m["end_date"], utc=True, format="ISO8601")
    return m


def load_obs(family: str) -> pd.DataFrame:
    name = "observations.parquet" if family == "daily" else "hourly_observations.parquet"
    return pd.read_parquet(PROCESSED / name)


def rescore_family(snaps: pd.DataFrame, family: str) -> pd.DataFrame:
    x = snaps[snaps.family == family].copy()
    if x.empty:
        return x
    x["timestamp"] = pd.to_datetime(x.timestamp, utc=True)
    btc = load_btc(family)
    windows = load_market_windows(family).set_index("slug")
    obs = load_obs(family)

    end_sorted = windows["end_date"].sort_values()
    x = x[x.slug.isin(windows.index)].copy()
    x["market_start"] = x.slug.map(windows["start_date"])

    fitted_model, pool_key = None, None
    fresh_probs = []
    for _, r in x.sort_values("timestamp").iterrows():
        k = int(end_sorted.searchsorted(r.market_start, side="left"))
        train_slugs = set(end_sorted.index[:k])
        key = frozenset(train_slugs)
        if key != pool_key or len(train_slugs) < 30:
            if len(train_slugs) < 30:
                fresh_probs.append(np.nan)
                pool_key = key
                continue
            train_df = obs[obs.slug.isin(train_slugs)]
            fitted_model = mdl.RegularizedLogisticModel().fit(train_df)
            pool_key = key

        epoch = r.timestamp.timestamp()
        btc_feat = ob._btc_features(btc, epoch)
        mid = (r.best_bid + r.best_ask) / 2 if r.best_bid else r.best_ask
        row = pd.DataFrame([{
            "yes_probability": mid, "horizon_hours": r.hours_to_res,
            "btc_return_15m": btc_feat["btc_return_15m"], "btc_return_1h": btc_feat["btc_return_1h"],
            "btc_return_6h": btc_feat["btc_return_6h"], "btc_realized_vol_6h": btc_feat["btc_realized_vol_6h"],
            "btc_spot_volume_1h": btc_feat["btc_spot_volume_1h"],
            "market_volume": r.volume, "market_liquidity": 0.0,
        }])
        try:
            fresh_probs.append(float(fitted_model.predict(row)[0]))
        except Exception:
            fresh_probs.append(np.nan)

    x["fresh_model_prob"] = fresh_probs
    return x.dropna(subset=["fresh_model_prob"])


def replay(x: pd.DataFrame, family: str, cfg: rk.RiskConfig) -> pd.DataFrame:
    """Same entry/sizing rule as scripts/20, but the fill price is walked against
    the real recorded depth bands (walk_real_book) instead of assumed to be the
    flat best_ask/best_bid the live trader (and script 20) actually used."""
    x = x.copy()
    x["resolves_at"] = x.timestamp + pd.to_timedelta(x.hours_to_res, unit="h")
    pf = rk.PortfolioState(cfg)
    opened, trades = {}, []

    for _, r in x.sort_values("timestamp").iterrows():
        for slug in [s for s, p in pf.open_positions.items() if p["resolves_at"] <= r.timestamp]:
            outcome = pt._resolved_outcome(slug, family)
            if outcome is not None:
                trades.append(pf.settle_position(slug, outcome, 0.0))
        if r.slug in opened or not pf.can_open_new_position():
            continue

        # 1) pick a side using the TOUCH price, same as the live system does when
        #    deciding whether a signal exists at all
        yes_edge_touch = r.fresh_model_prob - r.best_ask
        no_ask_touch = 1 - r.best_bid if r.best_bid else None
        no_edge_touch = (1 - r.fresh_model_prob) - no_ask_touch if no_ask_touch else -1
        if yes_edge_touch >= no_edge_touch:
            side, model_p, depth_1c, depth_2c, depth_5c = (
                "buy_yes", r.fresh_model_prob, r.buy_depth_1c, r.buy_depth_2c, r.buy_depth_5c)
        else:
            side, model_p = "buy_no", 1 - r.fresh_model_prob
            depth_1c, depth_2c, depth_5c = r.no_buy_depth_1c, r.no_buy_depth_2c, r.no_buy_depth_2c
        touch_price = r.best_ask if side == "buy_yes" else no_ask_touch
        touch_edge = yes_edge_touch if side == "buy_yes" else no_edge_touch
        if touch_edge <= cfg.min_model_edge or not touch_price or touch_price <= 0:
            continue

        # 2) size against the touch price (matches size_position's own contract)
        depth_cap = depth_5c if side == "buy_yes" else depth_2c
        stake = pf.size_position(model_p, touch_price, r.volume, 0.0, max_depth_usd=depth_cap or None)
        if stake < 1.0:
            continue

        # 3) NOW walk the real book for the chosen stake, and re-check the edge at
        #    the walked price -- a stake that clears the bar at the touch might not
        #    once the true fill cost is included.
        vwap = walk_real_book(stake, touch_price, depth_1c, depth_2c, depth_5c)
        if vwap is None:
            continue   # book cannot actually absorb this stake -- unfilled, not a free fill
        walked_edge = model_p - vwap
        if walked_edge <= cfg.min_model_edge:
            continue

        pf.open_position(r.slug, side, stake, vwap, stake / vwap, r.timestamp, r.resolves_at,
                          meta={"horizon": r.horizon, "edge": walked_edge, "touch_price": touch_price})
        opened[r.slug] = True

    for slug in list(pf.open_positions):
        outcome = pt._resolved_outcome(slug, family)
        if outcome is not None:
            trades.append(pf.settle_position(slug, outcome, 0.0))

    if not trades:
        return pd.DataFrame()
    t = pd.DataFrame([{k: v for k, v in tr.items() if k != "signal"} for tr in trades])
    t["horizon"] = [tr.get("horizon") for tr in trades]
    t["family"] = family
    return t


def main():
    snaps = load_snapshots_needing_rescoring()
    print(f"Re-scoring {len(snaps)} horizon-aligned check-ins with proper walk-forward "
          f"models AND real-book-walked entry prices...\n")

    cfg = rk.RiskConfig()
    all_trades = []
    for family in ("daily", "hourly"):
        x = rescore_family(snaps, family)
        print(f"[{family}] {len(x)}/{len(snaps[snaps.family==family])} snapshots scored "
              f"(rest fell before min_train_markets=30)")
        if not x.empty:
            print(f"   fresh model mean prob: {x.fresh_model_prob.mean():.3f}   "
                  f"market mean (mid): {((x.best_bid+x.best_ask)/2).mean():.3f}")

        t = replay(x, family, cfg)
        print(f"\n{'='*58}\n{family.upper()} -- rescored + VWAP-priced replay\n{'='*58}")
        if t.empty:
            print("  no qualifying entries")
            continue
        staked, pnl = t.stake.sum(), t.pnl_net.sum()
        print(f"  trades   : {len(t)}   staked ${staked:,.0f}")
        print(f"  P&L      : ${pnl:,.0f}  ({pnl/staked:+.2%} on stake, weighted)")
        print(f"  P&L      : {(t.pnl_net/t.stake).mean():+.2%} (equal-weighted mean per trade)")
        print(f"  hit rate : {t.won.mean():.1%}")
        print(f"  side split: {t.side.value_counts().to_dict()}")
        all_trades.append(t)

    if all_trades:
        c = pd.concat(all_trades, ignore_index=True)
        c.to_csv(PAPER / "rescored_vwap_replay_trades.csv", index=False)
        print(f"\n{'='*58}\nCOMBINED: {len(c)} trades, staked ${c.stake.sum():,.0f}, "
              f"P&L ${c.pnl_net.sum():,.0f} ({c.pnl_net.sum()/c.stake.sum():+.2%})")
        print(f"wrote {PAPER/'rescored_vwap_replay_trades.csv'}")


if __name__ == "__main__":
    main()
