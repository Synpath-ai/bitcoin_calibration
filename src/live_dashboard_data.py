"""
Real, settled P&L for the live paper trader -- every distinct market the live
system ever signaled on, first-qualifying-check-in only (matching the
one-position-per-market rule everywhere else in this project), settled against
the market's real resolved outcome wherever that outcome is known.

Two eras coexist in the log, and both are surfaced rather than blended away:
  - before 2026-09-08 13:12 (commit 391217b1), `signal.estimated_buy_price` is
    the flat best_ask the live trader actually assumed as its fill -- no
    `touch_price` key exists on those rows.
  - from that commit on, `signal.estimated_buy_price` is the REAL book-walked
    VWAP, and `signal.touch_price` records what the flat price would have
    been, for comparison.
`execution_pricing` tags which era a row belongs to so a dashboard can compare
them honestly instead of implying one uniform methodology across the whole
history.

Resolution status is cached to disk, but ONLY once it resolves to a definite
outcome -- caching a still-open market's "not resolved yet" answer is exactly
the bug fixed in polymarket_client.get_market_by_slug (see that function's
docstring); a market that has not settled must be re-checked every time this
module is asked about it, never frozen as "still open" forever.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import paper_trader as pt

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper_trading"
RESOLVED_CACHE_PATH = PAPER_DIR / "_resolved_outcome_cache.json"


def _load_resolved_cache() -> dict:
    if not RESOLVED_CACHE_PATH.exists():
        return {}
    return json.loads(RESOLVED_CACHE_PATH.read_text())


def _save_resolved_cache(cache: dict) -> None:
    RESOLVED_CACHE_PATH.write_text(json.dumps(cache, indent=2, sort_keys=True))


def resolved_outcome_cached(slug: str, family: str, cache: dict) -> int | None:
    """1 if Up won, 0 if Down won, None if not yet resolved. `cache` is mutated
    in place with newly-discovered (permanent) outcomes; still-unresolved
    markets are deliberately never written to it."""
    key = f"{family}:{slug}"
    if key in cache:
        return cache[key]
    outcome = pt._resolved_outcome(slug, family)
    if outcome is not None:
        cache[key] = outcome
    return outcome


def load_first_signal_per_market(path: Path, family: str) -> pd.DataFrame:
    """One row per distinct market: the first check-in where a real trade
    signal fired (paper_stake > 0), i.e. the one the live trader actually
    would have opened -- the same de-dup rule scripts/18 uses, so re-evaluating
    the same market on later 10-minute check-ins never counts as a second bet."""
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sig = r.get("signal") or {}
        if not sig or not r.get("paper_stake"):
            continue
        rows.append({
            "timestamp": r["timestamp"], "slug": r["market"]["slug"], "family": family,
            "resolution_time": r["market"]["resolution_time"],
            "side": sig["side"],
            "touch_price": sig.get("touch_price", sig["estimated_buy_price"]),
            "fill_price": sig["estimated_buy_price"],
            "execution_pricing": "walked_vwap" if "touch_price" in sig else "flat_touch",
            "model_prob": sig["model_probability"], "net_edge": sig["net_expected_edge"],
            "stake": r["paper_stake"], "hours_to_res": r["market"]["hours_to_resolution"],
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("timestamp")
    return df.groupby("slug", as_index=False).first()


def build_live_trade_log() -> pd.DataFrame:
    """Every distinct live-traded market across both families, settled against
    its real outcome where known. Columns: settled (bool), won, pnl_net
    (fee_bps=0, matching Polymarket's real fee schedule) for settled rows;
    outcome/won/pnl_net are NaN for markets still open."""
    cache = _load_resolved_cache()
    frames = []
    for family, fname in (("daily", "paper_trades.jsonl"), ("hourly", "paper_trades_hourly.jsonl")):
        df = load_first_signal_per_market(PAPER_DIR / fname, family)
        if df.empty:
            continue
        outcomes, won, pnl = [], [], []
        for _, t in df.iterrows():
            o = resolved_outcome_cached(t["slug"], family, cache)
            outcomes.append(o)
            if o is None:
                won.append(None)
                pnl.append(None)
                continue
            w = (o == 1) if t["side"] == "buy_yes" else (o == 0)
            contracts = t["stake"] / t["fill_price"]
            won.append(w)
            pnl.append((contracts * 1.0 if w else 0.0) - t["stake"])
        df["outcome_up"] = outcomes
        df["won"] = won
        df["pnl_net"] = pnl
        df["settled"] = df["won"].notna()
        frames.append(df)
    _save_resolved_cache(cache)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


RETRO_PATH = PAPER_DIR / "rescored_vwap_replay_trades.csv"


def build_combined_live_view() -> pd.DataFrame:
    """The project's single definition of 'live' P&L, per explicit direction:
    the 83 real trades from before the 2026-09-08 13:12 fix (commit 391217b1)
    ran on a stale, frozen-since-Aug-26 training set AND priced fills at the
    flat touch price -- both since-fixed bugs -- and are excluded entirely,
    not blended in. In their place, the 2026-09-02..09-08 window is
    represented by the retrospective counterfactual replay
    (scripts/21_rescore_live_with_fresh_model.py): same real recorded market
    states (book, depth, resolved outcomes) but re-priced through the CURRENT
    model architecture and real-book-walked VWAP entries -- i.e. what the
    live system's current code would have done. From 09-08 13:12 onward, the
    real live trades (already on that same current code) are used directly.
    So 'live' = 89 retrospective (09-02..09-08) + every real post-fix trade,
    NOT a mix of real-and-counterfactual for the same window, and never the
    83 pre-fix real trades."""
    retro = pd.read_csv(RETRO_PATH) if RETRO_PATH.exists() else pd.DataFrame()
    if not retro.empty:
        retro = retro.rename(columns={"buy_price": "fill_price", "opened_at": "timestamp"})
        retro["settled"] = True
        retro["source"] = "retrospective (09-02→09-08, current model + VWAP)"

    live = build_live_trade_log()
    post_fix = live[live.execution_pricing == "walked_vwap"].copy() if not live.empty else live
    if not post_fix.empty:
        post_fix["source"] = "live (post-fix, real)"

    cols = ["timestamp", "slug", "family", "side", "touch_price", "fill_price",
            "stake", "won", "pnl_net", "settled", "source"]
    frames = [df[cols] for df in (retro, post_fix) if not df.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, format="ISO8601")
    return combined.sort_values("timestamp").reset_index(drop=True)
