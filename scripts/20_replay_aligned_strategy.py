"""
Replay the ALIGNED strategy over the live mark-to-market log.

The live trader's own entries (before 2026-09-08) used the wrong timing and
sizing, so their P&L says nothing about the backtest's strategy. But it checked
in every 10 minutes and recorded the model probability, the book, and the market
volume each time -- which means the check-ins that happened to land on a backtest
horizon contain everything needed to reconstruct what the aligned strategy would
have done, without waiting to accumulate new data.

This is a replay of a rule over recorded market states, not a fresh live test:
prices, depths and model probabilities are all as observed at the time, but the
decision to trade is applied after the fact. It is still out-of-sample in the
sense that matters -- none of these markets existed when the model was fitted --
but it is not proof the live system would have executed identically.

Rules mirrored from src/backtest.py:
  - evaluate only at the backtest horizons, earliest first
  - at most one position per market, taken at the first horizon that qualifies
  - buy YES or NO, whichever has more edge, long only
  - require net edge > min_model_edge
  - size = min(fractional Kelly, observed book depth, participation cap,
    per-market risk cap, remaining capital, max trade size)
  - hold to resolution, settle at 1 or 0

The NO leg's ask is reconstructed as (1 - YES best bid): the two legs of a binary
market quote against each other, so the DOWN ask sits opposite the UP bid. Only
the UP book was logged.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import paper_trader as pt
from src import risk as rk

PAPER = Path(__file__).resolve().parent.parent / "paper_trading"


def load_horizon_snapshots() -> pd.DataFrame:
    rows = []
    for p in PAPER.glob("mark_to_market*.jsonl"):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("status") == "no_active_market":
                continue
            if d.get("calibrated_fair_value") is None or not d.get("best_ask"):
                continue
            h = d.get("hours_to_resolution")
            if h is None:
                continue
            horizon = pt.nearest_backtest_horizon(d["family"], h)
            if horizon is None:
                continue          # backtest would not have looked here
            rows.append({
                "timestamp": d["timestamp"], "slug": d["slug"], "family": d["family"],
                "horizon": horizon, "hours_to_res": h,
                "model_prob": d["calibrated_fair_value"],
                "best_bid": d.get("best_bid"), "best_ask": d["best_ask"],
                # Both legs capped at 2c depth, not 5c: the NO leg only ever had
                # 1c/2c logged (no_buy_depth_5c does not exist in the data), so using
                # YES's real 5c figure against NO's 2c would compare the two legs on
                # different scales and bias sizing toward YES. 2c is the widest band
                # actually measured for both sides.
                "buy_depth_2c": d.get("buy_depth_2c") or 0.0,
                "no_buy_depth_2c": d.get("no_buy_depth_2c") or 0.0,
                "volume": d.get("market_volume") or 0.0,
            })
    return pd.DataFrame(rows)


def replay(df: pd.DataFrame, family: str, cfg: rk.RiskConfig) -> pd.DataFrame:
    x = df[df.family == family].copy()
    if x.empty:
        return pd.DataFrame()
    x["timestamp"] = pd.to_datetime(x.timestamp, utc=True)
    x["resolves_at"] = x.timestamp + pd.to_timedelta(x.hours_to_res, unit="h")

    pf = rk.PortfolioState(cfg)
    opened, trades = {}, []

    # chronological, so capital and position limits bind the way they would live
    for _, r in x.sort_values("timestamp").iterrows():
        # settle anything that resolved before this moment
        for slug in [s for s, p in pf.open_positions.items() if p["resolves_at"] <= r.timestamp]:
            outcome = pt._resolved_outcome(slug, family)
            if outcome is None:
                continue
            trades.append(pf.settle_position(slug, outcome, 0.0))

        if r.slug in opened or not pf.can_open_new_position():
            continue

        yes_edge = r.model_prob - r.best_ask
        no_ask = 1 - r.best_bid if r.best_bid else None      # opposite leg of the same book
        no_edge = (1 - r.model_prob) - no_ask if no_ask else -1

        if yes_edge >= no_edge:
            side, price, edge, depth = "buy_yes", r.best_ask, yes_edge, r.buy_depth_2c
        else:
            side, price, edge, depth = "buy_no", no_ask, no_edge, r.no_buy_depth_2c
        if edge <= cfg.min_model_edge or not price or price <= 0:
            continue

        stake = pf.size_position(r.model_prob if side == "buy_yes" else 1 - r.model_prob,
                                  price, r.volume, 0.0, max_depth_usd=depth or None)
        if stake < 1.0:
            continue

        pf.open_position(r.slug, side, stake, price, stake / price,
                          r.timestamp, r.resolves_at,
                          meta={"horizon": r.horizon, "edge": edge})
        opened[r.slug] = True

    # settle whatever is left and resolvable
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
    snaps = load_horizon_snapshots()
    print(f"Check-ins landing on a backtest horizon: {len(snaps)} "
          f"across {snaps.slug.nunique()} markets")
    print(f"  (the full log has ~3,700 check-ins; the rest fall between horizons "
          f"and the backtest would not have evaluated them)\n")

    cfg = rk.RiskConfig()
    allt = []
    for fam in ("daily", "hourly"):
        t = replay(snaps, fam, cfg)
        print(f"{'='*60}\n{fam.upper()}\n{'='*60}")
        if t.empty:
            print("  no qualifying entries")
            continue
        staked, pnl = t.stake.sum(), t.pnl_net.sum()
        print(f"  trades   : {len(t)}   staked ${staked:,.0f}")
        print(f"  P&L      : ${pnl:,.0f}  ({pnl/staked:+.2%} on stake)")
        print(f"  hit rate : {t.won.mean():.1%}")
        print(f"  by horizon:")
        by = t.groupby("horizon").agg(n=("pnl_net", "size"), pnl=("pnl_net", "sum"), hit=("won", "mean"))
        print("   ", by.round(2).to_string().replace("\n", "\n    "))
        allt.append(t)

    if allt:
        c = pd.concat(allt, ignore_index=True)
        c.to_csv(PAPER / "aligned_replay_trades.csv", index=False)
        print(f"\n{'='*60}\nCOMBINED: {len(c)} trades, staked ${c.stake.sum():,.0f}, "
              f"P&L ${c.pnl_net.sum():,.0f} ({c.pnl_net.sum()/c.stake.sum():+.2%})")
        print(f"wrote {PAPER/'aligned_replay_trades.csv'}")


if __name__ == "__main__":
    main()
