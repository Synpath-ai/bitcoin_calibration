"""
Settle the live paper trader's recommendations against real resolved outcomes.

The paper trader records a recommendation on every 10-minute check-in but never
settles them, so until now there was no realized out-of-sample P&L -- only a log
of intentions. This script closes that loop.

Two things matter for the result to mean anything:

1. **One position per market.** The trader re-evaluates the same market every 10
   minutes, so a single market can appear dozens of times in the log. Counting
   each as a separate trade would multiply one bet into many and wildly
   overstate both turnover and P&L. We keep the FIRST qualifying recommendation
   per market, matching the backtest's own one-position-per-market rule.
2. **Only settled markets count.** Markets still open have no outcome yet and are
   reported separately as open exposure, never as profit.

Entry price is the real best-ask the trader saw (`actual_book` tier), not a
modelled fill -- this is the one place in the project where execution is
observed rather than assumed.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import polymarket_client as pc

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper_trading"


def load_first_per_market(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        sig = r.get("signal") or {}
        if not sig or not r.get("paper_stake"):
            continue
        rows.append({
            "timestamp": r["timestamp"], "slug": r["market"]["slug"],
            "resolution_time": r["market"]["resolution_time"],
            "side": sig["side"], "entry_price": sig["estimated_buy_price"],
            "model_prob": sig["model_probability"], "net_edge": sig["net_expected_edge"],
            "stake": r["paper_stake"], "hours_to_res": r["market"]["hours_to_resolution"],
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("timestamp")
    return df.groupby("slug", as_index=False).first()   # first qualifying signal only


def resolved_outcome(slug: str) -> int | None:
    """Return 1 if 'Up' won, 0 if 'Down' won, None if not yet resolved/ambiguous."""
    try:
        markets = pc.get_market_by_slug(slug)
    except RuntimeError:
        return None
    for m in markets:
        if m.get("slug") != slug or not m.get("closed"):
            continue
        try:
            prices = json.loads(m.get("outcomePrices") or "[]")
            up, down = float(prices[0]), float(prices[1])
        except (ValueError, IndexError):
            return None
        if up >= 0.99 and down <= 0.01:
            return 1
        if down >= 0.99 and up <= 0.01:
            return 0
    return None


def settle(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, t in df.iterrows():
        outcome = resolved_outcome(t["slug"])
        if outcome is None:
            out.append({**t, "outcome_up": None, "won": None, "pnl": None, "settled": False})
            continue
        won = (outcome == 1) if t["side"] == "buy_yes" else (outcome == 0)
        contracts = t["stake"] / t["entry_price"]
        pnl = (contracts * 1.0 if won else 0.0) - t["stake"]
        out.append({**t, "outcome_up": outcome, "won": won, "pnl": pnl, "settled": True})
    return pd.DataFrame(out)


def report(name: str, df: pd.DataFrame):
    print(f"\n{'='*62}\n{name}\n{'='*62}")
    if df.empty:
        print("  no recommendations logged")
        return
    s = df[df.settled]
    open_ = df[~df.settled]
    print(f"  distinct markets traded : {len(df)}   (settled {len(s)}, still open {len(open_)})")
    if open_.empty is False and len(open_):
        print(f"  open exposure           : ${open_.stake.sum():,.0f} (excluded from P&L)")
    if s.empty:
        print("  nothing settled yet -- no realized P&L to report")
        return
    staked, pnl = s.stake.sum(), s.pnl.sum()
    print(f"  staked                  : ${staked:,.0f}")
    print(f"  realized P&L            : ${pnl:,.0f}   ({pnl/staked:+.2%} on stake)")
    print(f"  hit rate                : {s.won.mean():.1%}  ({int(s.won.sum())}/{len(s)})")
    print(f"  avg stake / entry price : ${s.stake.mean():,.0f} / {s.entry_price.mean():.3f}")
    print(f"  avg predicted edge      : {s.net_edge.mean():+.1%}")
    by = s.groupby(s.side).agg(n=("pnl", "size"), pnl=("pnl", "sum"), hit=("won", "mean"))
    print("\n  by side:")
    print("   ", by.round(2).to_string().replace("\n", "\n    "))


def main():
    all_settled = []
    for fam, fname in (("DAILY", "paper_trades.jsonl"), ("HOURLY", "paper_trades_hourly.jsonl")):
        path = PAPER / fname
        if not path.exists():
            continue
        df = settle(load_first_per_market(path))
        report(f"{fam}  ({fname})", df)
        if not df.empty:
            df["family"] = fam.lower()
            all_settled.append(df)

    if all_settled:
        combined = pd.concat(all_settled, ignore_index=True)
        combined.to_csv(PAPER / "settled_paper_trades.csv", index=False)
        s = combined[combined.settled]
        if len(s):
            print(f"\n{'='*62}\nCOMBINED (real out-of-sample, actual_book entry prices)\n{'='*62}")
            print(f"  settled trades : {len(s)}   staked ${s.stake.sum():,.0f}")
            print(f"  realized P&L   : ${s.pnl.sum():,.0f}  ({s.pnl.sum()/s.stake.sum():+.2%} on stake)")
            print(f"  hit rate       : {s.won.mean():.1%}")
        print(f"\nwrote {PAPER/'settled_paper_trades.csv'}")


if __name__ == "__main__":
    main()
