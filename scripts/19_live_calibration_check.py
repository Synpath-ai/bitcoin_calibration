"""
Out-of-sample calibration check on the live mark-to-market log.

This is the most statistically powerful test the project has, and it is worth
understanding why it survives problems that invalidate other comparisons.

The live trader's *trading* was, until 2026-09-08, not the backtest's strategy:
it evaluated every 10 minutes instead of at fixed horizons, kept no portfolio
state, and applied no depth cap. That makes its realized P&L incomparable to the
backtest. But every check-in also recorded the model's probability and the
market's price for a market that later resolved, and *those* are untouched by
how the trader chose to size or time trades. So while only 80 trades settled,
1,858 probability estimates across 187 markets can be scored.

It asks the question the whole project rests on, directly: out of sample, is the
model's probability estimate more accurate than the market's own price? If it is
not, there is no edge for any entry rule to exploit.

Brier scores are bootstrapped by market, since ~6 check-ins on the same market
are one coin flip observed repeatedly, not six independent draws.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import calibration as cal
from src import paper_trader as pt

PAPER = Path(__file__).resolve().parent.parent / "paper_trading"


def load_observations() -> pd.DataFrame:
    rows = []
    for p in PAPER.glob("mark_to_market*.jsonl"):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("status") == "no_active_market" or d.get("calibrated_fair_value") is None:
                continue
            rows.append(d)
    return pd.DataFrame(rows)


def attach_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    outcomes = {}
    for slug, fam in df[["slug", "family"]].drop_duplicates().itertuples(index=False):
        outcomes[slug] = pt._resolved_outcome(slug, fam)
    df = df.copy()
    df["outcome"] = df.slug.map(outcomes)
    df = df[df.outcome.notna()].copy()
    df["outcome"] = df.outcome.astype(int)
    return df


def report(df: pd.DataFrame, family: str, min_markets: int = 30):
    x = df[df.family == family]
    if x.slug.nunique() < min_markets:
        print(f"\n[{family}] only {x.slug.nunique()} resolved markets -- too few to score, skipping")
        return
    print(f"\n[{family}] {len(x)} observations across {x.slug.nunique()} markets")
    for label, col in (("market price", "raw_probability"), ("model", "calibrated_fair_value")):
        pt_, lo, hi = cal.bootstrap_metric_by_market(x, col, "outcome", "slug", cal.brier_score, n_boot=500)
        intercept, slope = cal.calibration_intercept_slope(x[col].values, x.outcome.values)
        print(f"   {label:13s} Brier {pt_:.4f} [{lo:.4f}, {hi:.4f}]   slope {slope:.2f}  intercept {intercept:+.2f}")

    # Head to head, paired per observation then averaged per market so the test
    # is over independent markets rather than repeated check-ins.
    diff = (x.calibrated_fair_value - x.outcome) ** 2 - (x.raw_probability - x.outcome) ** 2
    per_market = diff.groupby(x.slug).mean()
    t, p = stats.ttest_1samp(per_market, 0)
    verdict = ("model better" if per_market.mean() < 0 and p < 0.05
               else "model worse" if per_market.mean() > 0 and p < 0.05
               else "no significant difference")
    print(f"   model Brier - market Brier = {per_market.mean():+.4f} (p={p:.3f}) -> {verdict}")


def main():
    df = attach_outcomes(load_observations())
    print(f"Scored {len(df)} live probability estimates across {df.slug.nunique()} resolved markets.")
    print("(Only 80 trades settled -- these estimates are ~23x more evidence, and are")
    print(" unaffected by the entry-timing and sizing divergences in the live trader.)")
    for fam in ("hourly", "daily"):
        report(df, fam)
    df.to_csv(PAPER / "live_calibration_observations.csv", index=False)
    print(f"\nwrote {PAPER / 'live_calibration_observations.csv'}")


if __name__ == "__main__":
    main()
