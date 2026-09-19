"""Build fixed-horizon observations for the extended hourly BTC Up/Down sample.

Horizons are scaled down to fit inside a ~1-hour market (45/30/15/5/2 minutes
before resolution, vs. 24/12/6/3/1 HOURS for the daily family), and staleness
tolerances are tightened accordingly (this market only lasts ~60 minutes total).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import observation_builder as ob

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
PRICE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "polymarket" / "price_series_hourly"

# minutes before resolution, expressed in hours (build_observations accepts fractional hours)
HORIZONS_MINUTES = [45, 30, 15, 5, 2]
HOURLY_HORIZONS_HOURS = [m / 60 for m in HORIZONS_MINUTES]

MAX_MARKET_STALENESS = pd.Timedelta(minutes=3)
MAX_BTC_STALENESS = pd.Timedelta(minutes=2)


def main():
    markets = json.loads((PROCESSED / "hourly_markets_accepted.json").read_text())
    btc = pd.read_parquet(PROCESSED / "btc_1m_hourly_window.parquet")

    use_live_gdelt = os.environ.get("USE_LIVE_GDELT", "0") == "1"
    print(f"Building hourly-market observations for {len(markets)} markets x "
          f"{HORIZONS_MINUTES} minute-horizons (use_live_gdelt={use_live_gdelt})...")

    obs_df, exclusions = ob.build_observations(
        markets, btc, use_live_gdelt=use_live_gdelt,
        horizons_hours=HOURLY_HORIZONS_HOURS,
        max_market_staleness=MAX_MARKET_STALENESS,
        max_btc_staleness=MAX_BTC_STALENESS,
        price_dir=PRICE_DIR,
    )

    print(f"Observations built: {len(obs_df)}")
    print(f"Observation-level exclusions: {len(exclusions)}")
    by_reason = {}
    for e in exclusions:
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
    print(json.dumps(by_reason, indent=2))

    if not obs_df.empty:
        # relabel horizon_hours (fractional) back to whole minutes for readability downstream
        obs_df["horizon_minutes"] = (obs_df["horizon_hours"] * 60).round().astype(int)

    obs_df.to_parquet(PROCESSED / "hourly_observations.parquet")
    (PROCESSED / "hourly_observation_exclusions.json").write_text(json.dumps(exclusions, indent=2, default=str))

    if not obs_df.empty:
        print("\nPer-horizon counts:")
        print(obs_df.groupby("horizon_minutes").size())
        print(f"\nDistinct markets represented: {obs_df['slug'].nunique()}")


if __name__ == "__main__":
    main()
