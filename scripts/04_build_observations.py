"""Build the fixed-horizon observation dataset from cached raw Polymarket + BTC data.

News features: GDELT's public DOC API was confirmed unreachable from this
environment (persistent HTTP 429 from a shared sandbox egress IP, verified
manually with >45s spacing between requests -- see src/news_client.py
docstring). Per the "do not fabricate news observations" requirement, this
run passes use_live_gdelt=False so news features are recorded as explicitly
unavailable (NaN + news_data_available=False) rather than guessed. The
adapter itself is real and will fetch live data if run from an unrestricted
IP; set USE_LIVE_GDELT=1 to try.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import observation_builder as ob

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def main():
    markets = json.loads((PROCESSED / "markets_accepted.json").read_text())
    btc = pd.read_parquet(PROCESSED / "btc_5m.parquet")

    use_live_gdelt = os.environ.get("USE_LIVE_GDELT", "0") == "1"
    print(f"Building observations for {len(markets)} markets x {ob.HORIZONS_HOURS} horizons "
          f"(use_live_gdelt={use_live_gdelt})...")

    obs_df, exclusions = ob.build_observations(markets, btc, use_live_gdelt=use_live_gdelt)

    print(f"Observations built: {len(obs_df)}")
    print(f"Observation-level exclusions: {len(exclusions)}")
    by_reason = {}
    for e in exclusions:
        by_reason[e["reason"]] = by_reason.get(e["reason"], 0) + 1
    print(json.dumps(by_reason, indent=2))

    obs_df.to_parquet(PROCESSED / "observations.parquet")
    (PROCESSED / "observation_exclusions.json").write_text(json.dumps(exclusions, indent=2))

    print("\nPer-horizon counts:")
    print(obs_df.groupby("horizon_hours").size())
    print(f"\nDistinct markets represented: {obs_df['slug'].nunique()}")
    print(f"news_data_available rate: {obs_df['news_data_available'].mean():.3f}")
    print(f"social_data_available rate: {obs_df['social_data_available'].mean():.3f}")


if __name__ == "__main__":
    main()
