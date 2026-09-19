"""Fetch CLOB price history (Up token) for every accepted hourly BTC market.
Fidelity=1 (1-minute bins) since the whole market only lasts ~1 hour."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import polymarket_client as pc

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
PRICE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "polymarket" / "price_series_hourly"
PRICE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rows = json.loads((PROCESSED / "hourly_markets_accepted.json").read_text())
    print(f"Fetching price history for {len(rows)} hourly markets...")

    ok, missing = 0, []
    for i, r in enumerate(rows):
        slug = r["slug"]
        out_path = PRICE_DIR / f"{slug}.json"
        if out_path.exists():
            ok += 1
            continue
        up_token = r["clob_token_ids"][0]
        start_ts = int(pd.Timestamp(r["start_date"]).timestamp()) - 600
        end_ts = int(pd.Timestamp(r["end_date"]).timestamp()) + 600
        try:
            hist = pc.get_price_history(up_token, start_ts, end_ts, fidelity=1)
        except RuntimeError as exc:
            missing.append({"slug": slug, "reason": "price_history_fetch_failed", "detail": str(exc)})
            continue
        if not hist:
            missing.append({"slug": slug, "reason": "price_history_empty"})
            continue
        out_path.write_text(json.dumps(hist))
        ok += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} done", flush=True)

    print(f"Price history fetched/cached OK: {ok}")
    print(f"Missing: {len(missing)}")
    (PROCESSED / "hourly_price_history_missing.json").write_text(json.dumps(missing, indent=2))


if __name__ == "__main__":
    main()
