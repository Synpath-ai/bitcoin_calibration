"""
Best-effort historical GDELT backfill: ONE query per market (a 72h window
covering start_date-48h through end_date, wide enough for every horizon's
lookback), rather than one query per observation, to stay within GDELT's
request-rate constraints. Raw article lists are cached under
data/raw/news/ and can be consumed by src/news_client.fetch_gdelt_window's
cache (same cache-key scheme) or reprocessed locally.

This is opportunistic: GDELT was confirmed unreachable earlier in this
project's session (persistent 429) and confirmed reachable later (see
src/news_client.py docstring and reports/data_quality_report.md) -- rate
limiting from the shared sandbox egress IP appears to fluctuate. This script
simply does its best within a time budget and reports exactly how many
markets it could and could not cover; nothing is fabricated for the rest.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import news_client as nc

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
LOG_PATH = PROCESSED / "news_backfill_progress.json"


def main():
    markets = json.loads((PROCESSED / "markets_accepted.json").read_text())
    markets = sorted(markets, key=lambda m: m["start_date"])

    ok, failed = [], []
    start_time = time.time()
    for i, m in enumerate(markets):
        start = pd.Timestamp(m["start_date"]) - pd.Timedelta(hours=48)
        end = pd.Timestamp(m["end_date"])
        articles = nc.fetch_gdelt_window("bitcoin", start, end, max_records=250, max_retries=3)
        if articles is None:
            failed.append(m["slug"])
        else:
            ok.append(m["slug"])

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{i+1}/{len(markets)}] ok={len(ok)} failed={len(failed)} elapsed={elapsed:.0f}s", flush=True)
            LOG_PATH.write_text(json.dumps({"done": i + 1, "total": len(markets),
                                             "ok": len(ok), "failed": len(failed)}, indent=2))

    LOG_PATH.write_text(json.dumps({"done": len(markets), "total": len(markets),
                                     "ok": len(ok), "failed": len(failed),
                                     "failed_slugs": failed}, indent=2))
    print(f"DONE. ok={len(ok)} failed={len(failed)}")


if __name__ == "__main__":
    main()
