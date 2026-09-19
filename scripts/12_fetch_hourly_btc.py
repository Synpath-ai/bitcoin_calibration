"""Fetch BTC/USDT 1-minute OHLCV covering the hourly-market date range (need finer
granularity than the daily pipeline's 5m bars since horizons go down to 2 minutes)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import btc_client as bc

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def main():
    rows = json.loads((PROCESSED / "hourly_markets_accepted.json").read_text())
    starts = [pd.Timestamp(r["start_date"]) for r in rows]
    ends = [pd.Timestamp(r["end_date"]) for r in rows]
    lo = min(starts) - pd.Timedelta(hours=2)
    hi = max(ends) + pd.Timedelta(minutes=5)

    print(f"Fetching BTCUSDT 1m klines from {lo} to {hi}")
    df = bc.fetch_and_cache("BTCUSDT", "1m", lo, hi, cache_name="btcusdt_1m_hourly_window")
    print(f"Fetched {len(df)} candles")
    df.to_parquet(PROCESSED / "btc_1m_hourly_window.parquet")


if __name__ == "__main__":
    main()
