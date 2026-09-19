"""Fetch BTC/USDT 5-minute OHLCV covering the full daily-market history + a buffer."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import btc_client as bc

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def main():
    rows = json.loads((PROCESSED / "markets_accepted.json").read_text())
    starts = [pd.Timestamp(r["start_date"]) for r in rows]
    ends = [pd.Timestamp(r["end_date"]) for r in rows]
    lo = min(starts) - pd.Timedelta(hours=30)   # buffer for pre-window momentum features
    hi = max(ends) + pd.Timedelta(hours=1)

    print(f"Fetching BTCUSDT 5m klines from {lo} to {hi}")
    df = bc.fetch_and_cache("BTCUSDT", "5m", lo, hi, cache_name="btcusdt_5m")
    print(f"Fetched {len(df)} candles")
    print(df.head(2))
    print(df.tail(2))
    df.to_parquet(PROCESSED / "btc_5m.parquet")


if __name__ == "__main__":
    main()
