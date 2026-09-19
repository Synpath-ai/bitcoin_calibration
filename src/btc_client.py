"""
BTC/USDT spot OHLCV. Binance's public REST API is the primary source (no key
required, used for the whole historical dataset). Binance.com is known to
geo-block requests from US-based datacenter IPs (HTTP 451/connection resets)
-- including GitHub Actions' hosted runners -- which `fetch_and_cache` (used
by the training-data refresh pipeline) and the live paper trader's
`get_latest_btc_snapshot` both fall back to Coinbase's public candles API
(also no key required) for, when Binance is unreachable. The two exchanges'
BTC/USD prices track each other extremely tightly (highly liquid, arbitraged
market), so this is a reasonable proxy -- but it IS a different venue than
Binance, so every fallback prints a clear warning to the run's log rather than
silently swapping in a different exchange's candles with no trace. The live
snapshot additionally tags this in its result (`btc_snapshot["source"]`).

This fallback was added to `fetch_and_cache` after `refresh_training_data.yml`
ran on GitHub's hosted runners for the first time (2026-09-10) and failed
outright with HTTP 451 on the very first `Extend BTC OHLCV` step -- until
then the workflow had only been verified with `fetch_and_cache`'s
already-up-to-date short-circuit (`last_ts >= end`), which never actually
calls Binance, so the geo-block never surfaced in that earlier check.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "btc"
RAW_DIR.mkdir(parents=True, exist_ok=True)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "btc-calibration-research/0.1"})

COLUMNS = [
    "open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms",
    "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int,
                  max_retries: int = 5) -> pd.DataFrame:
    """Fetch all klines in [start_ms, end_ms), paginating in chunks of 1000."""
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor,
                  "endTime": end_ms, "limit": 1000}
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = _SESSION.get(BINANCE_KLINES, params=params, timeout=20)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                batch = resp.json()
                break
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(1.5 ** attempt)
        else:
            raise RuntimeError(f"Binance klines fetch failed at cursor={cursor}: {last_exc}") from last_exc

        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        if last_open <= cursor:
            break
        cursor = last_open + 1
        if len(batch) < 1000:
            break

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "n_trades"]]


COINBASE_GRANULARITY_SEC = {"1m": 60, "5m": 300}  # the only two intervals this project ever requests


def fetch_coinbase_klines(interval: str, start_ms: int, end_ms: int, max_retries: int = 3) -> pd.DataFrame:
    """Coinbase's public candles endpoint, paginated in <=300-candle chunks (its
    documented per-request cap). Returns the same column schema as fetch_klines(),
    with quote_volume/n_trades as NaN (Coinbase doesn't report them). `interval`
    must be one of COINBASE_GRANULARITY_SEC's keys."""
    granularity = COINBASE_GRANULARITY_SEC[interval]
    rows = []
    chunk_span_ms = 300 * granularity * 1000  # 300 candles per request
    cursor = start_ms
    while cursor < end_ms:
        chunk_end = min(cursor + chunk_span_ms, end_ms)
        params = {"start": pd.Timestamp(cursor, unit="ms", tz="UTC").isoformat(),
                  "end": pd.Timestamp(chunk_end, unit="ms", tz="UTC").isoformat(), "granularity": granularity}
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = _SESSION.get(COINBASE_CANDLES, params=params, timeout=20)
                resp.raise_for_status()
                batch = resp.json()
                break
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(1.5 ** attempt)
        else:
            raise RuntimeError(f"Coinbase candles fetch failed at cursor={cursor}: {last_exc}") from last_exc
        rows.extend(batch)  # each row: [time_s, low, high, open, close, volume]
        cursor = chunk_end

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume",
                                      "quote_volume", "n_trades"])
    df = pd.DataFrame(rows, columns=["time_s", "low", "high", "open", "close", "volume"]).drop_duplicates("time_s")
    df["timestamp"] = pd.to_datetime(df["time_s"], unit="s", utc=True)
    df["quote_volume"] = float("nan")
    df["n_trades"] = float("nan")
    return df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "n_trades"]].sort_values("timestamp")


def fetch_coinbase_klines_5m(start_ms: int, end_ms: int, max_retries: int = 3) -> pd.DataFrame:
    """Back-compat wrapper -- src/paper_trader.py's live snapshot only ever needs 5m."""
    return fetch_coinbase_klines("5m", start_ms, end_ms, max_retries)


def _fetch_klines_with_fallback(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """fetch_klines (Binance), falling back to Coinbase on failure -- see module
    docstring for why this exists. Only 1m/5m have a Coinbase mapping; any other
    interval re-raises the original Binance error rather than silently returning
    nothing."""
    try:
        return fetch_klines(symbol, interval, start_ms, end_ms)
    except RuntimeError as exc:
        if interval not in COINBASE_GRANULARITY_SEC:
            raise
        print(f"WARNING: Binance fetch failed ({exc}); falling back to Coinbase for "
              f"{interval} candles {pd.Timestamp(start_ms, unit='ms', tz='UTC')} -> "
              f"{pd.Timestamp(end_ms, unit='ms', tz='UTC')}. Coinbase is a different venue -- "
              f"see src/btc_client.py module docstring.")
        return fetch_coinbase_klines(interval, start_ms, end_ms)


def fetch_and_cache(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp,
                     cache_name: str) -> pd.DataFrame:
    """Cache candles on disk, extending forward on repeat calls rather than
    freezing at whatever `end` was the first time. A prior version returned the
    cached file unchanged whenever it existed, with no comparison to the
    requested `end` -- fine for a one-off historical pull, but it meant the
    project's live training data silently stopped updating on the day the cache
    file was first written and nothing since was ever fetched again, which
    training on a 12-day-stale snapshot was traced back to."""
    cache_path = RAW_DIR / f"{cache_name}.parquet"
    if not cache_path.exists():
        df = _fetch_klines_with_fallback(symbol, interval, int(start.timestamp() * 1000), int(end.timestamp() * 1000))
        df.to_parquet(cache_path)
        return df

    cached = pd.read_parquet(cache_path)
    if cached.empty:
        return cached
    last_ts = cached["timestamp"].max()
    if last_ts >= end:
        return cached   # already covers the requested range

    new_rows = _fetch_klines_with_fallback(symbol, interval, int(last_ts.timestamp() * 1000) + 1, int(end.timestamp() * 1000))
    if new_rows.empty:
        return cached
    combined = (pd.concat([cached, new_rows], ignore_index=True)
                  .drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True))
    combined.to_parquet(cache_path)
    return combined
