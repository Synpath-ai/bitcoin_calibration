"""
Build the fixed-horizon observation dataset from cached raw data.

For every accepted, resolved daily BTC market and every horizon in
HORIZONS_HOURS, construct one observation at `resolution_time - horizon`,
using ONLY information available at or before that timestamp (last-observed-
value-carried-forward with an explicit staleness tolerance; never a future
value). Observations that can't be safely constructed are dropped and the
reason is recorded in the data-quality report.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import news_client as nc
from . import social_client as sc

HORIZONS_HOURS = [24, 12, 6, 3, 1]

MAX_POLYMARKET_PRICE_STALENESS = pd.Timedelta(minutes=20)
MAX_BTC_STALENESS = pd.Timedelta(minutes=15)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
PRICE_DIR = ROOT / "data" / "raw" / "polymarket" / "price_series"


def _last_value_as_of(sorted_ts: np.ndarray, sorted_vals: np.ndarray, as_of_epoch: float,
                       max_staleness_sec: float) -> tuple[float | None, float | None]:
    """Last-observation-carried-forward lookup. Returns (value, staleness_seconds) or (None, None).
    `sorted_ts` must be ascending. Strictly uses index <= as_of (no future leakage)."""
    idx = np.searchsorted(sorted_ts, as_of_epoch, side="right") - 1
    if idx < 0:
        return None, None
    staleness = as_of_epoch - sorted_ts[idx]
    if staleness > max_staleness_sec:
        return None, None
    return float(sorted_vals[idx]), float(staleness)


def _load_market_price_series(slug: str, price_dir: Path = PRICE_DIR) -> pd.DataFrame | None:
    path = price_dir / f"{slug}.json"
    if not path.exists():
        return None
    hist = json.loads(path.read_text())
    if not hist:
        return None
    df = pd.DataFrame(hist).sort_values("t")
    return df


def _btc_features(btc: pd.DataFrame, ts_epoch: float,
                   max_btc_staleness_sec: float = MAX_BTC_STALENESS.total_seconds()) -> dict:
    """All features computed strictly from candles with open_time <= ts_epoch."""
    t_arr = btc["_epoch"].values
    close_arr = btc["close"].values
    vol_arr = btc["volume"].values

    idx = np.searchsorted(t_arr, ts_epoch, side="right") - 1
    out = {
        "btc_spot_price": np.nan, "btc_staleness_sec": np.nan,
        "btc_return_15m": np.nan, "btc_return_1h": np.nan, "btc_return_6h": np.nan,
        "btc_realized_vol_6h": np.nan, "btc_spot_volume_1h": np.nan,
    }
    if idx < 0:
        return out
    staleness = ts_epoch - t_arr[idx]
    if staleness > max_btc_staleness_sec:
        return out

    now_px = close_arr[idx]
    out["btc_spot_price"] = float(now_px)
    out["btc_staleness_sec"] = float(staleness)

    def ret_over(seconds):
        j = np.searchsorted(t_arr, ts_epoch - seconds, side="right") - 1
        if j < 0:
            return np.nan
        past_px = close_arr[j]
        if past_px <= 0:
            return np.nan
        return float(now_px / past_px - 1.0)

    out["btc_return_15m"] = ret_over(15 * 60)
    out["btc_return_1h"] = ret_over(60 * 60)
    out["btc_return_6h"] = ret_over(6 * 60 * 60)

    # realized vol: stdev of 5m log returns over trailing 6h, annualization-free (per-period)
    j0 = np.searchsorted(t_arr, ts_epoch - 6 * 3600, side="right") - 1
    if j0 >= 0 and idx - j0 >= 5:
        window_close = close_arr[max(j0, 0):idx + 1]
        log_ret = np.diff(np.log(window_close))
        out["btc_realized_vol_6h"] = float(np.std(log_ret, ddof=1)) if len(log_ret) > 1 else np.nan

    j1h = np.searchsorted(t_arr, ts_epoch - 3600, side="right") - 1
    if j1h >= 0:
        out["btc_spot_volume_1h"] = float(np.sum(vol_arr[max(j1h, 0):idx + 1]))

    return out


def build_observations(markets: list[dict], btc_df: pd.DataFrame,
                        use_live_gdelt: bool = True, news_query: str = "bitcoin",
                        horizons_hours: list[float] = HORIZONS_HOURS,
                        max_market_staleness: pd.Timedelta = MAX_POLYMARKET_PRICE_STALENESS,
                        max_btc_staleness: pd.Timedelta = MAX_BTC_STALENESS,
                        price_dir: Path = PRICE_DIR) -> tuple[pd.DataFrame, list[dict]]:
    """horizons_hours may be fractional (e.g. 0.25 = 15 minutes) for markets shorter
    than a day, such as the hourly BTC Up/Down family -- see scripts/14_build_hourly_observations.py."""
    btc = btc_df.copy()
    # dtype from parquet may be datetime64[ms, UTC] or [ns, UTC]; normalize to ns
    # before converting to int64 so epoch seconds are computed correctly either way.
    btc["_epoch"] = btc["timestamp"].astype("datetime64[ns, UTC]").astype("int64") / 1e9
    btc = btc.sort_values("_epoch").reset_index(drop=True)

    rows, exclusions = [], []

    for m in markets:
        slug = m["slug"]
        resolution_time = pd.Timestamp(m["end_date"])
        market_open = pd.Timestamp(m["start_date"])
        market_date = resolution_time.date().isoformat()
        outcome = m["outcome_up"]

        price_df = _load_market_price_series(slug, price_dir=price_dir)
        if price_df is None:
            exclusions.append({"slug": slug, "reason": "no_price_series_cached"})
            continue
        p_ts = price_df["t"].values.astype(float)
        p_val = price_df["p"].values.astype(float)

        for h in horizons_hours:
            obs_ts = resolution_time - pd.Timedelta(hours=h)
            if obs_ts < market_open:
                exclusions.append({"slug": slug, "horizon_hours": h, "reason": "before_market_open"})
                continue
            obs_epoch = obs_ts.timestamp()

            yes_prob, staleness = _last_value_as_of(p_ts, p_val, obs_epoch,
                                                      max_market_staleness.total_seconds())
            if yes_prob is None:
                exclusions.append({"slug": slug, "horizon_hours": h, "reason": "stale_or_missing_market_price"})
                continue

            btc_feat = _btc_features(btc, obs_epoch, max_btc_staleness_sec=max_btc_staleness.total_seconds())
            if math.isnan(btc_feat["btc_spot_price"]):
                exclusions.append({"slug": slug, "horizon_hours": h, "reason": "stale_or_missing_btc_price"})
                continue

            news_feat = nc.get_news_features(obs_ts, query=news_query, use_live_gdelt=use_live_gdelt)
            social_feat = sc.get_social_features(obs_ts)

            row = {
                "market_id": m["market_id"],
                "slug": slug,
                "condition_id": m["condition_id"],
                "market_date": market_date,
                "observation_timestamp": obs_ts.isoformat(),
                "horizon_hours": h,
                "yes_probability": yes_prob,
                "price_staleness_sec": staleness,
                "outcome_up": outcome,
                "market_volume": m["volume"],
                "market_liquidity": m["liquidity"],
                **btc_feat,
                **news_feat.as_dict(),
                **social_feat.as_dict(),
            }
            rows.append(row)

    obs_df = pd.DataFrame(rows)
    return obs_df, exclusions
