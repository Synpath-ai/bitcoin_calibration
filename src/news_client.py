"""
Bitcoin news-intensity adapter (GDELT DOC 2.0, no API key required).

Reality check for this project run (documented, not hidden): GDELT's public
DOC API enforces a strict "1 request / 5 seconds" throttle *per source IP*.
From this project's network egress, every request returned HTTP 429 even
after >45s of spacing between calls (verified with several manual retries
before writing this module) -- consistent with a shared/NAT'd IP that other
tenants have already exhausted the quota on. That is an infrastructure
limitation of the sandbox this project was built in, not a code bug.

Per the project spec ("do not fabricate news observations"), this module:
  1. Implements a real, rate-limit-compliant GDELT client (`fetch_gdelt_window`)
     that works when run from an IP GDELT hasn't throttled.
  2. Implements a CSV adapter (`load_news_csv`) so a user can supply their own
     timestamped news counts/tone from any source (GDELT export, NewsAPI, a
     paid feed, ...) without changing any downstream code.
  3. Exposes `get_news_features(..)` which tries GDELT, falls back to a CSV
     if configured, and otherwise returns all-NaN features + an explicit
     `news_data_available=False` flag. Downstream code must treat that flag
     as authoritative and must never impute/guess values in its place.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "news"
RAW_DIR.mkdir(parents=True, exist_ok=True)

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "btc-calibration-research/0.1"})

MIN_REQUEST_GAP_SEC = 5.0
_last_request_ts = 0.0


@dataclass
class NewsFeatures:
    news_data_available: bool
    article_count_1h: float = float("nan")
    article_count_6h: float = float("nan")
    article_count_24h: float = float("nan")
    news_intensity_change: float = float("nan")   # count_24h(now) - count_24h(24h ago)
    avg_tone: float = float("nan")                # unavailable via DOC/artlist; see NOTE below
    n_distinct_sources_24h: float = float("nan")

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _throttle():
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < MIN_REQUEST_GAP_SEC:
        time.sleep(MIN_REQUEST_GAP_SEC - elapsed)
    _last_request_ts = time.time()


def fetch_gdelt_window(query: str, start: pd.Timestamp, end: pd.Timestamp,
                        max_records: int = 250, max_retries: int = 2) -> list[dict] | None:
    """Fetch GDELT article list for [start, end). Returns None (not []) on failure
    so callers can distinguish 'confirmed zero articles' from 'could not query'."""
    cache_key = f"{query}_{start.strftime('%Y%m%d%H%M')}_{end.strftime('%Y%m%d%H%M')}"
    cache_path = RAW_DIR / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    for attempt in range(max_retries):
        _throttle()
        try:
            resp = _SESSION.get(GDELT_DOC_URL, params=params, timeout=30)
        except requests.RequestException:
            continue
        if resp.status_code == 429:
            time.sleep(MIN_REQUEST_GAP_SEC * (attempt + 2))
            continue
        if resp.status_code != 200:
            continue
        try:
            data = resp.json()
        except ValueError:
            continue
        articles = data.get("articles", [])
        cache_path.write_text(json.dumps(articles))
        return articles
    return None


def load_news_csv(path: str | Path) -> pd.DataFrame:
    """Expected columns: timestamp (ISO8601, UTC), source (str), tone (float, optional)."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def _count_from_articles(articles: list[dict], as_of: pd.Timestamp, window: pd.Timedelta) -> tuple[int, int]:
    lo = as_of - window
    n, sources = 0, set()
    for a in articles:
        try:
            t = pd.Timestamp(a["seendate"]).tz_localize("UTC") if pd.Timestamp(a["seendate"]).tzinfo is None \
                else pd.Timestamp(a["seendate"])
        except (KeyError, ValueError):
            continue
        if lo <= t <= as_of:
            n += 1
            if a.get("domain"):
                sources.add(a["domain"])
    return n, len(sources)


def get_news_features(as_of: pd.Timestamp, query: str = "bitcoin",
                       csv_df: pd.DataFrame | None = None,
                       use_live_gdelt: bool = True) -> NewsFeatures:
    """Point-in-time news features using only articles seen at/before `as_of`."""
    if csv_df is not None:
        window24 = csv_df[(csv_df.timestamp <= as_of) & (csv_df.timestamp > as_of - pd.Timedelta(hours=24))]
        window6 = window24[window24.timestamp > as_of - pd.Timedelta(hours=6)]
        window1 = window6[window6.timestamp > as_of - pd.Timedelta(hours=1)]
        prior24 = csv_df[(csv_df.timestamp <= as_of - pd.Timedelta(hours=24)) &
                          (csv_df.timestamp > as_of - pd.Timedelta(hours=48))]
        return NewsFeatures(
            news_data_available=True,
            article_count_1h=len(window1),
            article_count_6h=len(window6),
            article_count_24h=len(window24),
            news_intensity_change=len(window24) - len(prior24),
            avg_tone=float(window24["tone"].mean()) if "tone" in window24 and len(window24) else float("nan"),
            n_distinct_sources_24h=window24["source"].nunique() if "source" in window24 and len(window24) else float("nan"),
        )

    if not use_live_gdelt:
        return NewsFeatures(news_data_available=False)

    articles = fetch_gdelt_window(query, as_of - pd.Timedelta(hours=48), as_of)
    if articles is None:
        return NewsFeatures(news_data_available=False)

    c1, _ = _count_from_articles(articles, as_of, pd.Timedelta(hours=1))
    c6, _ = _count_from_articles(articles, as_of, pd.Timedelta(hours=6))
    c24, src24 = _count_from_articles(articles, as_of, pd.Timedelta(hours=24))
    prior_articles_count = 0
    for a in articles:
        try:
            t = pd.Timestamp(a["seendate"])
            t = t.tz_localize("UTC") if t.tzinfo is None else t
        except (KeyError, ValueError):
            continue
        if as_of - pd.Timedelta(hours=48) <= t <= as_of - pd.Timedelta(hours=24):
            prior_articles_count += 1

    # NOTE: GDELT DOC 2.0 `artlist` mode does not return a per-article tone field
    # (that requires `mode=tonechart` or the raw GKG 2.0 tables, out of scope for
    # this adapter). avg_tone is therefore left NaN even when GDELT is reachable.
    return NewsFeatures(
        news_data_available=True,
        article_count_1h=c1,
        article_count_6h=c6,
        article_count_24h=c24,
        news_intensity_change=c24 - prior_articles_count,
        avg_tone=float("nan"),
        n_distinct_sources_24h=src24,
    )
