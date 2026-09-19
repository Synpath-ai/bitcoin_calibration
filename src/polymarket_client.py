"""
Thin, well-behaved client for Polymarket's public APIs.

Endpoints used (public, no API key required for reads):
  - https://gamma-api.polymarket.com   -> market/event discovery & metadata
  - https://clob.polymarket.com        -> historical price series, live order book

All raw responses that feed the research dataset are cached to disk under
data/raw/ so the project is reproducible without re-hitting the network.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
POLY_DIR = RAW_DIR / "polymarket"

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "btc-calibration-research/0.1"})

DEFAULT_TIMEOUT = 20
MAX_RETRIES = 4
BACKOFF_BASE = 1.5


def _cache_key(url: str, params: dict) -> str:
    raw = url + "?" + json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _get(url: str, params: dict | None = None, cache_subdir: str | None = None,
          use_cache: bool = True) -> Any:
    """GET with retries/backoff, optionally caching the raw JSON to disk.

    `use_cache=False` is required for any query whose answer changes over time
    (e.g. "which markets are open right now"). Caching those returns a snapshot
    of whenever it was first fetched, which is silently wrong rather than merely
    stale -- see get_events_page."""
    params = params or {}
    cache_path = None
    if cache_subdir and use_cache:
        cdir = POLY_DIR / cache_subdir
        cdir.mkdir(parents=True, exist_ok=True)
        cache_path = cdir / f"{_cache_key(url, params)}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text())

    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = _SESSION.get(url, params=params, timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(BACKOFF_BASE ** attempt * 2)
                continue
            resp.raise_for_status()
            data = resp.json()
            if cache_path is not None:
                cache_path.write_text(json.dumps(data))
            return data
        except (requests.RequestException, json.JSONDecodeError) as exc:
            last_exc = exc
            time.sleep(BACKOFF_BASE ** attempt)
    raise RuntimeError(f"GET failed after {MAX_RETRIES} retries: {url} {params}") from last_exc


# ---------------------------------------------------------------------------
# Gamma API (discovery / metadata)
# ---------------------------------------------------------------------------

def resolve_series_id(series_slug: str) -> str:
    """`series` is NOT a valid filter on /events (silently ignored, returns unfiltered
    results). The only reliable filter is the numeric `series_id`, so we resolve the
    slug -> id dynamically via /series rather than hard-coding the id."""
    rows = _get(f"{GAMMA_BASE}/series", {"slug": series_slug}, cache_subdir="series")
    for row in rows:
        if row.get("slug") == series_slug:
            return str(row["id"])
    raise RuntimeError(f"series slug not found: {series_slug}")


def get_events_page(series_id: str, limit: int = 100, offset: int = 0,
                     closed: bool | None = None, ascending: bool = True) -> list[dict]:
    """Note on caching: results are cached ONLY for `closed=True`. Resolved markets
    are immutable, so replaying them from disk is exactly what reproducibility
    wants. `closed=False` asks "what is open right now", whose answer changes by
    the hour -- caching it made the live paper trader replay a days-old snapshot
    and report "no active market" on 95% of check-ins while markets were in fact
    open the whole time."""
    params = {
        "series_id": series_id,
        "limit": limit,
        "offset": offset,
        "order": "startDate",
        "ascending": str(ascending).lower(),
    }
    if closed is not None:
        params["closed"] = str(closed).lower()
    return _get(f"{GAMMA_BASE}/events", params, cache_subdir="events",
                use_cache=(closed is True))


def get_all_events_for_series(series_slug: str, closed: bool | None = None) -> list[dict]:
    """Paginate through every event in a Gamma series (100/page hard cap)."""
    series_id = resolve_series_id(series_slug)
    out, offset, page_size = [], 0, 100
    while True:
        page = get_events_page(series_id, limit=page_size, offset=offset, closed=closed)
        if not page:
            break
        out.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return out


def get_series_by_slug(slug: str) -> list[dict]:
    return _get(f"{GAMMA_BASE}/series", {"slug": slug}, cache_subdir="series")


def get_market_by_slug(slug: str) -> list[dict]:
    """NOT cached, deliberately. This is the query settlement (_resolved_outcome
    in paper_trader.py) uses to check whether a market has resolved yet. The
    `closed=true` filter means an in-flight market returns []. Caching that --
    as this function did until it was traced to a real, already-manifesting bug
    on 2026-09-08 -- freezes the market as permanently unresolved: the very
    first empty result gets written to disk and every later call (including
    ones made after the market genuinely closes) returns that same stale [],
    so its trade silently drops out of every settlement and P&L figure with no
    error. The daily/hourly discovery caches (get_events_page) hit the same bug
    for growing lists and are fixed the same way -- cache only once the answer
    is actually immutable. Here that means: don't cache at all, since we cannot
    tell from the response alone that we've hit the moment it just resolved."""
    return _get(
        f"{GAMMA_BASE}/markets",
        {"slug": slug, "closed": "true", "order": "endDate", "ascending": "false"},
        cache_subdir="markets", use_cache=False,
    )


def get_market_by_condition_id(condition_id: str) -> list[dict]:
    return _get(f"{GAMMA_BASE}/markets", {"condition_ids": condition_id}, cache_subdir="markets")


def search_markets(query: str, limit_per_type: int = 20) -> dict:
    return _get(
        f"{GAMMA_BASE}/public-search",
        {"q": query, "limit_per_type": limit_per_type},
        cache_subdir="search",
    )


# ---------------------------------------------------------------------------
# CLOB API (price history / order book)
# ---------------------------------------------------------------------------

def get_price_history(token_id: str, start_ts: int, end_ts: int, fidelity: int = 10) -> list[dict]:
    """Return [{'t': unix_seconds, 'p': price}, ...] for a CLOB token id."""
    data = _get(
        f"{CLOB_BASE}/prices-history",
        {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity},
        cache_subdir="prices",
    )
    return data.get("history", [])


def get_order_book(token_id: str) -> dict:
    """Live order book snapshot. Not cached (point-in-time, used by the live paper trader)."""
    return _get(f"{CLOB_BASE}/book", {"token_id": token_id})
