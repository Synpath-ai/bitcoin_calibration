"""
Discover and validate Polymarket's recurring HOURLY "Bitcoin Up or Down" markets
(series slug `btc-up-or-down-hourly`) -- a much higher-frequency sibling of the
daily series in market_matching.py, used to get a larger, still-independent
sample for testing whether the short-horizon edge found in the daily backtest
replicates.

Key difference from the daily family: Gamma's `endDate` field for this series is
NOT the true resolution time (it's a placeholder, constant across many markets on
the same calendar day -- verified empirically, see README). The true resolution
time -- the close of the specific 1-hour BTC/USDT candle named in the market's
slug/title (e.g. "...-may-28-1-am-et-candle" -> the 1AM-2AM ET candle on May 28)
-- is parsed from the slug with an America/New_York (DST-aware) timezone lookup,
then cross-checked against `umaEndDate` (which is real, but includes a ~2h UMA
dispute-window buffer after the true candle close) as a sanity check.
"""
from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import polymarket_client as pc

HOURLY_SERIES_SLUG = "btc-up-or-down-hourly"
ET = ZoneInfo("America/New_York")

MONTH_NAMES = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
# Polymarket has used at least 3 slug formats for this series over time:
#   bitcoin-up-or-down-may-28-1-am-et-candle        (May 2025: dashed hour-ampm + "-candle")
#   bitcoin-up-or-down-june-6-12-pm-et               (Jun 2025: dashed hour-ampm, no "-candle")
#   bitcoin-up-or-down-june-23-8am-et                (late Jun 2025+: glued hour+ampm)
#   bitcoin-up-or-down-august-26-2026-10am-et        (2026+: year inserted before the hour)
# One regex with optional groups covers all four.
SLUG_RE = re.compile(
    r"^bitcoin-up-or-down-(?P<month>[a-z]+)-(?P<day>\d{1,2})"
    r"(?:-(?P<year>\d{4}))?"
    r"-(?P<hour>\d{1,2})-?(?P<ampm>am|pm)-et(?:-candle)?$")

RESOLUTION_KEYWORDS = ("binance", "candle")
# true resolution time must precede umaEndDate by this much slack (the observed
# UMA dispute-window buffer is ~2h; allow 0-6h to be tolerant of jitter)
UMA_BUFFER_MIN = pd.Timedelta(minutes=0)
UMA_BUFFER_MAX = pd.Timedelta(hours=6)


@dataclass
class ExclusionRecord:
    slug: str
    event_id: str | None
    reason: str
    detail: str = ""


@dataclass
class CandidateMarket:
    event_id: str
    market_id: str
    slug: str
    question: str
    condition_id: str
    description: str
    start_date: str          # market open (Gamma startDate, reliable)
    end_date: str             # PARSED true candle-close time (ISO), not Gamma's endDate
    closed: bool
    active: bool
    outcomes: list
    outcome_prices: list
    clob_token_ids: list
    volume: float
    liquidity: float
    resolution_source: str


def _safe_json_list(s):
    if isinstance(s, list):
        return s
    if not s:
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


def parse_candle_resolution_time(slug: str, creation_date: str) -> datetime | None:
    """Parse the true candle-close (resolution) UTC datetime from the slug, using
    creation_date only to disambiguate the year (not present in the slug)."""
    m = SLUG_RE.match(slug)
    if not m:
        return None
    month = MONTH_NAMES.get(m.group("month"))
    if not month:
        return None
    day = int(m.group("day"))
    hour_12 = int(m.group("hour"))
    ampm = m.group("ampm")
    if not (1 <= hour_12 <= 12):
        return None
    hour_24 = (hour_12 % 12) + (12 if ampm == "pm" else 0)  # 12am->0, 12pm->12

    created = pd.Timestamp(creation_date)
    slug_year = m.group("year")
    year = int(slug_year) if slug_year else created.year
    try:
        candle_start_et = datetime(year, month, day, hour_24, 0, 0, tzinfo=ET)
    except ValueError:
        return None
    candle_start_utc = pd.Timestamp(candle_start_et).tz_convert("UTC")

    # markets are created ~a few minutes to ~1 day before their candle starts;
    # if the parsed date lands >3 days before creation, it must really be next
    # year (December-created market for a January 1AM candle, etc.)
    if candle_start_utc < created - pd.Timedelta(days=3):
        candle_start_et = datetime(year + 1, month, day, hour_24, 0, 0, tzinfo=ET)
        candle_start_utc = pd.Timestamp(candle_start_et).tz_convert("UTC")

    candle_close_utc = candle_start_utc + pd.Timedelta(hours=1)
    return candle_close_utc


def discover_hourly_markets(max_events: int = 2100, ascending: bool = False,
                             closed: bool | None = True) -> tuple[list[CandidateMarket], list[ExclusionRecord]]:
    """Fetch up to `max_events` hourly BTC Up/Down events (offset pagination, capped
    at ~2100 by the API -- see polymarket_client) and validate each one. Default
    `ascending=False` fetches the MOST RECENT events first, which is what we want
    for a large-but-recent extended sample."""
    series_id = pc.resolve_series_id(HOURLY_SERIES_SLUG)
    events, offset, page_size = [], 0, 100
    while len(events) < max_events:
        page = pc.get_events_page(series_id, limit=page_size, offset=offset,
                                   closed=closed, ascending=ascending)
        if not page:
            break
        events.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    events = events[:max_events]

    accepted, excluded = [], []
    for ev in events:
        slug = ev.get("slug", "")
        event_id = ev.get("id")
        markets = ev.get("markets") or []
        if len(markets) != 1:
            excluded.append(ExclusionRecord(slug, event_id, "unexpected_market_count", str(len(markets))))
            continue
        m = markets[0]
        description = (m.get("description") or "").lower()
        if not all(k in description for k in RESOLUTION_KEYWORDS):
            excluded.append(ExclusionRecord(slug, event_id, "resolution_rule_mismatch"))
            continue

        resolution_time = parse_candle_resolution_time(slug, m.get("createdAt") or ev.get("creationDate"))
        if resolution_time is None:
            excluded.append(ExclusionRecord(slug, event_id, "slug_time_unparseable", slug))
            continue

        uma_end = m.get("umaEndDate")
        if uma_end:
            try:
                uma_ts = pd.Timestamp(uma_end)
            except ValueError:
                # a small number of records have a malformed umaEndDate string
                # (e.g. a duplicated UTC offset) -- skip the cross-check for those
                # rather than losing the market entirely.
                uma_ts = None
            if uma_ts is not None:
                slack = uma_ts - resolution_time
                if not (UMA_BUFFER_MIN <= slack <= UMA_BUFFER_MAX):
                    excluded.append(ExclusionRecord(slug, event_id, "resolution_time_sanity_check_failed",
                                                     f"parsed={resolution_time} umaEndDate={uma_ts} slack={slack}"))
                    continue

        start_date = m.get("startDate") or ev.get("startDate")
        if not start_date:
            excluded.append(ExclusionRecord(slug, event_id, "missing_start_date"))
            continue
        start_ts = pd.Timestamp(start_date)
        if resolution_time <= start_ts:
            excluded.append(ExclusionRecord(slug, event_id, "inconsistent_timestamps",
                                             f"resolution {resolution_time} <= start {start_ts}"))
            continue
        duration_min = (resolution_time - start_ts).total_seconds() / 60
        # Polymarket batch-creates these markets, sometimes ~2 days ahead of their
        # specific candle (observed up to ~49h lead time) -- the umaEndDate sanity
        # check above is the real correctness filter; this just catches broken
        # timestamps (negative/zero duration or multi-week outliers).
        if not (1 <= duration_min <= 96 * 60):
            excluded.append(ExclusionRecord(slug, event_id, "duration_out_of_range", f"{duration_min:.1f}min"))
            continue

        outcomes = _safe_json_list(m.get("outcomes"))
        if [o.lower() for o in outcomes] != ["up", "down"]:
            excluded.append(ExclusionRecord(slug, event_id, "unexpected_outcomes", str(outcomes)))
            continue
        token_ids = _safe_json_list(m.get("clobTokenIds"))
        if len(token_ids) != 2:
            excluded.append(ExclusionRecord(slug, event_id, "missing_token_ids"))
            continue
        cond_id = m.get("conditionId", "")
        if not cond_id:
            excluded.append(ExclusionRecord(slug, event_id, "missing_condition_id"))
            continue

        accepted.append(CandidateMarket(
            event_id=str(event_id), market_id=str(m.get("id")), slug=slug,
            question=ev.get("title") or m.get("question", ""), condition_id=cond_id,
            description=m.get("description") or "", start_date=start_date,
            end_date=resolution_time.isoformat(),
            closed=bool(m.get("closed")), active=bool(m.get("active")),
            outcomes=outcomes, outcome_prices=_safe_json_list(m.get("outcomePrices")),
            clob_token_ids=token_ids, volume=float(m.get("volumeNum") or m.get("volume") or 0),
            liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0),
            resolution_source=m.get("resolutionSource", ""),
        ))

    return accepted, excluded


def resolve_outcome(candidate: CandidateMarket) -> tuple[int | None, str]:
    if not candidate.closed:
        return None, "market_not_resolved"
    prices = candidate.outcome_prices
    if len(prices) != 2:
        return None, "outcome_prices_missing"
    try:
        up_p, down_p = float(prices[0]), float(prices[1])
    except ValueError:
        return None, "outcome_prices_unparseable"
    if up_p >= 0.99 and down_p <= 0.01:
        return 1, ""
    if down_p >= 0.99 and up_p <= 0.01:
        return 0, ""
    return None, f"ambiguous_final_prices up={up_p} down={down_p}"
