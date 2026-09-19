"""
Discover and validate Polymarket's recurring daily "Bitcoin Up or Down" markets.

Primary discovery path: Gamma's `series` object (slug `btc-up-or-down-daily`,
id 41 as of 2026-08) groups every daily BTC up/down event together, which is
the authoritative, non-hard-coded way to enumerate the market family.

Because a series link could in principle be missing/wrong for some historical
market, every candidate is *independently* re-validated against title/slug/
description/duration patterns before being accepted. Anything that fails
validation is excluded and the reason is recorded, satisfying the "robust
matching, do not hard-code a single market ID" requirement even though we
bootstrap from one known series slug.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

from . import polymarket_client as pc

DAILY_SERIES_SLUG = "btc-up-or-down-daily"

SLUG_PATTERN = re.compile(r"^bitcoin-up-or-down-on-[a-z]+-\d{1,2}(-\d{4})?(-noon)?$")
TITLE_PATTERN = re.compile(r"bitcoin up or down on", re.IGNORECASE)
RESOLUTION_KEYWORDS = ("binance", "close", "candle")

MIN_DURATION_HOURS = 20   # daily markets run roughly 1-4 days start->resolve
MAX_DURATION_HOURS = 96


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
    start_date: str
    end_date: str
    closed: bool
    active: bool
    outcomes: list[str]
    outcome_prices: list[str]
    clob_token_ids: list[str]
    volume: float
    liquidity: float
    resolution_source: str


def _safe_json_list(s) -> list:
    if isinstance(s, list):
        return s
    if not s:
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


def discover_daily_markets(closed: bool | None = True) -> tuple[list[CandidateMarket], list[ExclusionRecord]]:
    """Fetch every event in the daily BTC up/down series and validate each one.

    Returns (accepted_candidates, exclusions). This function only performs
    *structural* validation (slug/title/description/duration/outcome shape).
    Data-availability exclusions (missing price history, ambiguous outcome,
    etc.) happen later in the observation builder, which appends to the same
    data-quality report.
    """
    events = pc.get_all_events_for_series(DAILY_SERIES_SLUG, closed=closed)

    accepted: list[CandidateMarket] = []
    excluded: list[ExclusionRecord] = []

    for ev in events:
        slug = ev.get("slug", "")
        event_id = ev.get("id")
        markets = ev.get("markets") or []

        if len(markets) != 1:
            excluded.append(ExclusionRecord(slug, event_id, "unexpected_market_count",
                                             f"event has {len(markets)} markets, expected 1"))
            continue

        m = markets[0]
        title = ev.get("title") or m.get("question", "")
        description = (m.get("description") or "").strip()

        if not TITLE_PATTERN.search(title):
            excluded.append(ExclusionRecord(slug, event_id, "title_pattern_mismatch", title))
            continue

        if not SLUG_PATTERN.match(slug):
            excluded.append(ExclusionRecord(slug, event_id, "slug_pattern_mismatch", slug))
            continue

        desc_lower = description.lower()
        if not all(k in desc_lower for k in RESOLUTION_KEYWORDS):
            excluded.append(ExclusionRecord(slug, event_id, "resolution_rule_mismatch",
                                             "description missing expected Binance/close/candle language"))
            continue

        start_date = ev.get("startDate") or m.get("startDate")
        end_date = ev.get("endDate") or m.get("endDate")
        if not start_date or not end_date:
            excluded.append(ExclusionRecord(slug, event_id, "missing_timestamps",
                                             f"start={start_date} end={end_date}"))
            continue
        try:
            sd = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            ed = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError as exc:
            excluded.append(ExclusionRecord(slug, event_id, "unparseable_timestamps", str(exc)))
            continue
        if ed <= sd:
            excluded.append(ExclusionRecord(slug, event_id, "inconsistent_timestamps",
                                             f"end<=start: {start_date} -> {end_date}"))
            continue
        duration_h = (ed - sd).total_seconds() / 3600
        if not (MIN_DURATION_HOURS <= duration_h <= MAX_DURATION_HOURS):
            excluded.append(ExclusionRecord(slug, event_id, "duration_out_of_range",
                                             f"{duration_h:.1f}h"))
            continue

        outcomes = _safe_json_list(m.get("outcomes"))
        if [o.lower() for o in outcomes] != ["up", "down"]:
            excluded.append(ExclusionRecord(slug, event_id, "unexpected_outcomes", str(outcomes)))
            continue

        token_ids = _safe_json_list(m.get("clobTokenIds"))
        if len(token_ids) != 2:
            excluded.append(ExclusionRecord(slug, event_id, "missing_token_ids", str(token_ids)))
            continue

        cond_id = m.get("conditionId", "")
        if not cond_id:
            excluded.append(ExclusionRecord(slug, event_id, "missing_condition_id"))
            continue

        outcome_prices = _safe_json_list(m.get("outcomePrices"))

        accepted.append(CandidateMarket(
            event_id=str(event_id),
            market_id=str(m.get("id")),
            slug=slug,
            question=title,
            condition_id=cond_id,
            description=description,
            start_date=start_date,
            end_date=end_date,
            closed=bool(m.get("closed")),
            active=bool(m.get("active")),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            clob_token_ids=token_ids,
            volume=float(m.get("volumeNum") or m.get("volume") or 0),
            liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0),
            resolution_source=m.get("resolutionSource", ""),
        ))

    return accepted, excluded


def resolve_outcome(candidate: CandidateMarket) -> tuple[int | None, str]:
    """Map final outcomePrices -> {0,1} for the 'Up' outcome. Returns (outcome, reason_if_none)."""
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
