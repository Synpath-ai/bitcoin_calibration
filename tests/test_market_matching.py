import json
from pathlib import Path

from src import market_matching as mm
from src import polymarket_client as pc

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _load_fixture_events():
    return json.loads((FIXTURES / "sample_events.json").read_text())


def test_discover_daily_markets_uses_only_cached_fixture(monkeypatch):
    """No live API calls: get_all_events_for_series is monkeypatched to return
    the saved fixture."""
    monkeypatch.setattr(pc, "get_all_events_for_series", lambda *a, **kw: _load_fixture_events())
    accepted, excluded = mm.discover_daily_markets(closed=True)
    assert isinstance(accepted, list)
    assert isinstance(excluded, list)


def test_valid_daily_market_is_accepted(monkeypatch):
    monkeypatch.setattr(pc, "get_all_events_for_series", lambda *a, **kw: _load_fixture_events())
    accepted, _ = mm.discover_daily_markets(closed=True)
    slugs = {c.slug for c in accepted}
    assert "bitcoin-up-or-down-on-august-25" in slugs


def test_non_btc_updown_market_excluded_by_title(monkeypatch):
    monkeypatch.setattr(pc, "get_all_events_for_series", lambda *a, **kw: _load_fixture_events())
    accepted, excluded = mm.discover_daily_markets(closed=True)
    accepted_slugs = {c.slug for c in accepted}
    assert "will-bitcoin-hit-150k-in-2026" not in accepted_slugs
    reasons = {e.slug: e.reason for e in excluded}
    assert reasons["will-bitcoin-hit-150k-in-2026"] == "title_pattern_mismatch"


def test_out_of_range_duration_excluded(monkeypatch):
    monkeypatch.setattr(pc, "get_all_events_for_series", lambda *a, **kw: _load_fixture_events())
    _, excluded = mm.discover_daily_markets(closed=True)
    reasons = {e.slug: e.reason for e in excluded}
    assert reasons["bitcoin-up-or-down-on-may-27"] == "duration_out_of_range"


def test_slug_pattern_mismatch_excluded(monkeypatch):
    monkeypatch.setattr(pc, "get_all_events_for_series", lambda *a, **kw: _load_fixture_events())
    _, excluded = mm.discover_daily_markets(closed=True)
    reasons = {e.slug: e.reason for e in excluded}
    assert reasons["bitcoin-up-or-down-on-june-11-847"] == "slug_pattern_mismatch"


def test_outcome_mapping_up_and_down():
    up_market = mm.CandidateMarket(
        event_id="1", market_id="1", slug="s", question="q", condition_id="c",
        description="d", start_date="2025-01-01T00:00:00Z", end_date="2025-01-02T00:00:00Z",
        closed=True, active=True, outcomes=["Up", "Down"], outcome_prices=["1", "0"],
        clob_token_ids=["a", "b"], volume=1.0, liquidity=0.0, resolution_source="binance")
    down_market = mm.CandidateMarket(**{**up_market.__dict__, "outcome_prices": ["0", "1"]})
    outcome_up, reason = mm.resolve_outcome(up_market)
    assert outcome_up == 1 and reason == ""
    outcome_down, reason2 = mm.resolve_outcome(down_market)
    assert outcome_down == 0 and reason2 == ""


def test_ambiguous_outcome_excluded():
    ambiguous = mm.CandidateMarket(
        event_id="1", market_id="1", slug="s", question="q", condition_id="c",
        description="d", start_date="2025-01-01T00:00:00Z", end_date="2025-01-02T00:00:00Z",
        closed=True, active=True, outcomes=["Up", "Down"], outcome_prices=["0.5", "0.5"],
        clob_token_ids=["a", "b"], volume=1.0, liquidity=0.0, resolution_source="binance")
    outcome, reason = mm.resolve_outcome(ambiguous)
    assert outcome is None
    assert "ambiguous" in reason


def test_unresolved_market_excluded():
    open_market = mm.CandidateMarket(
        event_id="1", market_id="1", slug="s", question="q", condition_id="c",
        description="d", start_date="2025-01-01T00:00:00Z", end_date="2099-01-02T00:00:00Z",
        closed=False, active=True, outcomes=["Up", "Down"], outcome_prices=[],
        clob_token_ids=["a", "b"], volume=1.0, liquidity=0.0, resolution_source="binance")
    outcome, reason = mm.resolve_outcome(open_market)
    assert outcome is None and reason == "market_not_resolved"


def test_open_market_queries_are_not_cached(monkeypatch, tmp_path):
    """Regression: caching `closed=False` made the live paper trader replay a
    days-old snapshot of "which markets are open", reporting no active market on
    95% of check-ins while markets were open the whole time. Resolved markets
    (closed=True) are immutable and MUST still be cached for reproducibility."""
    calls = []

    def fake_get(url, params=None, cache_subdir=None, use_cache=True):
        calls.append({"closed": (params or {}).get("closed"), "use_cache": use_cache})
        return []

    monkeypatch.setattr(pc, "_get", fake_get)

    pc.get_events_page("41", closed=False)
    pc.get_events_page("41", closed=True)

    open_call = next(c for c in calls if c["closed"] == "false")
    resolved_call = next(c for c in calls if c["closed"] == "true")
    assert open_call["use_cache"] is False, "live/open-market queries must bypass the cache"
    assert resolved_call["use_cache"] is True, "resolved markets are immutable and should stay cached"


def test_market_by_slug_query_is_never_cached(monkeypatch):
    """Regression: get_market_by_slug used to cache unconditionally. The
    closed=true filter returns [] for an in-flight market, and caching that
    freezes it as permanently unresolved -- the first empty result gets written
    to disk and every later call, including ones made after the market actually
    resolves, returns that same stale []. This silently dropped a real open
    position out of settlement entirely on 2026-09-08 (verified: the market had
    genuinely resolved per a fresh, uncached API call, but the cached answer was
    still [] and the position sat past its resolution time unsettled)."""
    calls = []

    def fake_get(url, params=None, cache_subdir=None, use_cache=True):
        calls.append(use_cache)
        return []

    monkeypatch.setattr(pc, "_get", fake_get)
    pc.get_market_by_slug("some-slug")
    assert calls == [False]
