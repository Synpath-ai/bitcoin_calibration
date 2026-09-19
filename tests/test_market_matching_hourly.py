import pandas as pd

from src import market_matching_hourly as mmh


def test_parses_original_dashed_candle_format():
    # bitcoin-up-or-down-may-28-1-am-et-candle -> 1AM-2AM ET May 28 2025 candle
    resolution = mmh.parse_candle_resolution_time(
        "bitcoin-up-or-down-may-28-1-am-et-candle", "2025-05-27T15:14:31Z")
    assert resolution is not None
    # 1AM EDT May 28 = 05:00 UTC; candle CLOSES an hour later = 06:00 UTC
    assert resolution == pd.Timestamp("2025-05-28T06:00:00Z")


def test_parses_dashed_no_candle_suffix_format():
    resolution = mmh.parse_candle_resolution_time(
        "bitcoin-up-or-down-june-6-12-pm-et", "2025-06-04T15:34:33Z")
    assert resolution is not None
    # 12PM EDT June 6 = noon = 16:00 UTC; close = 17:00 UTC
    assert resolution == pd.Timestamp("2025-06-06T17:00:00Z")


def test_parses_glued_hour_ampm_format():
    resolution = mmh.parse_candle_resolution_time(
        "bitcoin-up-or-down-june-23-8am-et", "2025-06-20T16:50:23Z")
    assert resolution is not None
    # 8AM EDT June 23 = 12:00 UTC; close = 13:00 UTC
    assert resolution == pd.Timestamp("2025-06-23T13:00:00Z")


def test_parses_format_with_explicit_year():
    resolution = mmh.parse_candle_resolution_time(
        "bitcoin-up-or-down-august-26-2026-10am-et", "2026-08-24T14:00:09Z")
    assert resolution is not None
    # 10AM EDT Aug 26 2026 = 14:00 UTC; close = 15:00 UTC
    assert resolution == pd.Timestamp("2026-08-26T15:00:00Z")


def test_handles_dst_boundary_correctly():
    # December: ET is EST (UTC-5), not EDT (UTC-4)
    resolution = mmh.parse_candle_resolution_time(
        "bitcoin-up-or-down-december-15-9am-et", "2025-12-13T10:00:00Z")
    assert resolution is not None
    assert resolution == pd.Timestamp("2025-12-15T15:00:00Z")  # 9AM EST = 14:00 UTC, close = 15:00 UTC


def test_infers_year_rollover_for_december_created_january_market():
    # market created in late December for a candle in early January -> next year
    resolution = mmh.parse_candle_resolution_time(
        "bitcoin-up-or-down-january-2-3am-et", "2025-12-31T20:00:00Z")
    assert resolution is not None
    assert resolution.year == 2026


def test_noon_and_midnight_12_hour_clock_edge_cases():
    noon = mmh.parse_candle_resolution_time("bitcoin-up-or-down-july-4-12pm-et", "2025-07-02T10:00:00Z")
    midnight = mmh.parse_candle_resolution_time("bitcoin-up-or-down-july-4-12am-et", "2025-07-02T10:00:00Z")
    assert noon is not None and midnight is not None
    assert noon > midnight  # noon candle closes later in the day than the midnight one


def test_unparseable_slug_returns_none():
    assert mmh.parse_candle_resolution_time("bitcoin-up-or-down-on-august-25", "2025-08-22T00:00:00Z") is None
    assert mmh.parse_candle_resolution_time("some-other-market-slug", "2025-08-22T00:00:00Z") is None


def test_resolve_outcome_up_and_down():
    base = dict(event_id="1", market_id="1", slug="s", question="q", condition_id="c",
                description="d", start_date="2025-01-01T00:00:00Z", end_date="2025-01-01T01:00:00Z",
                closed=True, active=True, outcomes=["Up", "Down"],
                clob_token_ids=["a", "b"], volume=1.0, liquidity=0.0, resolution_source="binance")
    up = mmh.CandidateMarket(**{**base, "outcome_prices": ["1", "0"]})
    down = mmh.CandidateMarket(**{**base, "outcome_prices": ["0", "1"]})
    assert mmh.resolve_outcome(up) == (1, "")
    assert mmh.resolve_outcome(down) == (0, "")
