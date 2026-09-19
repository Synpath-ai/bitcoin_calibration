import numpy as np
import pandas as pd

from src import observation_builder as ob


def _make_btc_df(start, n, freq_minutes=5, base_price=100.0, spike_at=None, spike_price=None):
    ts = pd.date_range(start, periods=n, freq=f"{freq_minutes}min", tz="UTC")
    close = np.full(n, base_price)
    if spike_at is not None:
        close[spike_at:] = spike_price
    return pd.DataFrame({"timestamp": ts, "open": close, "high": close, "low": close,
                          "close": close, "volume": np.full(n, 10.0), "quote_volume": np.full(n, 1000.0),
                          "n_trades": np.full(n, 5)})


def test_last_value_as_of_never_returns_future_value():
    ts = np.array([0.0, 100.0, 200.0, 300.0])
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    # as_of exactly between two points -> should return the earlier one, never later
    v, staleness = ob._last_value_as_of(ts, vals, 250.0, max_staleness_sec=1000)
    assert v == 3.0
    assert staleness == 50.0


def test_last_value_as_of_respects_staleness_tolerance():
    ts = np.array([0.0, 100.0])
    vals = np.array([1.0, 2.0])
    # as_of far beyond tolerance from the last observed point -> dropped
    v, staleness = ob._last_value_as_of(ts, vals, 100.0 + 10_000, max_staleness_sec=60)
    assert v is None and staleness is None


def test_last_value_as_of_before_any_data_returns_none():
    ts = np.array([500.0, 600.0])
    vals = np.array([1.0, 2.0])
    v, staleness = ob._last_value_as_of(ts, vals, 100.0, max_staleness_sec=1000)
    assert v is None


def test_btc_features_never_leak_a_future_price_spike():
    """A price spike occurring AFTER the observation timestamp must not affect
    btc_spot_price or any of the trailing-return features computed at that ts."""
    btc = _make_btc_df("2025-01-01T00:00:00Z", 40, base_price=100.0, spike_at=30, spike_price=999.0)
    btc["_epoch"] = btc["timestamp"].astype("datetime64[ns, UTC]").astype("int64") / 1e9

    # observation timestamp is candle #20 (well before the spike at #30)
    obs_epoch = btc["_epoch"].iloc[20]
    feat = ob._btc_features(btc, obs_epoch)
    assert feat["btc_spot_price"] == 100.0
    assert feat["btc_return_15m"] == 0.0  # flat prior to spike


def test_btc_features_staleness_drops_when_no_recent_candle():
    btc = _make_btc_df("2025-01-01T00:00:00Z", 10, freq_minutes=5, base_price=100.0)
    btc["_epoch"] = btc["timestamp"].astype("datetime64[ns, UTC]").astype("int64") / 1e9
    far_future_epoch = btc["_epoch"].iloc[-1] + 3600  # 1 hour after last candle
    feat = ob._btc_features(btc, far_future_epoch)
    assert np.isnan(feat["btc_spot_price"])


def test_horizon_before_market_open_is_excluded():
    market = {
        "market_id": "1", "slug": "test-slug", "condition_id": "c",
        "start_date": "2025-06-01T10:00:00Z", "end_date": "2025-06-01T12:00:00Z",  # only 2h window
        "outcome_up": 1, "volume": 1000.0, "liquidity": 0.0,
        "clob_token_ids": ["a", "b"],
    }
    btc = _make_btc_df("2025-05-30T00:00:00Z", 2000, base_price=100.0)

    import json
    from pathlib import Path
    price_dir = ob.PRICE_DIR
    price_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = price_dir / "test-slug.json"
    ts0 = pd.Timestamp("2025-06-01T10:00:00Z").timestamp()
    fixture_path.write_text(json.dumps([{"t": ts0, "p": 0.5}, {"t": ts0 + 3600, "p": 0.6}]))
    try:
        obs_df, exclusions = ob.build_observations([market], btc, use_live_gdelt=False)
        # 24h, 12h, 6h horizons all precede the 2h-wide market -> excluded
        reasons = {e["horizon_hours"] for e in exclusions if e["reason"] == "before_market_open"}
        assert 24 in reasons and 12 in reasons and 6 in reasons
        assert set(obs_df["horizon_hours"]) <= {1}
    finally:
        fixture_path.unlink(missing_ok=True)
