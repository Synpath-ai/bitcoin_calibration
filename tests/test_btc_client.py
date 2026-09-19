import pandas as pd
import pytest

from src import btc_client as bc


def _fake_klines(n, start_ms, step_ms):
    rows = []
    for i in range(n):
        rows.append({
            "timestamp": pd.Timestamp(start_ms + i * step_ms, unit="ms", tz="UTC"),
            "open": 50000.0, "high": 50100.0, "low": 49900.0, "close": 50050.0,
            "volume": 1.0, "quote_volume": 50000.0, "n_trades": 10,
        })
    return pd.DataFrame(rows)


def test_fetch_with_fallback_uses_coinbase_when_binance_fails(monkeypatch):
    """Regression: refresh_training_data.yml failed outright on GitHub's hosted
    runners (2026-09-10) because fetch_and_cache called fetch_klines (Binance)
    directly with no fallback, unlike the live snapshot path -- Binance 451s
    datacenter IPs including GitHub Actions runners."""
    def boom(*a, **k):
        raise RuntimeError("Binance klines fetch failed: 451 Client Error")
    monkeypatch.setattr(bc, "fetch_klines", boom)

    called = {}
    def fake_coinbase(interval, start_ms, end_ms, max_retries=3):
        called["interval"] = interval
        return _fake_klines(3, start_ms, 300_000)
    monkeypatch.setattr(bc, "fetch_coinbase_klines", fake_coinbase)

    out = bc._fetch_klines_with_fallback("BTCUSDT", "5m", 0, 900_000)
    assert called["interval"] == "5m"
    assert len(out) == 3


def test_fetch_with_fallback_reraises_for_unmapped_interval(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Binance klines fetch failed: 451 Client Error")
    monkeypatch.setattr(bc, "fetch_klines", boom)

    with pytest.raises(RuntimeError):
        bc._fetch_klines_with_fallback("BTCUSDT", "15m", 0, 900_000)


def test_fetch_and_cache_extend_path_falls_back_to_coinbase(monkeypatch, tmp_path):
    """The extend-forward branch (cache exists but is stale) must also use the
    fallback, not just the first-ever-fetch branch -- this is the exact branch
    refresh_training_data.yml actually hits every day."""
    monkeypatch.setattr(bc, "RAW_DIR", tmp_path)
    cache_path = tmp_path / "btcusdt_5m.parquet"
    old = _fake_klines(2, 0, 300_000)
    old.to_parquet(cache_path)

    def boom(*a, **k):
        raise RuntimeError("451")
    monkeypatch.setattr(bc, "fetch_klines", boom)

    def fake_coinbase(interval, start_ms, end_ms, max_retries=3):
        return _fake_klines(2, start_ms, 300_000)
    monkeypatch.setattr(bc, "fetch_coinbase_klines", fake_coinbase)

    end = pd.Timestamp(2_000_000, unit="ms", tz="UTC")
    out = bc.fetch_and_cache("BTCUSDT", "5m", pd.Timestamp(0, unit="ms", tz="UTC"), end, cache_name="btcusdt_5m")
    assert len(out) == 4   # 2 cached + 2 from the Coinbase fallback
